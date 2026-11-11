"""Phân lớp nghiệp vụ, lọc ROI, đếm.

Chính sách phân lớp (1 bảng duy nhất, tổng = tổng các lớp):
  motorcycle <- COCO motorcycle      car   <- COCO car (gồm SUV, taxi, van nhỏ, pickup)
  bus        <- COCO bus             truck <- COCO truck
  other      <- COCO bicycle
  person không phải xe. Riêng khi bật RIDER_RECOVERY (EXPERIMENTAL): người không nằm trong ô tô
  và không chồng lên xe máy/xe đạp nào đã phát hiện -> coi là người lái xe máy bị che mất xe.
  Người chồng lên nhau (người ngồi sau) gộp thành 1 xe. Lớp ra: motorcycle, inferred=True.
  Rủi ro: người đi bộ dưới lòng đường bị tính là xe máy -> ROI phải loại vỉa hè.

Điểm neo ROI: bottom-center (cx, y2) của bbox = điểm bánh xe chạm đường. Với camera góc cao,
tâm bbox của xe cao (bus/truck) chiếu lệch lên làn/ROI phía trên, còn điểm chạm đất thì không.
Thuộc ROI theo ray-casting (even-odd) trên toạ độ chuẩn hoá 0..1, tính 1 lần, không phụ thuộc resize.

Tổng toàn ảnh = số xe có điểm neo nằm trong ÍT NHẤT 1 ROI (mỗi xe tính 1 lần dù ROI chồng nhau).
Không có ROI -> dùng toàn khung hình (roiMode = full_frame).
"""
from dataclasses import dataclass, field

from . import config

CLASS_MAP = {"motorcycle_rider": "motorcycle", "motorcycle": "motorcycle", "car": "car", "bus": "bus", "truck": "truck", "bicycle": "other"}
BUSINESS_CLASSES = ["motorcycle", "car", "bus", "truck", "other"]
# ngưỡng conf theo lớp (sau ngưỡng chung của model)
PERSON_CONF = 0.35
CLASS_CONF = {"motorcycle": 0.25, "car": 0.30, "bus": 0.35, "truck": 0.35, "other": 0.35}

FULL_FRAME = {"id": "full_frame", "name": "Toàn khung hình", "type": "count",
              "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}


@dataclass
class Counted:
    det: object
    cls: str
    anchor: tuple  # (x, y) chuẩn hoá
    rois: list
    inferred: bool = False


def point_in_polygon(x, y, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xin:
                inside = not inside
    return inside


def anchor_of(det, w, h):
    # kẹp vào trong khung: xe chạm mép dưới ảnh có y=1.0, ray-casting sẽ coi là nằm ngoài ROI chạm mép
    x = (det.x1 + det.x2) / 2 / w
    y = det.y2 / h
    return (min(max(x, 0.0005), 0.9995), min(max(y, 0.0005), 0.9995))


def validate_rois(rois):
    """Trả về list lỗi; rỗng = hợp lệ."""
    errs = []
    ids = set()
    for i, r in enumerate(rois):
        p = r.get("polygon") or []
        if len(p) < 3:
            errs.append(f"roi[{i}] cần ít nhất 3 điểm")
        if any(not (isinstance(pt, (list, tuple)) and len(pt) == 2 and all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in pt)) for pt in p):
            errs.append(f"roi[{i}] toạ độ phải chuẩn hoá trong [0,1]")
        if r.get("type", "count") not in ("count", "queue"):
            errs.append(f"roi[{i}] type phải là count hoặc queue")
        if not r.get("id") or r["id"] in ids:
            errs.append(f"roi[{i}] id rỗng hoặc trùng")
        ids.add(r.get("id"))
    return errs


def _ios(a, b):
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    return ix * iy / max(1e-6, min(a.area, b.area))


def recover_riders(persons, vehicles, img_h=None):
    """Người không thuộc xe nào -> nhóm lại -> mỗi nhóm = 1 xe máy suy ra (box hợp các người)."""
    from .detector import Detection
    four = [v for v in vehicles if v.raw_class in ("car", "bus", "truck")]
    two = [v for v in vehicles if v.raw_class in ("motorcycle", "bicycle")]
    free = []
    for p in persons:
        if img_h and p.y2 >= img_h - 3:
            continue  # bị cắt ở mép dưới ảnh, không thấy xe -> không suy đoán
        if any(_ios(p, v) >= 0.5 for v in four):
            continue  # tài xế/hành khách nhìn qua kính
        if any(_ios(p, v) >= 0.2 for v in two):
            continue  # đã có xe máy của người này
        free.append(p)
    # gộp người lái + người ngồi sau
    groups = []
    for p in sorted(free, key=lambda d: d.conf, reverse=True):
        for g in groups:
            if any(_ios(p, q) >= 0.45 for q in g):
                g.append(p)
                break
        else:
            groups.append([p])
    return [Detection(min(q.x1 for q in g), min(q.y1 for q in g), max(q.x2 for q in g), max(q.y2 for q in g),
                      max(q.conf for q in g), "motorcycle_rider") for g in groups]


def count(dets, w, h, rois, rider_recovery=None):
    rider_recovery = config.RIDER_RECOVERY if rider_recovery is None else rider_recovery
    rois = rois or []
    roi_mode = "configured" if rois else "full_frame"
    active = rois or [FULL_FRAME]

    vehicles, persons, rejected = [], [], 0
    for d in dets:
        if d.raw_class == "person":
            if d.conf >= PERSON_CONF:
                persons.append(d)
            continue
        cls = CLASS_MAP.get(d.raw_class)
        if cls is None or d.conf < CLASS_CONF[cls]:
            rejected += 1
            continue
        vehicles.append(d)
    if rider_recovery:
        vehicles += recover_riders(persons, vehicles, h)

    counted = []
    for d in vehicles:
        ax, ay = anchor_of(d, w, h)
        inside = [r["id"] for r in active if point_in_polygon(ax, ay, r["polygon"])]
        counted.append(Counted(d, CLASS_MAP[d.raw_class], (ax, ay), inside, d.raw_class == "motorcycle_rider"))

    in_any = [c for c in counted if c.rois]
    by_class = {k: 0 for k in BUSINESS_CLASSES}
    for c in in_any:
        by_class[c.cls] += 1

    per_roi = []
    for r in active:
        cs = {k: 0 for k in BUSINESS_CLASSES}
        for c in in_any:
            if r["id"] in c.rois:
                cs[c.cls] += 1
        per_roi.append({"id": r["id"], "name": r.get("name") or r["id"], "type": r.get("type", "count"),
                        "total": sum(cs.values()), "byClass": cs})

    return {
        "roiMode": roi_mode,
        "total": len(in_any),
        "byClass": by_class,
        "rois": per_roi,
        "motorcycleFromRider": sum(1 for c in in_any if c.inferred),
        "riderRecovery": bool(rider_recovery),
        "outsideRoi": len(counted) - len(in_any),
        "belowThreshold": rejected,
    }, counted
