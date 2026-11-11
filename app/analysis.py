"""Luồng chính: snapshot -> detect -> ROI count -> ảnh annotate -> lưu kết quả."""
import json
import logging
import shutil
import time
import uuid
from datetime import datetime, timezone

import cv2
import numpy as np

from . import annotate, config, roi_store
from .counting import count
from .snapshot import SnapshotError

log = logging.getLogger("analysis")

LIMITATIONS = [
    "Đếm trên 1 ảnh tĩnh: không đo được tốc độ, xe dừng hay chạy, thời gian chờ, lưu lượng/phút.",
    "'Queue zone' = số xe đang nằm trong vùng hàng chờ đã cấu hình, KHÔNG phải số xe đang đứng chờ.",
    "Xe bị che khuất hoàn toàn (sau xe buýt, cây) không thể đếm; xe che một phần có thể bị sót.",
]


class AnalysisError(Exception):
    def __init__(self, message, code):
        super().__init__(message)
        self.code = code


def quality_warnings(img, summary, counted, snap):
    w = []
    if not snap.is_live:
        w.append({"code": "cached_snapshot", "message": snap.note or "Ảnh không phải live"})
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.mean() < 50:
        w.append({"code": "low_light", "message": "Ảnh tối, độ chính xác giảm"})
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 40:
        w.append({"code": "blurry", "message": "Ảnh mờ/nhoè, độ chính xác giảm"})
    if gray.std() < 12:
        w.append({"code": "flat_image", "message": "Ảnh gần như đồng màu (camera lỗi/che ống kính?)"})

    inside = [c for c in counted if c.rois]
    h = img.shape[0]
    if inside:
        small = sum(1 for c in inside if (c.det.y2 - c.det.y1) < 0.035 * h)
        if small / len(inside) > 0.3:
            w.append({"code": "small_objects", "message": f"{small} xe rất nhỏ (ở xa) - dễ sót hoặc nhầm lớp"})
        # tỉ lệ cặp bbox chồng lấn -> che khuất
        boxes = [c.det for c in inside]
        overl = sum(1 for i, a in enumerate(boxes) for b in boxes[i + 1:]
                    if min(a.x2, b.x2) > max(a.x1, b.x1) and min(a.y2, b.y2) > max(a.y1, b.y1))
        if summary["byClass"]["motorcycle"] >= 25 or overl > 0.6 * len(boxes):
            w.append({"code": "dense_occlusion",
                      "message": "Mật độ cao/che khuất nhiều: số xe máy thực tế có thể cao hơn số đếm"})
    return w


class AnalysisService:
    def __init__(self, snapshots, detector):
        self.snapshots = snapshots
        self.detector = detector

    def analyze(self, camera_id, camera_name=None):
        t0 = time.time()
        try:
            snap = self.snapshots.get_snapshot(camera_id)
        except SnapshotError as e:
            raise AnalysisError(str(e), e.code)
        t_snap = time.time() - t0

        img = cv2.imdecode(np.frombuffer(snap.image, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise AnalysisError("Ảnh snapshot không giải mã được", "invalid_image")
        h, w = img.shape[:2]

        t1 = time.time()
        try:
            dets = self.detector.detect(img)
        except Exception as e:  # lỗi model/thiết bị
            log.exception("Inference lỗi")
            raise AnalysisError(f"Inference lỗi: {type(e).__name__}", "inference_failed")
        t_inf = time.time() - t1

        rois = roi_store.load(camera_id)["rois"]
        summary, counted = count(dets, w, h, rois)
        warnings = quality_warnings(img, summary, counted, snap)

        rid = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
        out_dir = config.RESULTS_DIR / rid
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "source.jpg").write_bytes(snap.image)  # giữ nguyên ảnh gốc, tách riêng ảnh vẽ
        header = f"{snap.source.upper()} {snap.timestamp[:19].replace('T', ' ')}Z"
        drawn = annotate.render(img, counted, summary, rois or [], header)
        cv2.imwrite(str(out_dir / "annotated.jpg"), drawn, [cv2.IMWRITE_JPEG_QUALITY, 88])

        result = {
            "resultId": rid,
            "cameraId": camera_id,
            "cameraName": camera_name,
            "snapshot": {"timestamp": snap.timestamp, "source": snap.source, "isLive": snap.is_live,
                         "width": w, "height": h, "note": snap.note},
            "counts": summary,
            "detections": [{
                "class": c.cls, "rawClass": c.det.raw_class, "conf": round(c.det.conf, 3),
                "box": [round(c.det.x1 / w, 4), round(c.det.y1 / h, 4), round(c.det.x2 / w, 4), round(c.det.y2 / h, 4)],
                "anchor": [round(c.anchor[0], 4), round(c.anchor[1], 4)], "rois": c.rois,
            } for c in counted],
            "roiConfig": rois,
            "warnings": warnings,
            "limitations": LIMITATIONS,
            "model": self.detector.info,
            "timingMs": {"snapshot": round(t_snap * 1000), "inference": round(t_inf * 1000),
                         "total": round((time.time() - t0) * 1000)},
            "images": {"source": f"/api/results/{rid}/source.jpg", "annotated": f"/api/results/{rid}/annotated.jpg"},
        }
        (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        self._prune()
        log.info("Phân tích %s: %d xe, nguồn %s, inference %dms.", camera_id, summary["total"], snap.source,
                 result["timingMs"]["inference"])
        return result

    def _prune(self):
        dirs = sorted(p for p in config.RESULTS_DIR.iterdir() if p.is_dir())
        for p in dirs[:-config.MAX_RESULTS]:
            shutil.rmtree(p, ignore_errors=True)
