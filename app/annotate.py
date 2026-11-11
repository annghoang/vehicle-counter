"""Vẽ ảnh kết quả: ROI, bbox, nhãn lớp, chấm điểm neo (xe được đếm), bảng đếm."""
import cv2
import numpy as np

# màu BGR
COLORS = {"motorcycle": (0, 200, 255), "car": (80, 220, 60), "bus": (255, 120, 0),
          "truck": (200, 60, 200), "other": (180, 180, 180)}
LABELS = {"motorcycle": "xe may", "car": "o to", "bus": "bus", "truck": "tai", "other": "khac"}
ROI_COLORS = {"count": (255, 255, 0), "queue": (0, 140, 255)}


def render(img, counted, summary, rois, header):
    out = img.copy()
    h, w = out.shape[:2]
    s = max(1.0, w / 1920)
    th = max(1, int(2 * s))

    # ROI: viền + nền mờ
    overlay = out.copy()
    for r in rois:
        pts = (np.array(r["polygon"]) * [w, h]).astype(np.int32)
        col = ROI_COLORS.get(r.get("type", "count"), (255, 255, 0))
        cv2.fillPoly(overlay, [pts], col)
    cv2.addWeighted(overlay, 0.12, out, 0.88, 0, out)
    for r in rois:
        pts = (np.array(r["polygon"]) * [w, h]).astype(np.int32)
        col = ROI_COLORS.get(r.get("type", "count"), (255, 255, 0))
        cv2.polylines(out, [pts], True, col, th + 1)
        tx, ty = pts[:, 0].min(), pts[:, 1].min()
        cv2.putText(out, r.get("name") or r["id"], (int(tx) + 6, int(ty) + int(28 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, col, th, cv2.LINE_AA)

    # bbox + chấm: xe ngoài ROI vẽ mờ, không chấm
    for c in counted:
        d, col = c.det, COLORS[c.cls]
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        if not c.rois:
            cv2.rectangle(out, p1, p2, (120, 120, 120), 1)
            continue
        cv2.rectangle(out, p1, p2, col, th)
        lbl = f"{LABELS[c.cls]}{'*' if c.inferred else ''} {d.conf:.2f}"
        (tw, tht), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, 1)
        cv2.rectangle(out, (p1[0], p1[1] - tht - 4), (p1[0] + tw + 4, p1[1]), col, -1)
        cv2.putText(out, lbl, (p1[0] + 2, p1[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, (0, 0, 0), 1, cv2.LINE_AA)
        ax, ay = int(c.anchor[0] * w), int(c.anchor[1] * h)
        cv2.circle(out, (ax, ay), int(6 * s), (0, 0, 0), -1)
        cv2.circle(out, (ax, ay), int(4 * s), col, -1)

    # bảng đếm góc trái dưới
    lines = [header, f"Tong: {summary['total']}"]
    lines += [f"{LABELS[k]}: {v}" for k, v in summary["byClass"].items()]
    if summary["roiMode"] == "configured":
        lines += [f"[{r['type']}] {r['name']}: {r['total']}" for r in summary["rois"]]
    fs, lh = 0.7 * s, int(30 * s)
    bw = int(max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0][0] for l in lines) + 24 * s)
    bh = lh * len(lines) + int(14 * s)
    y0 = h - bh - int(70 * s)
    sub = out[y0:y0 + bh, 10:10 + bw]
    out[y0:y0 + bh, 10:10 + bw] = (sub * 0.35).astype(np.uint8)
    for i, l in enumerate(lines):
        cv2.putText(out, l, (22, y0 + lh * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
    return out
