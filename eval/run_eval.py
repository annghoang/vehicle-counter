"""Đánh giá đếm xe trên tập ảnh đã đếm tay.

Chạy: .venv/bin/python -m eval.run_eval
Ghi: eval/report.json + eval/output/<config>/<file>.jpg
"""
import json
import time
from pathlib import Path

import cv2

from app import annotate
from app.counting import BUSINESS_CLASSES, count
from app.detector import YoloDetector

ROOT = Path(__file__).parent
CONFIGS = [
    ("yolo26m", False), ("yolo26m", True),
    ("yolo26x", False), ("yolo26x", True),
]


def main():
    gt = json.loads((ROOT / "ground_truth.json").read_text())["images"]
    report = []
    for model, tiled in CONFIGS:
        det = YoloDetector(model_path=str(ROOT.parent / "models" / f"{model}.pt"), tiled=tiled, imgsz=1280)
        det.detect(cv2.imread(str(ROOT / "images" / gt[0]["file"])))  # warm-up
        raw, lat = {}, []
        for g in gt:
            img = cv2.imread(str(ROOT / "images" / g["file"]))
            t = time.time()
            raw[g["file"]] = det.detect(img)
            lat.append(time.time() - t)
        for rider in (False, True):
            name = f"{model}{'-tiled2x2' if tiled else ''}{'-rider' if rider else ''}"
            out_dir = ROOT / "output" / name
            out_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            for g in gt:
                img = cv2.imread(str(ROOT / "images" / g["file"]))
                h, w = img.shape[:2]
                roi = [{"id": "eval", "name": "eval", "type": "count", "polygon": g["roi"]}]
                s, counted = count(raw[g["file"]], w, h, roi, rider_recovery=rider)
                cv2.imwrite(str(out_dir / g["file"]), annotate.render(img, counted, s, roi, name),
                            [cv2.IMWRITE_JPEG_QUALITY, 80])
                gt_total = sum(g["counts"].values())
                rows.append({"file": g["file"], "scene": g["scene"], "gtTotal": gt_total, "predTotal": s["total"],
                             "absErr": abs(s["total"] - gt_total),
                             "classErr": {k: s["byClass"][k] - g["counts"].get(k, 0) for k in BUSINESS_CLASSES},
                             "fromRider": s["motorcycleFromRider"]})
            dense = [r for r in rows if r["scene"].startswith("dense")]
            normal = [r for r in rows if not r["scene"].startswith("dense")]
            mae = lambda rs: round(sum(r["absErr"] for r in rs) / len(rs), 2) if rs else None
            per_class = {k: round(sum(abs(r["classErr"][k]) for r in rows) / len(rows), 2) for k in BUSINESS_CLASSES}
            rep = {"config": name, "maeAll": mae(rows), "maeNonDense": mae(normal), "maeDense": mae(dense),
                   "perClassMAE": per_class, "latencyMsMean": round(1000 * sum(lat) / len(lat)),
                   "latencyMsMax": round(1000 * max(lat)), "rows": rows}
            report.append(rep)
            print(f"{name:28s} MAE all={rep['maeAll']:5} nonDense={rep['maeNonDense']:5} dense={rep['maeDense']:5} "
                  f"perClass={per_class} lat={rep['latencyMsMean']}ms")
    (ROOT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
