"""Detector xe. Giao diện chung: detect(image_bgr) -> list[Detection] (toạ độ pixel ảnh gốc).

YoloDetector dùng Ultralytics (YOLO26 COCO pretrained). Thay model khác chỉ cần viết class
có cùng hàm detect() và trả về raw_class theo tên lớp của model.
"""
import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from . import config

log = logging.getLogger("detector")


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    raw_class: str  # tên lớp của model (vd "motorcycle")

    @property
    def area(self):
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


# lớp COCO quan tâm; person giữ lại để bù xe máy bị che (xem counting.recover_riders)
VEHICLE_RAW_CLASSES = {"bicycle", "car", "motorcycle", "bus", "truck", "person"}
FOUR_WHEEL = {"car", "bus", "truck"}


def _iou_ios(a, b):
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    if inter <= 0:
        return 0.0, 0.0
    union = a.area + b.area - inter
    return inter / union, inter / max(1e-6, min(a.area, b.area))


def merge_detections(dets, iou_thr=0.55, ios_thr=0.8):
    """Gộp kết quả full-frame + tile.
    - IoU >= iou_thr (không phân biệt lớp): coi là cùng 1 xe, giữ box conf cao hơn.
    - IoS >= ios_thr cùng lớp: box bị tile cắt cụt nằm trong box đầy đủ -> bỏ box nhỏ.
    """
    dets = sorted(dets, key=lambda d: d.conf, reverse=True)
    keep = []
    for d in dets:
        dup = False
        for i, k in enumerate(keep):
            if (d.raw_class == "person") != (k.raw_class == "person"):
                continue  # người và xe không triệt tiêu nhau
            iou, ios = _iou_ios(d, k)
            if iou >= iou_thr:
                dup = True
                break
            same = d.raw_class == k.raw_class or (d.raw_class in FOUR_WHEEL and k.raw_class in FOUR_WHEEL)
            thr = ios_thr if d.raw_class == k.raw_class else 0.6
            if same and ios >= thr:  # xe lớn bị tile tách thành cabin/thùng (truck vs bus) -> giữ box lớn
                if d.area > k.area:  # giữ box lớn (đầy đủ), bỏ box bị cắt
                    keep[i] = Detection(d.x1, d.y1, d.x2, d.y2, d.conf, d.raw_class)
                dup = True
                break
        if not dup:
            keep.append(d)
    return keep


def drop_cut_boxes(dets, win, w, h, margin=4):
    """Bỏ box trong tile chạm cạnh tile nằm bên trong ảnh (xe bị cắt, full-frame/tile kề sẽ bắt đủ)."""
    x1, y1, x2, y2 = win
    out = []
    for d in dets:
        cut = ((x1 > 0 and d.x1 <= x1 + margin) or (y1 > 0 and d.y1 <= y1 + margin) or
               (x2 < w and d.x2 >= x2 - margin) or (y2 < h and d.y2 >= y2 - margin))
        if not cut:
            out.append(d)
    return out


def tile_windows(w, h, rows=2, cols=2, overlap=0.15):
    tw, th = int(w / (cols - (cols - 1) * overlap)), int(h / (rows - (rows - 1) * overlap))
    xs = np.linspace(0, w - tw, cols).astype(int)
    ys = np.linspace(0, h - th, rows).astype(int)
    return [(int(x), int(y), int(x) + tw, int(y) + th) for y in ys for x in xs]


class YoloDetector:
    name = "ultralytics-yolo"

    def __init__(self, model_path=None, device=None, imgsz=None, conf=None, iou=None, tiled=None, grid=None):
        self.model_path = model_path or config.MODEL_PATH
        self.device = device or config.DEVICE
        self.imgsz = imgsz or config.IMGSZ
        self.conf = conf if conf is not None else config.CONF
        self.iou = iou if iou is not None else config.IOU
        self.tiled = config.TILED if tiled is None else tiled
        self.grid = grid or tuple(int(x) for x in config.TILE_GRID.lower().split("x"))  # (rows, cols)
        self._model = None
        self._lock = threading.Lock()  # 1 inference 1 lúc, tránh tranh chấp GPU/MPS

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO
            t = time.time()
            self._model = YOLO(self.model_path)
            log.info("Nạp model %s (%.1fs).", self.model_path, time.time() - t)
        return self._model

    @property
    def info(self):
        return {"name": self.name, "weights": str(self.model_path).split("/")[-1], "imgsz": self.imgsz,
                "conf": self.conf, "iou": self.iou, "tiled": self.tiled,
                "tileGrid": f"{self.grid[0]}x{self.grid[1]}" if self.tiled else None}

    def _predict(self, img, ox=0, oy=0, imgsz=None):
        model = self._load()
        res = model.predict(img, imgsz=imgsz or self.imgsz, conf=self.conf, iou=self.iou,
                            device=self.device, verbose=False, agnostic_nms=False)[0]
        names = res.names
        out = []
        for (x1, y1, x2, y2), c, k in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()):
            name = names[int(k)]
            if name in VEHICLE_RAW_CLASSES:
                out.append(Detection(x1 + ox, y1 + oy, x2 + ox, y2 + oy, float(c), name))
        return out

    def detect(self, img):
        with self._lock:
            dets = self._predict(img)
            if self.tiled:
                h, w = img.shape[:2]
                for win in tile_windows(w, h, *self.grid):
                    x1, y1, x2, y2 = win
                    part = self._predict(np.ascontiguousarray(img[y1:y2, x1:x2]), x1, y1)
                    dets += drop_cut_boxes(part, win, w, h)
            # gộp cả khi không tile: 1 xe có thể ra cả car + truck
            return merge_detections(dets)
