"""ROI theo camera, lưu file JSON data/roi/<cameraId>.json. Toạ độ chuẩn hoá 0..1."""
import json
import re
from datetime import datetime, timezone

from . import config
from .counting import validate_rois

_SAFE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class RoiError(ValueError):
    pass


def _path(camera_id):
    if not _SAFE.match(camera_id or ""):
        raise RoiError("cameraId không hợp lệ")
    return config.ROI_DIR / f"{camera_id}.json"


def load(camera_id):
    p = _path(camera_id)
    if not p.exists():
        return {"cameraId": camera_id, "rois": [], "updatedAt": None}
    return json.loads(p.read_text(encoding="utf-8"))


def save(camera_id, rois):
    errs = validate_rois(rois)
    if errs:
        raise RoiError("; ".join(errs))
    clean = [{"id": r["id"], "name": (r.get("name") or r["id"])[:60], "type": r.get("type", "count"),
              "polygon": [[round(float(x), 5), round(float(y), 5)] for x, y in r["polygon"]]} for r in rois]
    doc = {"cameraId": camera_id, "rois": clean, "updatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    config.ROI_DIR.mkdir(parents=True, exist_ok=True)
    _path(camera_id).write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return doc
