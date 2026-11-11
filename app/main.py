import logging
import re

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, roi_store
from .analysis import AnalysisError, AnalysisService
from .detector import YoloDetector
from .invipo import InvipoCatalog, InvipoError
from .snapshot import SnapshotError, SnapshotProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # httpx log cả URL (có session VMS)
log = logging.getLogger("api")

STATUS = {"camera_not_found": 404, "invalid_camera_config": 422,
          "camera_offline": 503, "snapshot_unavailable": 503, "invalid_image": 502,
          "inference_failed": 500}
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

catalog = InvipoCatalog()
snapshots = SnapshotProvider(catalog)
detector = YoloDetector()
service = AnalysisService(snapshots, detector)

app = FastAPI(title="Vehicle Counter (snapshot)", version="0.1.0")


def _catalog():
    try:
        catalog.refresh()
    except InvipoError as e:
        raise HTTPException(503, {"error": str(e), "code": "invipo_unavailable"})
    except Exception as e:
        raise HTTPException(503, {"error": f"Không kết nối được Invipo: {type(e).__name__}", "code": "invipo_unavailable"})
    return catalog


def _check_id(camera_id):
    if not SAFE_ID.match(camera_id):
        raise HTTPException(422, {"error": "cameraId không hợp lệ", "code": "invalid_camera_id"})


@app.get("/api/health")
def health():
    return {"ok": True, "model": detector.info, "camerasLoaded": len(catalog.cameras)}


@app.get("/api/junctions")
def junctions(q: str | None = None):
    cat = _catalog()
    items = [j for j in cat.junctions.values() if j["cameraCount"] > 0]
    if q:
        items = [j for j in items if q.lower() in f"j{j['junction']} {j['name']}".lower()]
    return {"items": sorted(items, key=lambda j: j["junction"])}


@app.get("/api/cameras")
def cameras(junction: str | None = None, q: str | None = None):
    cat = _catalog()
    cams = list(cat.cameras.values())
    if junction:
        m = re.search(r"\d+", junction)
        if not m:
            raise HTTPException(422, {"error": "junction phải chứa số, vd J196 hoặc 196", "code": "invalid_junction"})
        n = int(m.group())
        cams = [c for c in cams if n in c.junctions]
    if q:
        cams = [c for c in cams if q.lower() in c.name.lower()]
    return {"items": [c.public() for c in sorted(cams, key=lambda c: c.name)]}


@app.get("/api/cameras/{camera_id}/snapshot")
def camera_snapshot(camera_id: str):
    """Ảnh gốc để vẽ ROI (không chạy AI)."""
    _check_id(camera_id)
    _catalog()
    try:
        s = snapshots.get_snapshot(camera_id)
    except SnapshotError as e:
        raise HTTPException(STATUS.get(e.code, 503), {"error": str(e), "code": e.code})
    return Response(s.image, media_type="image/jpeg", headers={
        "X-Snapshot-Source": s.source, "X-Snapshot-Timestamp": s.timestamp, "Cache-Control": "no-store"})


@app.get("/api/cameras/{camera_id}/roi")
def get_roi(camera_id: str):
    _check_id(camera_id)
    return roi_store.load(camera_id)


class RoiBody(BaseModel):
    rois: list[dict]


@app.put("/api/cameras/{camera_id}/roi")
def put_roi(camera_id: str, body: RoiBody):
    _check_id(camera_id)
    try:
        return roi_store.save(camera_id, body.rois)
    except roi_store.RoiError as e:
        raise HTTPException(422, {"error": str(e), "code": "invalid_roi"})


class AnalyzeBody(BaseModel):
    cameraId: str


@app.post("/api/analyze")
def analyze(body: AnalyzeBody):
    _check_id(body.cameraId)
    cam = _catalog().cameras.get(body.cameraId)
    try:
        return service.analyze(body.cameraId, camera_name=cam.name if cam else None)
    except AnalysisError as e:
        raise HTTPException(STATUS.get(e.code, 500), {"error": str(e), "code": e.code})


@app.get("/api/results/{rid}")
def get_result(rid: str):
    p = _result_path(rid, "result.json")
    return Response(p.read_bytes(), media_type="application/json")


@app.get("/api/results/{rid}/{kind}.jpg")
def get_result_image(rid: str, kind: str):
    if kind not in ("source", "annotated"):
        raise HTTPException(404, {"error": "Không tìm thấy ảnh"})
    return FileResponse(_result_path(rid, f"{kind}.jpg"), media_type="image/jpeg")


def _result_path(rid, name):
    if not re.match(r"^[0-9T]{15}-[0-9a-f]{6}$", rid):
        raise HTTPException(404, {"error": "Không tìm thấy kết quả"})
    p = config.RESULTS_DIR / rid / name
    if not p.exists():
        raise HTTPException(404, {"error": "Không tìm thấy kết quả"})
    return p


app.mount("/", StaticFiles(directory=config.ROOT / "static", html=True), name="static")
