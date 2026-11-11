import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name, default=None):
    v = os.getenv(name)
    return v if v not in (None, "") else default


INVIPO_BASE_URL = _env("INVIPO_BASE_URL", "http://5.28.4.81").rstrip("/")
INVIPO_EMAIL = _env("INVIPO_EMAIL")
INVIPO_PASSWORD = _env("INVIPO_PASSWORD")
# catalog camera/junction cache (giây)
CATALOG_TTL_S = int(_env("CATALOG_TTL_S", "600"))

# chỉ cho phép backend gọi tới các host nội bộ (chống SSRF)
ALLOWED_SNAPSHOT_NETS = [n.strip() for n in _env("ALLOWED_SNAPSHOT_NETS", "5.28.0.0/16").split(",")]
SNAPSHOT_TIMEOUT_S = float(_env("SNAPSHOT_TIMEOUT_S", "10"))
SNAPSHOT_RETRIES = int(_env("SNAPSHOT_RETRIES", "1"))
# ảnh cache Invipo cũ hơn ngưỡng này thì cảnh báo
SNAPSHOT_STALE_S = int(_env("SNAPSHOT_STALE_S", "600"))

MODEL_PATH = _env("MODEL_PATH", str(ROOT / "models" / "yolo26x.pt"))
DEVICE = _env("DEVICE")  # None = auto (mps/cuda/cpu)
IMGSZ = int(_env("IMGSZ", "1280"))
CONF = float(_env("CONF", "0.25"))
IOU = float(_env("IOU", "0.6"))
# tiled inference: thêm 2x2 tile để bắt xe máy nhỏ ở xa
TILED = _env("TILED", "true").lower() == "true"
TILE_GRID = _env("TILE_GRID", "2x2")  # rows x cols
# EXPERIMENTAL: suy ra xe máy bị che từ người lái (xem counting.py)
RIDER_RECOVERY = _env("RIDER_RECOVERY", "true").lower() == "true"

DATA_DIR = Path(_env("DATA_DIR", str(ROOT / "data")))
ROI_DIR = DATA_DIR / "roi"
RESULTS_DIR = DATA_DIR / "results"
MAX_RESULTS = int(_env("MAX_RESULTS", "200"))
