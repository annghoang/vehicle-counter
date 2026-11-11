"""Invipo client: đăng nhập, tải danh sách camera AC + junction.

Camera AC = item Invipo có power `TrafficLoadDetection` (đúng lớp "Loads" trên map).
Junction của camera xác định theo thứ tự ưu tiên:
  1. area TrafficLoads tên "N. ..." có arm trỏ tới camera  -> reason "loads_area"
  2. controller J{N} gần nhất, trong bán kính JUNCTION_RADIUS_M -> reason "near_controller"
  3. mã C{NNN} trong tên camera                             -> reason "name_code"
"""
import logging
import math
import re
import threading
import time
from dataclasses import dataclass, field

import httpx

from . import config

log = logging.getLogger("invipo")

JUNCTION_RADIUS_M = 120


@dataclass
class Camera:
    id: str
    name: str
    model: str
    ip: str | None
    status: str | None
    location: list | None
    stream_url: str | None = field(repr=False, default=None)  # có session VMS, không log
    snapshot_url: str | None = field(repr=False, default=None)
    snapshot_ts: str | None = None
    junctions: dict = field(default_factory=dict)  # {196: {"reason":..., "distanceM":...}}

    def public(self):
        return {
            "id": self.id,
            "name": self.name,
            "model": self.model,
            "status": self.status,
            "location": self.location,
            "junctions": [{"junction": j, **v} for j, v in sorted(self.junctions.items())],
        }


class InvipoError(Exception):
    pass


def _dist_m(a, b):
    return math.hypot((a[0] - b[0]) * 104000, (a[1] - b[1]) * 111000)


def _loc(item):
    return ((item.get("geometry") or {}).get("location") or {}).get("coordinates")


class InvipoCatalog:
    def __init__(self, base_url=None, email=None, password=None, http=None):
        self.base_url = base_url or config.INVIPO_BASE_URL
        self.email = email or config.INVIPO_EMAIL
        self.password = password or config.INVIPO_PASSWORD
        self.http = http or httpx.Client(timeout=60)
        self._token = None
        self._lock = threading.Lock()
        self._loaded_at = 0
        self.cameras: dict[str, Camera] = {}
        self.junctions: dict[int, dict] = {}

    # 1. đăng nhập
    def _login(self):
        if not self.email or not self.password:
            raise InvipoError("Thiếu INVIPO_EMAIL / INVIPO_PASSWORD trong .env")
        r = self.http.post(f"{self.base_url}/api/invipo/auth/login",
                           json={"email": self.email, "password": self.password})
        if r.status_code != 200:
            raise InvipoError(f"Đăng nhập Invipo thất bại: HTTP {r.status_code}")
        self._token = r.json().get("accessToken")
        if not self._token:
            raise InvipoError("Invipo không trả về accessToken")
        log.info("Đăng nhập Invipo thành công.")

    def _dataset(self, name, page_size=500):
        if not self._token:
            self._login()
        out, page = [], 1
        while page <= 50:
            body = {"paging": {"pageSize": page_size, "page": page}}
            r = self.http.post(f"{self.base_url}/api/data/datasets/{name}", json=body,
                               headers={"Authorization": f"Bearer {self._token}"})
            if r.status_code == 401:  # token hết hạn -> login lại 1 lần
                self._login()
                r = self.http.post(f"{self.base_url}/api/data/datasets/{name}", json=body,
                                   headers={"Authorization": f"Bearer {self._token}"})
            if r.status_code != 200:
                raise InvipoError(f"Invipo datasets/{name}: HTTP {r.status_code}")
            items = r.json().get("items") or []
            out += items
            if len(items) < page_size:
                break
            page += 1
        return out

    # 2. list camera + junction
    def refresh(self, force=False):
        with self._lock:
            if not force and self.cameras and time.time() - self._loaded_at < config.CATALOG_TTL_S:
                return
            t0 = time.time()
            items = self._dataset("items")
            areas = self._dataset("areas")
            self.cameras, self.junctions = build_catalog(items, areas)
            self._loaded_at = time.time()
            log.info("Tải %d camera AC, %d junction (%.1fs).", len(self.cameras), len(self.junctions), time.time() - t0)

    def get(self, camera_id):
        self.refresh()
        return self.cameras.get(camera_id)


def build_catalog(items, areas):
    """Tách riêng để test không cần mạng."""
    controllers = []
    junctions = {}
    for it in items:
        if it.get("type") != "TrafficLightController":
            continue
        nums = [int(x) for x in re.findall(r"J(\d+)", it.get("name", ""))]
        if nums and _loc(it):
            controllers.append((nums, _loc(it)))
            for n in nums:
                junctions.setdefault(n, {"junction": n, "name": it["name"].strip(), "location": _loc(it)})

    cams = {}
    for it in items:
        if it.get("type") != "Camera" or "TrafficLoadDetection" not in (it.get("powers") or []):
            continue
        meta = it.get("meta") or {}
        snap = (it.get("data") or {}).get("snapshot") or {}
        cams[it["_id"]] = Camera(
            id=it["_id"], name=it.get("name", "").strip(), model=it.get("model", ""),
            ip=meta.get("ip"), status=(it.get("connection") or {}).get("status"),
            location=_loc(it), stream_url=meta.get("streamUrl"),
            snapshot_url=snap.get("image"), snapshot_ts=snap.get("timestamp"),
        )

    # 1. area Loads
    for a in areas:
        m = re.match(r"\s*(\d+)\s*\.", a.get("name", ""))
        if not m:
            continue
        n = int(m.group(1))
        junctions.setdefault(n, {"junction": n, "name": a["name"].strip(), "location": None})
        junctions[n]["areaName"] = a["name"].strip()
        for arm in ((a.get("schemas") or {}).get("loadsDiagram") or {}).get("arms") or []:
            cid = (arm.get("item") or {}).get("_id")
            if cid in cams:
                cams[cid].junctions[n] = {"reason": "loads_area", "arm": arm.get("name")}

    # 2. controller gần nhất
    for c in cams.values():
        if not c.location or not controllers:
            continue
        nums, loc = min(controllers, key=lambda x: _dist_m(c.location, x[1]))
        d = _dist_m(c.location, loc)
        if d <= JUNCTION_RADIUS_M:
            for n in nums:
                c.junctions.setdefault(n, {"reason": "near_controller", "distanceM": round(d)})

    # 3. mã C trong tên (chỉ khi chưa có gì)
    for c in cams.values():
        if c.junctions:
            continue
        m = re.search(r"C(\d{3})", c.name)
        if m:
            n = int(m.group(1))
            c.junctions[n] = {"reason": "name_code"}
            junctions.setdefault(n, {"junction": n, "name": f"C{m.group(1)}", "location": None})

    for n, j in junctions.items():
        j["cameraCount"] = sum(1 for c in cams.values() if n in c.junctions)
    return cams, junctions
