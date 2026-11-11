"""SnapshotProvider.get_snapshot(camera_id)

Thứ tự nguồn ảnh:
  1. vms_live      : Avigilon Web Endpoint qua proxy 5.28.4.232/vms-XX (URL + session lấy từ
                     meta.streamUrl của Invipo, đổi format=fmp4 -> jpeg, t=live). Ảnh hiện tại, full res.
  2. invipo_cached : data.snapshot.image Invipo tự chụp định kỳ (vài phút). Có timestamp riêng, có thể cũ.

Không nhận URL từ client: URL chỉ lấy từ catalog Invipo và phải nằm trong ALLOWED_SNAPSHOT_NETS.
Không log URL (có session VMS).
"""
import ipaddress
import logging
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from . import config

log = logging.getLogger("snapshot")


@dataclass
class Snapshot:
    image: bytes
    timestamp: str  # ISO UTC, thời điểm ảnh được chụp
    source: str  # vms_live | invipo_cached
    camera_id: str
    is_live: bool
    note: str | None = None


class SnapshotError(Exception):
    def __init__(self, message, code="snapshot_unavailable"):
        super().__init__(message)
        self.code = code


def _now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _host_allowed(url):
    try:
        host = urllib.parse.urlsplit(url).hostname
        ip = ipaddress.ip_address(host)
    except (ValueError, TypeError):
        return False
    return any(ip in ipaddress.ip_network(n) for n in config.ALLOWED_SNAPSHOT_NETS)


def vms_jpeg_url(stream_url):
    """meta.streamUrl (fmp4) -> URL lấy 1 ảnh JPEG live, giữ nguyên session/cameraId."""
    u = urllib.parse.urlsplit(stream_url)
    q = dict(urllib.parse.parse_qsl(u.query))
    if "cameraId" not in q or "session" not in q:
        raise SnapshotError("streamUrl thiếu cameraId/session", "invalid_camera_config")
    q["format"] = "jpeg"
    q["t"] = "live"
    q.pop("range", None)
    return urllib.parse.urlunsplit((u.scheme, u.netloc, u.path, urllib.parse.urlencode(q), ""))


def _is_jpeg(b):
    return len(b) > 1000 and b[:2] == b"\xff\xd8"


class SnapshotProvider:
    def __init__(self, catalog, http=None):
        self.catalog = catalog
        self.http = http or httpx.Client(timeout=config.SNAPSHOT_TIMEOUT_S, verify=False)

    def _fetch(self, url, label):
        if not _host_allowed(url):
            raise SnapshotError(f"{label}: host không nằm trong dải cho phép", "invalid_camera_config")
        last = None
        for attempt in range(1 + config.SNAPSHOT_RETRIES):
            try:
                r = self.http.get(url)
                if r.status_code == 200 and _is_jpeg(r.content):
                    return r.content
                last = f"HTTP {r.status_code}, {len(r.content)} bytes"
            except httpx.TimeoutException:
                last = "timeout"
            except httpx.HTTPError as e:
                last = type(e).__name__
            log.warning("[%s] lần %d lỗi: %s", label, attempt + 1, last)
            time.sleep(0.5)
        raise SnapshotError(f"{label}: {last}")

    def get_snapshot(self, camera_id, allow_cached=True):
        cam = self.catalog.get(camera_id)
        if cam is None:
            raise SnapshotError("Không tìm thấy camera", "camera_not_found")

        errors = []
        # 1. ảnh live từ VMS
        if cam.stream_url:
            try:
                img = self._fetch(vms_jpeg_url(cam.stream_url), "vms_live")
                return Snapshot(img, _now_iso(), "vms_live", camera_id, True)
            except SnapshotError as e:
                errors.append(str(e))
        else:
            errors.append("camera không có streamUrl")

        # 2. ảnh cache của Invipo
        if allow_cached and cam.snapshot_url:
            try:
                img = self._fetch(cam.snapshot_url, "invipo_cached")
                age = _age_s(cam.snapshot_ts)
                note = f"Ảnh cache Invipo, chụp cách đây ~{age}s" if age is not None else "Ảnh cache Invipo"
                return Snapshot(img, cam.snapshot_ts or _now_iso(), "invipo_cached", camera_id, False, note)
            except SnapshotError as e:
                errors.append(str(e))

        status = f" (Invipo: {cam.status})" if cam.status and cam.status != "Connected" else ""
        raise SnapshotError("Không lấy được ảnh" + status + ": " + "; ".join(errors), "camera_offline")


def _age_s(ts):
    if not ts:
        return None
    try:
        t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return int((datetime.now(timezone.utc) - t).total_seconds())
    except ValueError:
        return None
