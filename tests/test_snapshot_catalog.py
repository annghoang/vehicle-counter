import httpx
import pytest

from app.invipo import build_catalog, Camera
from app.snapshot import SnapshotProvider, SnapshotError, vms_jpeg_url, _host_allowed

JPEG = b"\xff\xd8" + b"0" * 2000
STREAM = "http://5.28.4.232/vms-34/mt/api/rest/v1/media?session=SECRET&cameraId=abc&format=fmp4&t=live"
CACHED = "http://5.28.4.78:7073/snapshots/x.jpeg"


class FakeCatalog:
    def __init__(self, cams):
        self.cameras = {c.id: c for c in cams}

    def get(self, cid):
        return self.cameras.get(cid)


def cam(**kw):
    base = dict(id="c1", name="Test-AC-VA", model="IN-TFS", ip="5.28.1.1", status="Connected",
                location=[105.8, 21.0], stream_url=STREAM, snapshot_url=CACHED, snapshot_ts="2026-11-11T04:00:00Z")
    base.update(kw)
    return Camera(**base)


def provider(handler, cams):
    return SnapshotProvider(FakeCatalog(cams), http=httpx.Client(transport=httpx.MockTransport(handler)))


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr("app.snapshot.time.sleep", lambda s: None)


def test_vms_url_rewrite():
    u = vms_jpeg_url(STREAM)
    assert "format=jpeg" in u and "t=live" in u and "session=SECRET" in u and "fmp4" not in u
    with pytest.raises(SnapshotError):
        vms_jpeg_url("http://5.28.4.232/x?format=fmp4")


def test_live_ok():
    p = provider(lambda r: httpx.Response(200, content=JPEG), [cam()])
    s = p.get_snapshot("c1")
    assert s.source == "vms_live" and s.is_live and s.image == JPEG


def test_fallback_to_cached_when_vms_fails():
    def h(r):
        return httpx.Response(500, content=b"err") if "vms-34" in str(r.url) else httpx.Response(200, content=JPEG)
    s = provider(h, [cam()]).get_snapshot("c1")
    assert s.source == "invipo_cached" and not s.is_live and "cache" in s.note


def test_all_fail_offline():
    def h(r):
        raise httpx.ConnectTimeout("timeout")
    with pytest.raises(SnapshotError) as e:
        provider(h, [cam(status="Disconnected")]).get_snapshot("c1")
    assert e.value.code == "camera_offline" and "Disconnected" in str(e.value)
    assert "SECRET" not in str(e.value)  # không lộ session


def test_unknown_camera_and_bad_config():
    p = provider(lambda r: httpx.Response(200, content=JPEG), [cam(stream_url=None, snapshot_url=None)])
    with pytest.raises(SnapshotError) as e:
        p.get_snapshot("nope")
    assert e.value.code == "camera_not_found"
    with pytest.raises(SnapshotError):
        p.get_snapshot("c1")


def test_ssrf_block():
    assert _host_allowed("http://5.28.4.232/x") and not _host_allowed("http://169.254.169.254/latest")
    assert not _host_allowed("http://evil.example.com/a.jpg")
    calls = []
    p = provider(lambda r: calls.append(r) or httpx.Response(200, content=JPEG),
                 [cam(stream_url="http://10.0.0.1/vms/media?session=s&cameraId=x", snapshot_url="http://1.2.3.4/a.jpg")])
    with pytest.raises(SnapshotError):
        p.get_snapshot("c1")
    assert calls == []


def test_non_jpeg_rejected():
    p = provider(lambda r: httpx.Response(200, content=b"<html>login</html>"), [cam()])
    with pytest.raises(SnapshotError):
        p.get_snapshot("c1")


def test_build_catalog_junctions():
    items = [
        {"_id": "t1", "type": "TrafficLightController", "name": "J196", "geometry": {"location": {"coordinates": [105.0, 21.0]}}},
        {"_id": "a", "type": "Camera", "name": "X-AC-VA", "powers": ["TrafficLoadDetection"], "meta": {"streamUrl": STREAM},
         "geometry": {"location": {"coordinates": [105.0003, 21.0]}}, "connection": {"status": "Connected"}},
        {"_id": "b", "type": "Camera", "name": "far DA165.C077-AC-VB", "powers": ["TrafficLoadDetection"], "meta": {},
         "geometry": {"location": {"coordinates": [105.1, 21.0]}}},
        {"_id": "c", "type": "Camera", "name": "Y-VIO", "powers": ["CameraSnapshot"], "meta": {},
         "geometry": {"location": {"coordinates": [105.0, 21.0]}}},
    ]
    areas = [{"name": "201. Area", "schemas": {"loadsDiagram": {"arms": [{"name": "A", "item": {"_id": "a"}}]}}}]
    cams, junctions = build_catalog(items, areas)
    assert set(cams) == {"a", "b"}  # chỉ camera Loads
    assert cams["a"].junctions[196]["reason"] == "near_controller"
    assert cams["a"].junctions[201]["reason"] == "loads_area"
    assert cams["b"].junctions == {77: {"reason": "name_code"}}
    assert junctions[196]["cameraCount"] == 1
    assert "streamUrl" not in str(cams["a"].public()) and "SECRET" not in str(cams["a"].public())
