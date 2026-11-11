import numpy as np
import cv2
import pytest
from fastapi.testclient import TestClient

from app import main
from app.detector import Detection
from app.snapshot import Snapshot, SnapshotError


def jpeg(w=640, h=360):
    img = np.full((h, w, 3), 90, np.uint8)
    cv2.rectangle(img, (100, 100), (200, 200), (255, 255, 255), -1)
    return cv2.imencode(".jpg", img)[1].tobytes()


class FakeSnaps:
    def __init__(self, fail=None):
        self.fail = fail

    def get_snapshot(self, cid, allow_cached=True):
        if self.fail:
            raise SnapshotError("Không lấy được ảnh", self.fail)
        return Snapshot(jpeg(), "2026-11-11T04:00:00+00:00", "vms_live", cid, True)


class FakeDet:
    info = {"name": "fake"}

    def __init__(self, dets=None, boom=False):
        self.dets, self.boom = dets or [], boom

    def detect(self, img):
        if self.boom:
            raise RuntimeError("mps oom")
        return self.dets


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main.catalog, "refresh", lambda force=False: None)
    monkeypatch.setattr(main.catalog, "cameras", {})
    return TestClient(main.app)


def use(monkeypatch, snaps, det):
    monkeypatch.setattr(main.service, "snapshots", snaps)
    monkeypatch.setattr(main.service, "detector", det)


def test_analyze_ok(client, monkeypatch):
    use(monkeypatch, FakeSnaps(), FakeDet([Detection(100, 100, 200, 300, 0.9, "car"),
                                           Detection(400, 200, 430, 300, 0.8, "motorcycle")]))
    r = client.put("/api/cameras/cam1/roi", json={"rois": [{"id": "r1", "name": "VA", "type": "queue",
                                                            "polygon": [[0, 0.5], [1, 0.5], [1, 1], [0, 1]]}]})
    assert r.status_code == 200
    r = client.post("/api/analyze", json={"cameraId": "cam1"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["counts"]["total"] == 2 and j["counts"]["roiMode"] == "configured"
    assert j["counts"]["rois"][0]["type"] == "queue"
    assert j["snapshot"]["source"] == "vms_live"
    assert client.get(j["images"]["annotated"]).headers["content-type"] == "image/jpeg"
    assert client.get(j["images"]["source"]).status_code == 200
    assert client.get(f"/api/results/{j['resultId']}").json()["resultId"] == j["resultId"]


def test_analyze_offline(client, monkeypatch):
    use(monkeypatch, FakeSnaps("camera_offline"), FakeDet())
    r = client.post("/api/analyze", json={"cameraId": "cam1"})
    assert r.status_code == 503 and r.json()["detail"]["code"] == "camera_offline"


def test_analyze_inference_failure(client, monkeypatch):
    use(monkeypatch, FakeSnaps(), FakeDet(boom=True))
    r = client.post("/api/analyze", json={"cameraId": "cam1"})
    assert r.status_code == 500 and r.json()["detail"]["code"] == "inference_failed"


def test_empty_road(client, monkeypatch):
    use(monkeypatch, FakeSnaps(), FakeDet([]))
    j = client.post("/api/analyze", json={"cameraId": "cam1"}).json()
    assert j["counts"]["total"] == 0 and j["detections"] == []


def test_bad_inputs(client):
    assert client.post("/api/analyze", json={"cameraId": "../etc"}).status_code == 422
    assert client.put("/api/cameras/cam1/roi", json={"rois": [{"id": "a", "polygon": [[0, 0], [5, 0], [1, 1]]}]}).status_code == 422
    assert client.get("/api/results/../../x").status_code == 404
    assert client.get("/api/results/20261111T000000-abcdef/evil.jpg").status_code == 404
