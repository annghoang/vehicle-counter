from app.counting import count, point_in_polygon, validate_rois, recover_riders
from app.detector import Detection, merge_detections, drop_cut_boxes, tile_windows

W, H = 1000, 500


def D(x1, y1, x2, y2, cls, conf=0.9):
    return Detection(x1, y1, x2, y2, conf, cls)


def test_class_mapping_and_totals():
    dets = [D(0, 0, 10, 10, "motorcycle"), D(20, 0, 60, 30, "car"), D(100, 0, 200, 80, "bus"),
            D(300, 0, 400, 90, "truck"), D(500, 0, 510, 20, "bicycle"), D(600, 0, 620, 40, "person"),
            D(700, 0, 720, 40, "traffic light")]
    s, _ = count(dets, W, H, [], rider_recovery=False)
    assert s["byClass"] == {"motorcycle": 1, "car": 1, "bus": 1, "truck": 1, "other": 1}
    assert s["total"] == sum(s["byClass"].values()) == 5
    assert s["roiMode"] == "full_frame"


def test_class_threshold():
    s, _ = count([D(0, 0, 10, 10, "car", 0.2), D(0, 0, 10, 10, "motorcycle", 0.26)], W, H, [], rider_recovery=False)
    assert s["byClass"]["car"] == 0 and s["byClass"]["motorcycle"] == 1 and s["belowThreshold"] == 1


def test_roi_uses_bottom_center():
    roi = [{"id": "low", "type": "count", "polygon": [[0, 0.5], [1, 0.5], [1, 1], [0, 1]]}]
    # bus cao: tâm box ở nửa trên, chân ở nửa dưới -> vẫn trong ROI
    s, _ = count([D(100, 50, 300, 300, "bus")], W, H, roi, rider_recovery=False)
    assert s["total"] == 1
    # xe nằm hẳn nửa trên -> ngoài ROI
    s, _ = count([D(100, 10, 200, 200, "car")], W, H, roi, rider_recovery=False)
    assert s["total"] == 0 and s["outsideRoi"] == 1


def test_overlapping_rois_not_double_counted():
    a = {"id": "a", "type": "count", "polygon": [[0, 0], [0.6, 0], [0.6, 1], [0, 1]]}
    b = {"id": "b", "type": "queue", "polygon": [[0.4, 0], [1, 0], [1, 1], [0.4, 1]]}
    dets = [D(450, 0, 550, 100, "car"), D(50, 0, 100, 100, "motorcycle"), D(900, 0, 950, 100, "motorcycle")]
    s, _ = count(dets, W, H, [a, b], rider_recovery=False)
    assert s["total"] == 3  # xe ở vùng chồng chỉ tính 1 lần
    per = {r["id"]: r["total"] for r in s["rois"]}
    assert per == {"a": 2, "b": 2}
    assert s["rois"][1]["type"] == "queue"


def test_empty_and_no_valid():
    s, c = count([], W, H, [])
    assert s["total"] == 0 and c == []
    s, _ = count([D(0, 0, 5, 5, "person", 0.1)], W, H, [], rider_recovery=True)
    assert s["total"] == 0


def test_resize_invariance():
    roi = [{"id": "r", "type": "count", "polygon": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.9], [0.2, 0.9]]}]
    d1 = [D(400, 100, 500, 400, "car")]
    d2 = [D(800, 200, 1000, 800, "car")]  # cùng xe trên ảnh gấp đôi
    s1, c1 = count(d1, 1000, 500, roi, rider_recovery=False)
    s2, c2 = count(d2, 2000, 1000, roi, rider_recovery=False)
    assert s1["total"] == s2["total"] == 1 and c1[0].anchor == c2[0].anchor


def test_point_in_polygon():
    sq = [[0, 0], [1, 0], [1, 1], [0, 1]]
    assert point_in_polygon(0.5, 0.5, sq) and not point_in_polygon(1.5, 0.5, sq)
    tri = [[0, 0], [1, 0], [0, 1]]
    assert point_in_polygon(0.2, 0.2, tri) and not point_in_polygon(0.8, 0.8, tri)


def test_validate_rois():
    assert validate_rois([{"id": "a", "polygon": [[0, 0], [1, 0], [1, 1]]}]) == []
    assert validate_rois([{"id": "a", "polygon": [[0, 0], [1, 0]]}])
    assert validate_rois([{"id": "a", "polygon": [[0, 0], [2, 0], [1, 1]]}])
    assert validate_rois([{"id": "a", "type": "x", "polygon": [[0, 0], [1, 0], [1, 1]]}])
    assert validate_rois([{"id": "a", "polygon": [[0, 0], [1, 0], [1, 1]]}, {"id": "a", "polygon": [[0, 0], [1, 0], [1, 1]]}])


def test_rider_recovery():
    car = D(0, 0, 200, 200, "car")
    moto = D(300, 100, 340, 200, "motorcycle")
    driver = D(50, 50, 100, 120, "person")  # trong ô tô
    on_moto = D(300, 60, 340, 150, "person")  # đã có xe máy
    rider = D(500, 50, 540, 150, "person")
    passenger = D(505, 40, 545, 145, "person")  # ngồi sau -> cùng xe
    cut = D(700, 470, 740, 500, "person")  # chạm mép dưới
    out = recover_riders([driver, on_moto, rider, passenger, cut], [car, moto], img_h=500)
    assert len(out) == 1 and out[0].raw_class == "motorcycle_rider"
    s, _ = count([car, moto, driver, on_moto, rider, passenger, cut], W, H, [], rider_recovery=True)
    assert s["byClass"]["motorcycle"] == 2 and s["motorcycleFromRider"] == 1


def test_merge_keeps_full_box_and_dedups_classes():
    full = D(0, 0, 400, 300, "car", 0.6)
    cut = D(250, 0, 400, 300, "car", 0.9)  # mảnh bị tile cắt, conf cao hơn
    out = merge_detections([full, cut])
    assert len(out) == 1 and out[0].x1 == 0
    cab = D(100, 200, 300, 400, "bus", 0.5)
    body = D(50, 0, 320, 410, "truck", 0.6)
    assert len(merge_detections([cab, body])) == 1
    m1, m2 = D(0, 0, 40, 80, "motorcycle"), D(30, 0, 70, 80, "motorcycle")
    assert len(merge_detections([m1, m2])) == 2  # xe máy sát nhau giữ nguyên
    p, m = D(0, 0, 40, 80, "person"), D(0, 10, 40, 90, "motorcycle")
    assert len(merge_detections([p, m])) == 2


def test_tiles_and_cut_boxes():
    wins = tile_windows(1920, 1080)
    assert len(wins) == 4 and wins[0][:2] == (0, 0) and wins[-1][2:] == (1920, 1080)
    win = (877, 0, 1920, 587)
    dets = [D(878, 10, 950, 100, "car"), D(1000, 10, 1100, 100, "car"), D(1800, 10, 1920, 100, "car")]
    kept = drop_cut_boxes(dets, win, 1920, 1080)
    assert len(kept) == 2  # chỉ bỏ box chạm cạnh tile nằm trong ảnh


def test_vehicle_touching_bottom_edge_counted():
    roi = [{"id": "low", "type": "count", "polygon": [[0, 0.5], [1, 0.5], [1, 1], [0, 1]]}]
    d = [D(0, 300, 200, 500, "car"), D(900, 400, 1000, 500, "motorcycle")]  # chạm mép dưới + góc phải
    s, _ = count(d, W, H, roi, rider_recovery=False)
    assert s["total"] == 2
    s, _ = count(d, W, H, [], rider_recovery=False)
    assert s["total"] == 2 and s["outsideRoi"] == 0
