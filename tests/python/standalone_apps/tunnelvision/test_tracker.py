import pytest

from hailo_apps.python.standalone_apps.tunnelvision.tracker import (
    IoUTracker, iou,
)


def test_iou_perfect_match():
    assert iou([0, 0, 100, 100], [0, 0, 100, 100]) == pytest.approx(1.0)


def test_iou_no_overlap():
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_iou_half_overlap():
    # Two 100x100 boxes shifted by 50px → intersection 50x100=5000, union 15000
    val = iou([0, 0, 100, 100], [50, 0, 150, 100])
    assert val == pytest.approx(5000 / 15000)


def test_tracker_assigns_new_id_to_first_detection():
    t = IoUTracker(iou_threshold=0.3, max_lost_frames=5)
    out = t.update([[0, 0, 100, 100]])
    assert len(out) == 1
    assert out[0].track_id == 1


def test_tracker_keeps_id_across_frames():
    t = IoUTracker(iou_threshold=0.3, max_lost_frames=5)
    out1 = t.update([[0, 0, 100, 100]])
    out2 = t.update([[5, 5, 105, 105]])  # high IoU
    assert out1[0].track_id == out2[0].track_id


def test_tracker_assigns_new_id_when_iou_too_low():
    t = IoUTracker(iou_threshold=0.3, max_lost_frames=5)
    t.update([[0, 0, 100, 100]])
    out = t.update([[200, 200, 300, 300]])
    assert out[0].track_id == 2


def test_tracker_retires_lost_track():
    t = IoUTracker(iou_threshold=0.3, max_lost_frames=2)
    out1 = t.update([[0, 0, 100, 100]])
    first_id = out1[0].track_id
    for _ in range(3):
        t.update([])
    out = t.update([[0, 0, 100, 100]])  # should be a NEW track now
    assert out[0].track_id != first_id


def test_tracker_handles_two_simultaneous_vehicles():
    t = IoUTracker(iou_threshold=0.3, max_lost_frames=5)
    out1 = t.update([[0, 0, 100, 100], [200, 200, 300, 300]])
    ids = sorted(o.track_id for o in out1)
    assert ids == [1, 2]
    out2 = t.update([[5, 5, 105, 105], [205, 205, 305, 305]])
    assert sorted(o.track_id for o in out2) == [1, 2]
