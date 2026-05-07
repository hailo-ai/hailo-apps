from datetime import datetime, timezone

from hailo_apps.python.standalone_apps.tunnelvision.state import (
    TrackState, VehicleTrack, FrameCandidate, transition,
)


def _make_track(state=TrackState.NEW) -> VehicleTrack:
    now = datetime.now(timezone.utc)
    return VehicleTrack(
        track_id=1, camera_role="ingress", state=state,
        first_seen_at=now, last_seen_at=now, current_bbox=[0, 0, 100, 100],
    )


def test_new_to_tracking_on_first_detection():
    t = _make_track(TrackState.NEW)
    next_state = transition(t, current_zone="approach", fully_visible=True,
                            plate_visible=False, plate_blocked=False)
    assert next_state == TrackState.TRACKING


def test_tracking_to_fully_visible_when_in_capture():
    t = _make_track(TrackState.TRACKING)
    next_state = transition(t, current_zone="capture", fully_visible=True,
                            plate_visible=False, plate_blocked=False)
    assert next_state == TrackState.FULLY_VISIBLE


def test_fully_visible_to_plate_blocked():
    t = _make_track(TrackState.FULLY_VISIBLE)
    next_state = transition(t, current_zone="capture", fully_visible=True,
                            plate_visible=False, plate_blocked=True)
    assert next_state == TrackState.PLATE_BLOCKED


def test_plate_blocked_recovers_when_unblocked():
    t = _make_track(TrackState.PLATE_BLOCKED)
    next_state = transition(t, current_zone="capture", fully_visible=True,
                            plate_visible=True, plate_blocked=False)
    assert next_state == TrackState.PLATE_CANDIDATE


def test_processed_track_keeps_tracking():
    t = _make_track(TrackState.PROCESSED_BUT_TRACKING)
    t.processed = True
    next_state = transition(t, current_zone="capture", fully_visible=True,
                            plate_visible=True, plate_blocked=False)
    # Already processed, do not re-enter capture cycle
    assert next_state == TrackState.PROCESSED_BUT_TRACKING


def test_lost_track_after_max_lost_frames():
    t = _make_track(TrackState.TRACKING)
    t.lost_frames = 11  # > default max_lost_frames
    next_state = transition(t, current_zone=None, fully_visible=False,
                            plate_visible=False, plate_blocked=False,
                            max_lost_frames=10)
    assert next_state == TrackState.LOST


def test_frame_candidate_dataclass_holds_metadata():
    c = FrameCandidate(
        frame=None, quality_score=85.0, blur_score=120.0,
        motion_score=0.5, plate_width_px=80, plate_confidence=0.91,
        bbox_stability=0.95, vehicle_bbox=[0, 0, 100, 200],
        plate_bbox=[10, 60, 80, 90],
    )
    assert c.quality_score == 85.0
