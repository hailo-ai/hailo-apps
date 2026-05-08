import time
import numpy as np
import pytest

from hailo_apps.python.standalone_apps.carwash_lpr.voter import TemporalVoter, VotedPlate


def _dummy_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_no_emit_below_threshold():
    voter = TemporalVoter(window_size=10, emit_threshold=6, dedup_seconds=30,
                          fallback_threshold=0.60)
    results = []
    for _ in range(5):
        voted = voter.add("ABC123", 0.90, _dummy_frame())
        if voted:
            results.append(voted)
    assert results == []


def test_emit_at_threshold():
    voter = TemporalVoter(window_size=10, emit_threshold=6, dedup_seconds=30,
                          fallback_threshold=0.60)
    result = None
    for _ in range(6):
        result = voter.add("ABC123", 0.90, _dummy_frame())
    assert isinstance(result, VotedPlate)
    assert result.plate_string == "ABC123"
    assert result.confidence >= 0.6


def test_dedup_suppresses_same_plate(monkeypatch):
    voter = TemporalVoter(window_size=10, emit_threshold=6, dedup_seconds=30,
                          fallback_threshold=0.60)
    # First emission
    for _ in range(6):
        voter.add("ABC123", 0.90, _dummy_frame())
    # Second round — should be suppressed within dedup window
    voter._window.clear()
    results = []
    for _ in range(6):
        r = voter.add("ABC123", 0.90, _dummy_frame())
        if r:
            results.append(r)
    assert results == []


def test_dedup_allows_after_window(monkeypatch):
    voter = TemporalVoter(window_size=10, emit_threshold=6, dedup_seconds=1,
                          fallback_threshold=0.60)
    for _ in range(6):
        voter.add("ABC123", 0.90, _dummy_frame())
    time.sleep(1.1)
    voter._window.clear()
    result = None
    for _ in range(6):
        result = voter.add("ABC123", 0.90, _dummy_frame())
    assert result is not None


def test_needs_fallback_flag_when_low_confidence():
    voter = TemporalVoter(window_size=10, emit_threshold=6, dedup_seconds=30,
                          fallback_threshold=0.80)
    result = None
    for _ in range(6):
        result = voter.add("ABC123", 0.70, _dummy_frame())
    assert result is not None
    assert result.needs_fallback is True


def test_best_frame_is_highest_confidence_frame():
    voter = TemporalVoter(window_size=10, emit_threshold=6, dedup_seconds=30,
                          fallback_threshold=0.60)
    frames = [np.full((480, 640, 3), i, dtype=np.uint8) for i in range(6)]
    confidences = [0.70, 0.80, 0.95, 0.72, 0.68, 0.75]
    result = None
    for i in range(6):
        result = voter.add("ABC123", confidences[i], frames[i])
    assert result is not None
    assert np.array_equal(result.best_frame, frames[2])  # index 2 = 0.95
