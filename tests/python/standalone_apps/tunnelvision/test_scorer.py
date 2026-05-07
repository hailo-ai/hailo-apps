import numpy as np
import pytest

from hailo_apps.python.standalone_apps.tunnelvision.scorer import (
    compute_blur_score, compute_quality_score, frame_passes_gates,
    select_best_candidate, ScoringRule,
)
from hailo_apps.python.standalone_apps.tunnelvision.state import FrameCandidate


def _checkerboard(size=128):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[::8, :] = 255
    img[:, ::8] = 255
    return img


def _blurred(size=128):
    return np.full((size, size, 3), 128, dtype=np.uint8)


def test_blur_score_higher_for_sharp_image():
    sharp = compute_blur_score(_checkerboard())
    blur = compute_blur_score(_blurred())
    assert sharp > blur >= 0


def test_quality_score_increases_with_plate_confidence():
    base = dict(
        plate_confidence=0.5, blur_score=100, plate_width_px=80,
        bbox_stability=0.9, roi_fit=0.9, motion_score=0.0,
        plate_blocked=False, plate_width_ref=120,
    )
    s_low = compute_quality_score(**base)
    s_high = compute_quality_score(**{**base, "plate_confidence": 0.95})
    assert s_high > s_low


def test_frame_passes_gates_rejects_blurry():
    rule = ScoringRule()
    assert not frame_passes_gates(
        plate_confidence=0.9, blur_score=10.0,
        plate_width_px=80, motion_score=0.0,
        bbox_growth_rate=0.0, fully_visible=True,
        plate_visible=True, plate_blocked=False,
        rule=rule,
    )


def test_frame_passes_gates_accepts_good():
    rule = ScoringRule()
    assert frame_passes_gates(
        plate_confidence=0.85, blur_score=200.0,
        plate_width_px=80, motion_score=0.5,
        bbox_growth_rate=0.05, fully_visible=True,
        plate_visible=True, plate_blocked=False,
        rule=rule,
    )


def test_select_best_picks_highest_score():
    cs = [
        FrameCandidate(frame=None, quality_score=70, blur_score=80, motion_score=0,
                       plate_width_px=70, plate_confidence=0.8, bbox_stability=0.9,
                       vehicle_bbox=[0,0,100,100], plate_bbox=[10,60,80,90]),
        FrameCandidate(frame=None, quality_score=90, blur_score=200, motion_score=0,
                       plate_width_px=80, plate_confidence=0.92, bbox_stability=0.95,
                       vehicle_bbox=[0,0,100,100], plate_bbox=[10,60,80,90]),
        FrameCandidate(frame=None, quality_score=80, blur_score=150, motion_score=0,
                       plate_width_px=75, plate_confidence=0.85, bbox_stability=0.93,
                       vehicle_bbox=[0,0,100,100], plate_bbox=[10,60,80,90]),
    ]
    best = select_best_candidate(cs)
    assert best.quality_score == 90
