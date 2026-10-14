from dataclasses import dataclass

import cv2
import numpy as np

from hailo_apps.python.standalone_apps.tunnelvision.state import FrameCandidate


@dataclass
class ScoringRule:
    min_vehicle_confidence: float = 0.5
    min_plate_confidence: float = 0.6
    min_plate_width_px: int = 40
    min_blur_score: float = 50.0
    max_motion_score: float = 5.0
    max_bbox_growth_rate: float = 0.20
    plate_width_ref: int = 120  # px reference for plate_width_score normalization


def compute_blur_score(image: np.ndarray) -> float:
    """Laplacian variance — higher = sharper. Range typically 0..2000+."""
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_motion_score(prev_centroid, curr_centroid) -> float:
    """Pixel delta of bbox centroid between frames. Lower = stable."""
    if prev_centroid is None or curr_centroid is None:
        return 0.0
    dx = curr_centroid[0] - prev_centroid[0]
    dy = curr_centroid[1] - prev_centroid[1]
    return float(np.hypot(dx, dy))


def compute_quality_score(
    *,
    plate_confidence: float,
    blur_score: float,
    plate_width_px: int,
    bbox_stability: float,
    roi_fit: float,
    motion_score: float,
    plate_blocked: bool,
    plate_width_ref: int = 120,
    blur_ref: float = 200.0,
) -> float:
    """PRD §11 quality formula."""
    plate_width_norm = min(plate_width_px / max(plate_width_ref, 1), 1.0)
    blur_norm = min(blur_score / max(blur_ref, 1), 1.0)
    motion_penalty = min(motion_score, 10.0) * 2.0
    blockage_penalty = 50.0 if plate_blocked else 0.0
    return (
        plate_confidence * 30.0
        + blur_norm * 25.0
        + plate_width_norm * 20.0
        + bbox_stability * 15.0
        + roi_fit * 10.0
        - motion_penalty
        - blockage_penalty
    )


def frame_passes_gates(
    *,
    plate_confidence: float,
    blur_score: float,
    plate_width_px: int,
    motion_score: float,
    bbox_growth_rate: float,
    fully_visible: bool,
    plate_visible: bool,
    plate_blocked: bool,
    rule: ScoringRule,
) -> bool:
    if not (fully_visible and plate_visible):
        return False
    if plate_blocked:
        return False
    if plate_confidence < rule.min_plate_confidence:
        return False
    if plate_width_px < rule.min_plate_width_px:
        return False
    if blur_score < rule.min_blur_score:
        return False
    if motion_score > rule.max_motion_score:
        return False
    if abs(bbox_growth_rate) > rule.max_bbox_growth_rate:
        return False
    return True


def select_best_candidate(candidates: list) -> FrameCandidate:
    return max(candidates, key=lambda c: c.quality_score)
