from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np


class TrackState(str, Enum):
    NEW = "new"
    TRACKING = "tracking"
    PARTIAL_VISIBLE = "partial_visible"
    FULLY_VISIBLE = "fully_visible"
    PLATE_BLOCKED = "plate_blocked"
    PLATE_CANDIDATE = "plate_candidate"
    BEST_IMAGE_SELECTED = "best_image_selected"
    REKOR_SENT = "rekor_sent"
    PROCESSED_BUT_TRACKING = "processed_but_tracking"
    IN_TUNNEL = "in_tunnel"
    MATCHED_AT_EGRESS = "matched_at_egress"
    COMPLETED = "completed"
    LOST = "lost"
    EXITED = "exited"
    NO_READ = "no_read"


@dataclass
class FrameCandidate:
    frame: Optional[np.ndarray]
    quality_score: float
    blur_score: float
    motion_score: float
    plate_width_px: int
    plate_confidence: float
    bbox_stability: float
    vehicle_bbox: list
    plate_bbox: list


@dataclass
class VehicleTrack:
    track_id: int
    camera_role: str
    state: TrackState
    first_seen_at: datetime
    last_seen_at: datetime
    current_bbox: list
    db_id: Optional[str] = None
    observation_db_id: Optional[str] = None
    current_zone: Optional[str] = None
    fully_visible: bool = False
    processed: bool = False
    rekor_sent: bool = False
    candidates: list = field(default_factory=list)
    best_image_path: Optional[str] = None
    best_candidate: Optional[FrameCandidate] = None
    lost_frames: int = 0


_TERMINAL = {TrackState.LOST, TrackState.EXITED, TrackState.NO_READ, TrackState.COMPLETED}


def transition(
    track: VehicleTrack,
    current_zone: Optional[str],
    fully_visible: bool,
    plate_visible: bool,
    plate_blocked: bool,
    max_lost_frames: int = 10,
) -> TrackState:
    """Pure function — returns the next state without mutating the track."""
    if track.state in _TERMINAL:
        return track.state

    if track.lost_frames > max_lost_frames:
        return TrackState.LOST

    if track.processed:
        return TrackState.PROCESSED_BUT_TRACKING

    if track.state == TrackState.NEW:
        return TrackState.TRACKING

    if current_zone == "capture":
        if not fully_visible:
            return TrackState.PARTIAL_VISIBLE
        if plate_blocked:
            return TrackState.PLATE_BLOCKED
        if plate_visible:
            return TrackState.PLATE_CANDIDATE
        return TrackState.FULLY_VISIBLE

    return TrackState.TRACKING
