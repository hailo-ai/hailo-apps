from dataclasses import dataclass, field
from typing import Iterable, Optional


def iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    bbox: list
    lost_frames: int = 0
    score: float = 0.0
    bbox_history: list = field(default_factory=list)


class IoUTracker:
    """Greedy IoU multi-object tracker.

    On update(detections):
      - For each existing track, find the detection with max IoU above iou_threshold.
      - Matched: update bbox + reset lost_frames.
      - Unmatched detections start new tracks.
      - Tracks not matched this frame have lost_frames++.
      - Tracks with lost_frames > max_lost_frames are retired.
    """

    def __init__(self, iou_threshold: float = 0.3, max_lost_frames: int = 10):
        self._iou_threshold = iou_threshold
        self._max_lost_frames = max_lost_frames
        self._next_id = 1
        self._tracks: list = []

    def update(self, detections: Iterable[list], scores: Optional[list] = None) -> list:
        dets = list(detections)
        scores = list(scores) if scores is not None else [0.0] * len(dets)

        # Greedy assignment: best IoU pair first
        unmatched_dets = set(range(len(dets)))
        matched_pairs = []
        for ti, track in enumerate(self._tracks):
            best_iou, best_di = 0.0, None
            for di in unmatched_dets:
                v = iou(track.bbox, dets[di])
                if v > best_iou:
                    best_iou, best_di = v, di
            if best_di is not None and best_iou >= self._iou_threshold:
                matched_pairs.append((ti, best_di))
                unmatched_dets.discard(best_di)

        matched_track_idx = {ti for ti, _ in matched_pairs}

        # Update matched
        for ti, di in matched_pairs:
            self._tracks[ti].bbox = list(dets[di])
            self._tracks[ti].score = scores[di]
            self._tracks[ti].lost_frames = 0
            self._tracks[ti].bbox_history.append(list(dets[di]))

        # Age unmatched tracks
        for ti, track in enumerate(self._tracks):
            if ti not in matched_track_idx:
                track.lost_frames += 1

        # New tracks for unmatched detections
        for di in unmatched_dets:
            self._tracks.append(Track(
                track_id=self._next_id, bbox=list(dets[di]),
                score=scores[di], bbox_history=[list(dets[di])],
            ))
            self._next_id += 1

        # Retire stale tracks
        self._tracks = [t for t in self._tracks if t.lost_frames <= self._max_lost_frames]

        # Return only tracks that were matched or created this frame (currently visible).
        return [t for t in self._tracks if t.lost_frames == 0]
