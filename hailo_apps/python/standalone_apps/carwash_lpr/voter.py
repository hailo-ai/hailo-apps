import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class VotedPlate:
    plate_string: str
    confidence: float
    best_frame: np.ndarray
    needs_fallback: bool = False


class TemporalVoter:
    def __init__(self, window_size: int, emit_threshold: int,
                 dedup_seconds: float, fallback_threshold: float):
        self._window_size = window_size
        self._emit_threshold = emit_threshold
        self._dedup_seconds = dedup_seconds
        self._fallback_threshold = fallback_threshold
        self._window: deque = deque(maxlen=window_size)
        self._last_emitted: dict[str, float] = {}

    def add(self, plate_string: str, confidence: float,
            frame: np.ndarray) -> Optional[VotedPlate]:
        self._window.append((plate_string, confidence, frame))

        if len(self._window) < self._emit_threshold:
            return None

        # Count occurrences of most common string
        counts: dict[str, int] = {}
        for p, _, _ in self._window:
            counts[p] = counts.get(p, 0) + 1

        best_plate = max(counts, key=lambda k: counts[k])
        if counts[best_plate] < self._emit_threshold:
            return None

        # Dedup check
        now = time.monotonic()
        last = self._last_emitted.get(best_plate, 0.0)
        if now - last < self._dedup_seconds:
            return None

        self._last_emitted[best_plate] = now

        # Best frame = highest confidence frame for this plate in window
        candidates = [(conf, frm) for p, conf, frm in self._window if p == best_plate]
        best_conf, best_frame = max(candidates, key=lambda x: x[0])

        return VotedPlate(
            plate_string=best_plate,
            confidence=best_conf,
            best_frame=best_frame,
            needs_fallback=best_conf < self._fallback_threshold,
        )
