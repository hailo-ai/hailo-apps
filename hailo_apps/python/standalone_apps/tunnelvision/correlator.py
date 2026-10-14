from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Optional


@dataclass
class ActiveVisit:
    visit_id: str
    plate: Optional[str]
    ingress_at: datetime
    color: Optional[str] = None
    make_model: Optional[str] = None
    body_type: Optional[str] = None


@dataclass
class EgressEvent:
    plate: Optional[str]
    egress_at: datetime
    color: Optional[str] = None
    make_model: Optional[str] = None
    body_type: Optional[str] = None


@dataclass
class Match:
    visit_id: str
    confidence: float
    method: str  # "plate_exact" | "plate_fuzzy" | "lane_time_window" | "track_sequence"


def _normalize(plate: Optional[str]) -> Optional[str]:
    if not plate:
        return None
    return "".join(c for c in plate.upper() if c.isalnum())


def _plate_match(a: Optional[str], b: Optional[str]) -> float:
    """1.0 exact, 0..1 fuzzy via char-overlap, 0 if either missing."""
    a_n, b_n = _normalize(a), _normalize(b)
    if not a_n or not b_n:
        return 0.0
    if a_n == b_n:
        return 1.0
    if len(a_n) != len(b_n):
        return 0.0
    matches = sum(1 for x, y in zip(a_n, b_n) if x == y)
    return matches / len(a_n)


class Correlator:
    """Per-lane FIFO tunnel queue + egress->visit matcher (PRD 5.3)."""

    def __init__(
        self,
        *,
        min_match_score: float = 60.0,
        expected_tunnel_seconds: int = 300,
        plate_match_weight: float = 100.0,
        queue_order_weight: float = 40.0,
        time_similarity_weight: float = 25.0,
        attribute_weight: float = 15.0,
        color_weight: float = 5.0,
    ):
        self._min = min_match_score
        self._expected = expected_tunnel_seconds
        self._w_plate = plate_match_weight
        self._w_queue = queue_order_weight
        self._w_time = time_similarity_weight
        self._w_attr = attribute_weight
        self._w_color = color_weight
        self._queue: list = []
        self._lock = Lock()

    def enqueue_ingress(self, visit: ActiveVisit) -> None:
        with self._lock:
            self._queue.append(visit)
            self._queue.sort(key=lambda v: v.ingress_at)

    def active_visit_count(self) -> int:
        with self._lock:
            return len(self._queue)

    def match_egress(self, event: EgressEvent) -> Optional[Match]:
        with self._lock:
            if not self._queue:
                return None
            best, best_score, best_method = None, -1.0, "track_sequence"
            for idx, visit in enumerate(self._queue):
                pm = _plate_match(visit.plate, event.plate)
                qm = 1.0 if idx == 0 else max(0.0, 1.0 - 0.2 * idx)
                dt = (event.egress_at - visit.ingress_at).total_seconds()
                ts = max(0.0, 1.0 - abs(dt - self._expected) / max(self._expected, 1))
                attr_match = 0.0
                attr_count = 0
                if visit.make_model and event.make_model:
                    attr_count += 1
                    attr_match += 1.0 if visit.make_model == event.make_model else 0.0
                if visit.body_type and event.body_type:
                    attr_count += 1
                    attr_match += 1.0 if visit.body_type == event.body_type else 0.0
                attr_norm = (attr_match / attr_count) if attr_count else 0.0
                color_match = 1.0 if (visit.color and event.color and visit.color == event.color) else 0.0

                score = (
                    pm * self._w_plate
                    + qm * self._w_queue
                    + ts * self._w_time
                    + attr_norm * self._w_attr
                    + color_match * self._w_color
                )

                if score > best_score:
                    best_score = score
                    best = visit
                    if pm == 1.0:
                        best_method = "plate_exact"
                    elif pm > 0:
                        best_method = "plate_fuzzy"
                    elif _normalize(event.plate) is None and not any(
                        _normalize(v.plate) for v in self._queue
                    ):
                        # No plate signal anywhere — FIFO queue order is the decider.
                        best_method = "track_sequence"
                    elif ts > 0.5:
                        best_method = "lane_time_window"
                    else:
                        best_method = "track_sequence"

            if best is None or best_score < self._min:
                return None
            self._queue.remove(best)
            return Match(visit_id=best.visit_id, confidence=best_score, method=best_method)
