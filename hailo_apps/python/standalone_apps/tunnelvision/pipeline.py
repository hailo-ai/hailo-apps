import logging
import queue
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from hailo_apps.python.standalone_apps.tunnelvision.correlator import (
    ActiveVisit, Correlator, EgressEvent,
)
from hailo_apps.python.standalone_apps.tunnelvision.db import (
    InsertBestImage, InsertCameraObservation, InsertTrackStateEvent,
    InsertVehicleTrack, InsertVisit, InsertVisitMatchEvent,
    UpdateVehicleTrack, UpdateVisitStatus,
)
from hailo_apps.python.standalone_apps.tunnelvision.rekor import (
    CreditPolicy, RekorRequest, file_sha256, should_call_rekor,
)
from hailo_apps.python.standalone_apps.tunnelvision.scorer import (
    ScoringRule, compute_blur_score, compute_quality_score,
    frame_passes_gates, select_best_candidate,
)
from hailo_apps.python.standalone_apps.tunnelvision.state import (
    FrameCandidate, TrackState, VehicleTrack, transition,
)
from hailo_apps.python.standalone_apps.tunnelvision.tracker import IoUTracker
from hailo_apps.python.standalone_apps.tunnelvision.zone import ZoneConfig

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_dt() -> datetime:
    return datetime.now(timezone.utc)


def _centroid(bbox) -> tuple:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _plate_in_frame_coords(plate_bbox, vehicle_x: int, vehicle_y: int) -> list:
    x1, y1, x2, y2 = plate_bbox
    return [x1 + vehicle_x, y1 + vehicle_y, x2 + vehicle_x, y2 + vehicle_y]


class CameraPipeline(threading.Thread):
    """Per-camera frame loop. Wires tracker + zone + state + scorer +
    inference, emits DB write events to db_queue, Rekor requests to rekor_queue,
    and updates the shared correlator on ingress/egress."""

    def __init__(
        self,
        *,
        camera_role: str,
        camera_id: int,
        frame_queue: queue.Queue,
        db_queue: queue.Queue,
        rekor_queue: queue.Queue,
        correlator: Correlator,
        vehicle_detector,
        plate_detector,
        zone_config: ZoneConfig,
        scoring_rule: ScoringRule,
        credit_policy: CreditPolicy,
        snapshot_dir: str,
        secret_key: str = "",
        min_stable_candidates: int = 3,
        max_lost_frames: int = 10,
        iou_threshold: float = 0.3,
        name: Optional[str] = None,
    ):
        super().__init__(name=name or f"pipeline-{camera_role}", daemon=True)
        assert camera_role in ("ingress", "egress")
        self._camera_role = camera_role
        self._camera_id = camera_id
        self._in = frame_queue
        self._db = db_queue
        self._rekor = rekor_queue
        self._correlator = correlator
        self._vd = vehicle_detector
        self._pd = plate_detector
        self._zones = zone_config
        self._rule = scoring_rule
        self._policy = credit_policy
        self._secret = secret_key
        self._snap_dir = Path(snapshot_dir)
        self._snap_dir.mkdir(parents=True, exist_ok=True)
        self._min_stable = min_stable_candidates
        self._max_lost = max_lost_frames
        self._tracker = IoUTracker(iou_threshold=iou_threshold,
                                   max_lost_frames=max_lost_frames)
        self._tracks: dict = {}  # edge_track_id -> VehicleTrack
        self._prev_frame: Optional[np.ndarray] = None
        self._frame_index = 0
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=5)

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._in.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process(frame)
            except Exception as exc:  # noqa: BLE001
                logger.exception("pipeline %s frame failed: %s", self._camera_role, exc)
            finally:
                self._frame_index += 1

    def _process(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if not self._motion_gate(frame):
            self._prev_frame = frame
            self._update_no_detections()
            return

        vehicles = self._vd.detect(frame)
        if not vehicles:
            self._prev_frame = frame
            self._update_no_detections()
            return

        bboxes = [v.bbox for v in vehicles]
        scores = [v.score for v in vehicles]
        active_tracks = self._tracker.update(bboxes, scores)
        active_ids = {str(t.track_id) for t in active_tracks}

        # Mark previously-seen tracks that are missing this frame as lost
        for edge_id, track in list(self._tracks.items()):
            if edge_id not in active_ids:
                track.lost_frames += 1
                if track.lost_frames > self._max_lost:
                    track.state = TrackState.LOST
                    self._db.put(UpdateVehicleTrack(
                        id=track.db_id, state=TrackState.LOST.value,
                        last_seen_at=_now(),
                    ))
                    self._tracks.pop(edge_id, None)

        for raw in active_tracks:
            edge_id = str(raw.track_id)
            track = self._tracks.get(edge_id)
            if track is None:
                track = self._create_track(edge_id, raw.bbox)
            else:
                track.last_seen_at = _now_dt()
                track.current_bbox = list(raw.bbox)
                track.lost_frames = 0

            zone_label = self._zone_for_track(track, w, h)
            track.current_zone = zone_label

            fully_visible = self._is_fully_visible(track, w, h)
            track.fully_visible = fully_visible

            plate_visible = False
            plate_blocked = False
            plate_det = None

            if zone_label == "capture" and fully_visible and not track.processed:
                plate_det, plate_blocked = self._detect_plate(frame, track, active_tracks)
                plate_visible = plate_det is not None

                if plate_det is not None:
                    candidate = self._score_candidate(frame, track, plate_det)
                    if candidate is not None:
                        track.candidates.append(candidate)

                        if len(track.candidates) >= self._min_stable:
                            best = select_best_candidate(track.candidates)
                            self._select_and_emit(track, best)

            new_state = transition(
                track,
                current_zone=zone_label,
                fully_visible=fully_visible,
                plate_visible=plate_visible,
                plate_blocked=plate_blocked,
                max_lost_frames=self._max_lost,
            )
            self._maybe_log_state(track, new_state, zone_label)
            track.state = new_state

        self._prev_frame = frame

    def _update_no_detections(self) -> None:
        # advance internal tracker so it ages tracks correctly
        self._tracker.update([])
        for edge_id, track in list(self._tracks.items()):
            track.lost_frames += 1
            if track.lost_frames > self._max_lost:
                self._db.put(UpdateVehicleTrack(
                    id=track.db_id, state=TrackState.LOST.value,
                    last_seen_at=_now(),
                ))
                self._tracks.pop(edge_id, None)

    def _motion_gate(self, frame: np.ndarray) -> bool:
        if self._prev_frame is None:
            return True
        # Always process while we have active tracks to keep them alive.
        if self._tracks:
            return True
        diff = cv2.absdiff(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(self._prev_frame, cv2.COLOR_BGR2GRAY),
        )
        return float(np.mean(diff)) > 1.0

    def _create_track(self, edge_id: str, bbox: list) -> VehicleTrack:
        obs_id = str(uuid.uuid4())
        track_db_id = str(uuid.uuid4())
        now = _now_dt()
        track = VehicleTrack(
            track_id=int(edge_id),
            camera_role=self._camera_role,
            state=TrackState.NEW,
            first_seen_at=now,
            last_seen_at=now,
            current_bbox=list(bbox),
            db_id=track_db_id,
            observation_db_id=obs_id,
        )
        self._tracks[edge_id] = track
        self._db.put(InsertCameraObservation(
            id=obs_id, visit_id=None, camera_role=self._camera_role,
            started_at=now.isoformat(), observation_status="tracking",
        ))
        self._db.put(InsertVehicleTrack(
            id=track_db_id, camera_observation_id=obs_id,
            camera_role=self._camera_role, edge_track_id=edge_id,
            state=TrackState.NEW.value,
            first_seen_at=now.isoformat(), last_seen_at=now.isoformat(),
        ))
        return track

    def _zone_for_track(self, track: VehicleTrack, w: int, h: int) -> Optional[str]:
        cx, cy = _centroid(track.current_bbox)
        nx, ny = cx / w, cy / h
        # Priority order: capture_plate > end_visit > start_visit > skip
        # (capture is the work-doing zone; outer trigger zones are fallbacks).
        priority = {"capture_plate": 0, "end_visit": 1, "start_visit": 2, "skip": 3}
        label_map = {"capture_plate": "capture", "end_visit": "exit",
                     "start_visit": "approach", "skip": "ignore"}
        best_action = None
        best_rank = 99
        for zone in self._zones.for_camera(self._camera_id):
            if not zone.contains_point(nx, ny):
                continue
            rank = priority.get(zone.action, 99)
            if rank < best_rank:
                best_rank = rank
                best_action = zone.action
        return label_map.get(best_action) if best_action else None

    def _is_fully_visible(self, track: VehicleTrack, w: int, h: int) -> bool:
        x1, y1, x2, y2 = track.current_bbox
        margin = 5
        return x1 > margin and y1 > margin and x2 < w - margin and y2 < h - margin

    def _detect_plate(self, frame: np.ndarray, track: VehicleTrack, all_tracks):
        x1, y1, x2, y2 = map(int, track.current_bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None, False
        crop = frame[y1:y2, x1:x2]
        plates = self._pd.detect(crop)
        if not plates:
            return None, False
        plate = plates[0]
        # naive blockage check: is another track's bbox overlapping the plate area?
        plate_in_frame = _plate_in_frame_coords(plate.bbox, x1, y1)
        for other in all_tracks:
            if str(other.track_id) == str(track.track_id):
                continue
            ox1, oy1, ox2, oy2 = other.bbox
            if (ox1 < plate_in_frame[2] and ox2 > plate_in_frame[0]
                    and oy1 < plate_in_frame[3] and oy2 > plate_in_frame[1]):
                return None, True
        return plate, False

    def _score_candidate(self, frame, track, plate_det) -> Optional[FrameCandidate]:
        x1, y1, x2, y2 = map(int, track.current_bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        veh_crop = frame[y1:y2, x1:x2]
        if veh_crop.size == 0:
            return None
        px1, py1, px2, py2 = map(int, plate_det.bbox)
        plate_crop = veh_crop[max(0, py1):py2, max(0, px1):px2]
        if plate_crop.size == 0:
            return None
        blur = compute_blur_score(plate_crop)
        plate_w_px = max(0, px2 - px1)

        if track.candidates:
            last_bbox = track.candidates[-1].vehicle_bbox
            dx = (track.current_bbox[0] + track.current_bbox[2]) / 2 - (last_bbox[0] + last_bbox[2]) / 2
            dy = (track.current_bbox[1] + track.current_bbox[3]) / 2 - (last_bbox[1] + last_bbox[3]) / 2
            motion = float(np.hypot(dx, dy))
        else:
            motion = 0.0
        bbox_stability = 1.0 / (1.0 + motion / 50.0)
        roi_fit = 1.0
        bbox_growth_rate = 0.0

        if not frame_passes_gates(
            plate_confidence=plate_det.score,
            blur_score=blur,
            plate_width_px=plate_w_px,
            motion_score=motion,
            bbox_growth_rate=bbox_growth_rate,
            fully_visible=track.fully_visible,
            plate_visible=True,
            plate_blocked=False,
            rule=self._rule,
        ):
            return None

        quality = compute_quality_score(
            plate_confidence=plate_det.score, blur_score=blur,
            plate_width_px=plate_w_px, bbox_stability=bbox_stability,
            roi_fit=roi_fit, motion_score=motion, plate_blocked=False,
        )
        return FrameCandidate(
            frame=frame.copy(), quality_score=quality, blur_score=blur,
            motion_score=motion, plate_width_px=plate_w_px,
            plate_confidence=plate_det.score, bbox_stability=bbox_stability,
            vehicle_bbox=list(track.current_bbox),
            plate_bbox=_plate_in_frame_coords(plate_det.bbox, x1, y1),
        )

    def _select_and_emit(self, track: VehicleTrack, best: FrameCandidate) -> None:
        if track.processed:
            return
        ts = _now()
        filename = f"{self._camera_role}_{track.observation_db_id}.jpg"
        path = self._snap_dir / filename
        cv2.imwrite(str(path), best.frame)
        sha = file_sha256(str(path))
        best_image_id = str(uuid.uuid4())
        track.best_image_path = str(path)
        track.best_candidate = best

        self._db.put(InsertBestImage(
            id=best_image_id, vehicle_track_id=track.db_id,
            observation_id=track.observation_db_id,
            image_path=str(path), image_sha256=sha,
            width=best.frame.shape[1], height=best.frame.shape[0],
            quality_score=best.quality_score, blur_score=best.blur_score,
            motion_score=best.motion_score, plate_width_px=best.plate_width_px,
            car_fully_visible=int(track.fully_visible),
            plate_visible=1, plate_blocked=0,
            vehicle_bbox=str(best.vehicle_bbox), plate_bbox=str(best.plate_bbox),
            selected_at=ts,
        ))
        self._db.put(UpdateVehicleTrack(
            id=track.db_id, state=TrackState.BEST_IMAGE_SELECTED.value,
            last_seen_at=ts, processed=1, best_image_id=best_image_id,
        ))
        track.processed = True

        call, recv, reason = should_call_rekor(
            quality_score=best.quality_score,
            camera_role=self._camera_role,
            credits_used=0, monthly_budget=self._policy.monthly_credit_budget,
            known_recently=False, policy=self._policy,
        )
        if call:
            req_id = str(uuid.uuid4())
            self._rekor.put(RekorRequest(
                request_id=req_id, image_path=str(path), image_sha256=sha,
                observation_id=track.observation_db_id,
                vehicle_track_id=track.db_id, best_image_id=best_image_id,
                visit_id=None, recognize_vehicle=recv, credit_policy=reason,
            ))
            track.rekor_sent = True

        if self._camera_role == "ingress":
            visit_id = str(uuid.uuid4())
            self._db.put(InsertVisit(
                id=visit_id, status="ingress_captured", started_at=ts,
                ingress_observation_id=track.observation_db_id,
                ingress_captured_at=ts,
            ))
            self._correlator.enqueue_ingress(ActiveVisit(
                visit_id=visit_id, plate=None,
                ingress_at=_now_dt(),
            ))
        else:  # egress
            match = self._correlator.match_egress(EgressEvent(
                plate=None, egress_at=_now_dt(),
            ))
            if match:
                self._db.put(UpdateVisitStatus(
                    id=match.visit_id, status="completed",
                    egress_observation_id=track.observation_db_id,
                    egress_captured_at=ts, completed_at=ts,
                    match_confidence=match.confidence,
                ))
                self._db.put(InsertVisitMatchEvent(
                    id=str(uuid.uuid4()),
                    ingress_visit_id=match.visit_id,
                    egress_observation_id=track.observation_db_id,
                    match_method=match.method,
                    match_confidence=match.confidence,
                    matched_at=ts,
                ))

    def _maybe_log_state(self, track: VehicleTrack, new_state: TrackState,
                         zone_label: Optional[str]) -> None:
        if new_state == track.state:
            return
        self._db.put(InsertTrackStateEvent(
            id=str(uuid.uuid4()),
            vehicle_track_id=track.db_id,
            previous_state=track.state.value,
            next_state=new_state.value,
            reason=None, zone_type=zone_label,
            frame_index=self._frame_index, created_at=_now(),
        ))
