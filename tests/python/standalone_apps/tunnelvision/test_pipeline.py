import queue
import threading
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

from hailo_apps.python.standalone_apps.tunnelvision.pipeline import CameraPipeline
from hailo_apps.python.standalone_apps.tunnelvision.zone import Zone, ZoneConfig
from hailo_apps.python.standalone_apps.tunnelvision.scorer import ScoringRule
from hailo_apps.python.standalone_apps.tunnelvision.rekor import CreditPolicy
from hailo_apps.python.standalone_apps.tunnelvision.correlator import Correlator


def _frame() -> np.ndarray:
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 255, size=(720, 1280, 3), dtype=np.uint8)


def _zone_config_full_frame_capture() -> ZoneConfig:
    return ZoneConfig(zones=[
        Zone(name="ingress_gate", zone_type="trigger", action="start_visit",
             camera_id=0, polygon=[[0, 0], [1, 0], [1, 1], [0, 1]]),
        Zone(name="lpr_zone", zone_type="alpr", action="capture_plate",
             camera_id=0, polygon=[[0, 0], [1, 0], [1, 1], [0, 1]]),
    ])


def test_pipeline_emits_db_events_on_full_capture_cycle(tmp_path):
    """Feed the pipeline ~5 frames with a stationary detected vehicle and a
    plate detection that passes all gates; verify it persists best_image,
    queues a Rekor request, and enqueues an ingress visit on the correlator."""
    frame_q: queue.Queue = queue.Queue()
    db_q: queue.Queue = queue.Queue()
    rekor_q: queue.Queue = queue.Queue()
    correlator = Correlator()

    vehicle_det = MagicMock()
    vehicle_det.detect.return_value = [MagicMock(bbox=[400, 300, 800, 600],
                                                 score=0.95, label="car")]
    plate_det = MagicMock()
    plate_det.detect.return_value = [MagicMock(bbox=[40, 200, 200, 240], score=0.92)]

    pipe = CameraPipeline(
        camera_role="ingress",
        camera_id=0,
        frame_queue=frame_q,
        db_queue=db_q,
        rekor_queue=rekor_q,
        correlator=correlator,
        vehicle_detector=vehicle_det,
        plate_detector=plate_det,
        zone_config=_zone_config_full_frame_capture(),
        scoring_rule=ScoringRule(min_blur_score=0.0, min_plate_width_px=10),
        credit_policy=CreditPolicy(min_quality_score_to_call=0.0),
        snapshot_dir=str(tmp_path / "snaps"),
        min_stable_candidates=2,
    )
    pipe.start()
    try:
        for _ in range(5):
            frame_q.put(_frame())
        deadline = time.time() + 5.0
        while time.time() < deadline and rekor_q.empty():
            time.sleep(0.05)
    finally:
        pipe.stop()

    assert not rekor_q.empty(), "expected at least one Rekor request enqueued"
    db_events = []
    while not db_q.empty():
        db_events.append(db_q.get_nowait())
    types = {type(e).__name__ for e in db_events}
    assert "InsertCameraObservation" in types
    assert "InsertVehicleTrack" in types
    assert "InsertBestImage" in types
    assert "InsertVisit" in types
    assert correlator.active_visit_count() == 1
