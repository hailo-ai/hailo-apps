import os
import queue
import threading
import time
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest

from hailo_apps.python.standalone_apps.tunnelvision.rekor import (
    RekorRequest, RekorWorker, CreditPolicy,
)
from hailo_apps.python.standalone_apps.tunnelvision.db import (
    InsertRekorRequest, UpdateRekorRequest, InsertRekorResponse,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def test_worker_drains_queue_and_persists(tmp_path):
    img = tmp_path / "best.jpg"
    img.write_bytes(b"fake-jpeg")

    rekor_q: queue.Queue = queue.Queue()
    db_q: queue.Queue = queue.Queue()

    sample = {
        "results": [{"plate": "XY", "confidence": 99.0, "candidates": []}],
        "credit_cost": 1, "credits_monthly_used": 5, "credits_monthly_total": 100,
        "processing_time": {"total": 50.0},
    }
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = sample
    fake_resp.raise_for_status.return_value = None

    with patch("hailo_apps.python.standalone_apps.tunnelvision.rekor.requests.post",
               return_value=fake_resp):
        w = RekorWorker(rekor_q, db_q, secret_key="SK", policy=CreditPolicy())
        w.start()
        try:
            rekor_q.put(RekorRequest(
                request_id="r1", image_path=str(img), image_sha256="h",
                observation_id="o1", vehicle_track_id="t1", best_image_id="b1",
                visit_id=None, recognize_vehicle=False, credit_policy="plate_only",
            ))
            rekor_q.join()
        finally:
            w.stop()

    events = []
    while not db_q.empty():
        events.append(db_q.get_nowait())

    assert any(isinstance(e, InsertRekorRequest) for e in events)
    assert any(isinstance(e, UpdateRekorRequest) and e.request_status == "succeeded" for e in events)
    assert any(isinstance(e, InsertRekorResponse) for e in events)


def test_worker_marks_failed_on_http_error(tmp_path):
    img = tmp_path / "best.jpg"
    img.write_bytes(b"fake-jpeg")
    rekor_q: queue.Queue = queue.Queue()
    db_q: queue.Queue = queue.Queue()

    with patch("hailo_apps.python.standalone_apps.tunnelvision.rekor.requests.post",
               side_effect=Exception("boom")):
        w = RekorWorker(rekor_q, db_q, secret_key="SK", policy=CreditPolicy(),
                        max_retries=1, retry_backoff=0.01)
        w.start()
        try:
            rekor_q.put(RekorRequest(
                request_id="r2", image_path=str(img), image_sha256="h",
                observation_id="o", vehicle_track_id="t", best_image_id="b",
                visit_id=None, recognize_vehicle=False, credit_policy="plate_only",
            ))
            rekor_q.join()
        finally:
            w.stop()

    statuses = [e.request_status for e in list(db_q.queue) if isinstance(e, UpdateRekorRequest)]
    assert "failed" in statuses
