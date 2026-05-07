import queue
import sqlite3
import time
import uuid
from datetime import datetime, timezone

import pytest

from hailo_apps.python.standalone_apps.tunnelvision.db import (
    DBWriter,
    InsertVisit,
    InsertCameraObservation,
    InsertVehicleTrack,
    InsertTrackStateEvent,
    InsertBestImage,
    init_schema,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def test_db_writer_persists_visit(temp_db_path):
    q: queue.Queue = queue.Queue()
    writer = DBWriter(temp_db_path, q)
    writer.start()
    try:
        visit_id = str(uuid.uuid4())
        q.put(InsertVisit(
            id=visit_id, status="approaching", started_at=_now(),
        ))
        q.join()  # wait for writer to drain
    finally:
        writer.stop()

    conn = sqlite3.connect(temp_db_path)
    row = conn.execute(
        "SELECT id, status FROM visits WHERE id=?", (visit_id,)
    ).fetchone()
    assert row == (visit_id, "approaching")


def test_db_writer_handles_chain_of_events(temp_db_path):
    q: queue.Queue = queue.Queue()
    writer = DBWriter(temp_db_path, q)
    writer.start()
    try:
        obs_id = str(uuid.uuid4())
        track_id = str(uuid.uuid4())
        q.put(InsertCameraObservation(
            id=obs_id, visit_id=None, camera_role="ingress",
            started_at=_now(), observation_status="tracking",
        ))
        q.put(InsertVehicleTrack(
            id=track_id, camera_observation_id=obs_id, camera_role="ingress",
            edge_track_id="42", state="new",
            first_seen_at=_now(), last_seen_at=_now(),
        ))
        q.put(InsertTrackStateEvent(
            id=str(uuid.uuid4()), vehicle_track_id=track_id,
            previous_state="new", next_state="tracking",
            reason="vehicle detected", zone_type="approach",
            frame_index=0, created_at=_now(),
        ))
        q.join()
    finally:
        writer.stop()

    conn = sqlite3.connect(temp_db_path)
    assert conn.execute("SELECT COUNT(*) FROM camera_observations").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM vehicle_tracks").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM track_state_events").fetchone()[0] == 1


def test_db_writer_survives_event_error(temp_db_path):
    """A bad event should be logged but not kill the writer."""
    q: queue.Queue = queue.Queue()
    writer = DBWriter(temp_db_path, q)
    writer.start()
    try:
        q.put(InsertVisit(id="v1", status="bogus_status", started_at=_now()))  # CHECK fails
        q.put(InsertVisit(id="v2", status="approaching", started_at=_now()))   # ok
        q.join()
    finally:
        writer.stop()

    conn = sqlite3.connect(temp_db_path)
    row = conn.execute("SELECT id FROM visits WHERE id='v2'").fetchone()
    assert row == ("v2",), "writer should keep running after a bad event"
