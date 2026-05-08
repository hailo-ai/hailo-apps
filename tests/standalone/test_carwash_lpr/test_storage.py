import sqlite3
import tempfile
import os
import numpy as np
import pytest
from pathlib import Path

from hailo_apps.python.standalone_apps.carwash_lpr.storage import ResultStorage


@pytest.fixture
def store(tmp_path):
    db = str(tmp_path / "plates.db")
    snaps = str(tmp_path / "snapshots")
    s = ResultStorage(db_path=db, snapshot_dir=snaps)
    yield s
    s.close()


def test_schema_created(store, tmp_path):
    conn = sqlite3.connect(str(tmp_path / "plates.db"))
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "plate_reads" in tables
    assert "sessions" in tables
    conn.close()


def test_write_plate_read_returns_id(store):
    rid = store.write_plate_read(
        camera_id="ingress",
        timestamp="2026-05-06T10:00:00",
        plate_string="ABC123",
        confidence=0.95,
        snapshot_path="/tmp/snap.jpg",
        source="hailo_ocr",
    )
    assert isinstance(rid, int)
    assert rid > 0


def test_create_and_find_open_session(store):
    rid = store.write_plate_read("ingress", "2026-05-06T10:00:00", "ABC123", 0.95, "", "hailo_ocr")
    sid = store.create_session(ingress_read_id=rid, plate_string="ABC123", entry_time="2026-05-06T10:00:00")
    assert isinstance(sid, int)

    result = store.find_open_session_by_entry(
        min_entry="2026-05-06T09:55:00",
        max_entry="2026-05-06T10:08:00",
    )
    assert result is not None
    session_id, plate, entry = result
    assert plate == "ABC123"


def test_confirm_session(store):
    rid = store.write_plate_read("ingress", "2026-05-06T10:00:00", "ABC123", 0.95, "", "hailo_ocr")
    sid = store.create_session(rid, "ABC123", "2026-05-06T10:00:00")
    eid = store.write_plate_read("egress", "2026-05-06T10:05:00", "ABC123", 0.92, "", "hailo_ocr")
    store.confirm_session(session_id=sid, egress_read_id=eid, exit_time="2026-05-06T10:05:00", duration_sec=300)

    conn = sqlite3.connect(store.db_path)
    row = conn.execute("SELECT status, egress_read_id FROM sessions WHERE id=?", (sid,)).fetchone()
    conn.close()
    assert row[0] == "confirmed"
    assert row[1] == eid


def test_review_session(store):
    rid = store.write_plate_read("ingress", "2026-05-06T10:00:00", "ABC123", 0.95, "", "hailo_ocr")
    sid = store.create_session(rid, "ABC123", "2026-05-06T10:00:00")
    eid = store.write_plate_read("egress", "2026-05-06T10:05:00", "XYZ999", 0.88, "", "hailo_ocr")
    store.review_session(sid, eid, "XYZ999", "2026-05-06T10:05:00", 300)

    conn = sqlite3.connect(store.db_path)
    row = conn.execute("SELECT status, egress_plate FROM sessions WHERE id=?", (sid,)).fetchone()
    conn.close()
    assert row[0] == "needs_review"
    assert row[1] == "XYZ999"


def test_save_snapshot_creates_file(store, tmp_path):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    path = store.save_snapshot(frame, "ingress", "ABC123")
    assert Path(path).exists()
    assert Path(path).suffix == ".jpg"


def test_snapshot_dir_pruned_at_10001(tmp_path):
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    for i in range(10001):
        (snaps / f"old_{i:05d}.jpg").write_bytes(b"x")
    store = ResultStorage(db_path=str(tmp_path / "p.db"), snapshot_dir=str(snaps))
    store.close()
    assert len(list(snaps.glob("*.jpg"))) <= 10000
