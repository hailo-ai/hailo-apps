import sqlite3

import pytest

from hailo_apps.python.standalone_apps.tunnelvision.db import init_schema

EXPECTED_TABLES = {
    "visits",
    "camera_observations",
    "vehicle_tracks",
    "track_state_events",
    "best_images",
    "rekor_carcheck_requests",
    "rekor_carcheck_responses",
    "rekor_plate_results",
    "rekor_plate_candidates",
    "plates",
    "visit_match_events",
}


def test_init_schema_creates_all_tables(temp_db_path):
    conn = sqlite3.connect(temp_db_path)
    init_schema(conn)
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    names = {r[0] for r in rows}
    assert EXPECTED_TABLES.issubset(names), f"Missing: {EXPECTED_TABLES - names}"


def test_init_schema_is_idempotent(temp_db_path):
    conn = sqlite3.connect(temp_db_path)
    init_schema(conn)
    init_schema(conn)  # should not raise


def test_visits_status_constraint(temp_db_path):
    conn = sqlite3.connect(temp_db_path)
    init_schema(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO visits(id, status, started_at) VALUES (?, ?, ?)",
            ("v1", "garbage_status", "2026-05-07T00:00:00Z"),
        )
        conn.commit()
