import sqlite3
import tempfile
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from hailo_apps.python.standalone_apps.carwash_lpr.storage import ResultStorage
from hailo_apps.python.standalone_apps.carwash_lpr.session import SessionTracker


@pytest.fixture
def store(tmp_path):
    s = ResultStorage(str(tmp_path / "p.db"), str(tmp_path / "snaps"))
    yield s
    s.close()


def _frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_ingress_creates_open_session(store):
    tracker = SessionTracker(store, tunnel_min_minutes=3, tunnel_max_minutes=8)
    tracker.handle_ingress("ABC123", 0.95, _frame(), "2026-05-06T10:00:00", "hailo_ocr")

    conn = sqlite3.connect(store.db_path)
    row = conn.execute("SELECT status, plate_string FROM sessions").fetchone()
    conn.close()
    assert row[0] == "open"
    assert row[1] == "ABC123"


def test_egress_confirms_matching_session(store):
    tracker = SessionTracker(store, tunnel_min_minutes=0, tunnel_max_minutes=60)
    tracker.handle_ingress("ABC123", 0.95, _frame(), "2026-05-06T10:00:00", "hailo_ocr")
    tracker.handle_egress("ABC123", 0.92, _frame(), "2026-05-06T10:05:00", "hailo_ocr")

    conn = sqlite3.connect(store.db_path)
    row = conn.execute("SELECT status, egress_plate FROM sessions").fetchone()
    conn.close()
    assert row[0] == "confirmed"
    assert row[1] is None  # no mismatch


def test_egress_flags_mismatch(store):
    tracker = SessionTracker(store, tunnel_min_minutes=0, tunnel_max_minutes=60)
    tracker.handle_ingress("ABC123", 0.95, _frame(), "2026-05-06T10:00:00", "hailo_ocr")
    tracker.handle_egress("XYZ999", 0.88, _frame(), "2026-05-06T10:05:00", "hailo_ocr")

    conn = sqlite3.connect(store.db_path)
    row = conn.execute("SELECT status, egress_plate FROM sessions").fetchone()
    conn.close()
    assert row[0] == "needs_review"
    assert row[1] == "XYZ999"


def test_egress_only_when_no_open_session(store):
    tracker = SessionTracker(store, tunnel_min_minutes=3, tunnel_max_minutes=8)
    # No ingress recorded — egress arrives alone
    tracker.handle_egress("ABC123", 0.88, _frame(), "2026-05-06T10:05:00", "hailo_ocr")

    conn = sqlite3.connect(store.db_path)
    row = conn.execute("SELECT status FROM sessions").fetchone()
    conn.close()
    assert row[0] == "egress_only"
