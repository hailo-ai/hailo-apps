# Car Wash License Plate Recognition — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 10-hour-stable, two-camera RTSP license plate recognition app on Hailo-8L that writes authoritative ingress plate reads to SQLite and matches egress reads back to open sessions.

**Architecture:** GStreamer `rtspsrc` per camera feeds a `Queue(maxsize=1)` (always-newest, no-backlog). A Python inference worker per camera runs Hailo YOLOv8 plate detector → crop → OCR → temporal voting → session tracking. Storage is SQLite (WAL mode) + JPEG snapshots. PaddleOCR is a confidence-gated fallback.

**Tech Stack:** Python 3, GStreamer (gi.repository.Gst), HailoInfer (hailo_platform), OpenCV (cv2), SQLite3, existing `hailo_apps.python.standalone_apps.paddle_ocr` for fallback.

---

## File Map

| Action | Path |
|--------|------|
| Create | `hailo_apps/python/standalone_apps/carwash_lpr/__init__.py` |
| Create | `hailo_apps/python/standalone_apps/carwash_lpr/storage.py` |
| Create | `hailo_apps/python/standalone_apps/carwash_lpr/voter.py` |
| Create | `hailo_apps/python/standalone_apps/carwash_lpr/session.py` |
| Create | `hailo_apps/python/standalone_apps/carwash_lpr/fallback.py` |
| Create | `hailo_apps/python/standalone_apps/carwash_lpr/inference.py` |
| Create | `hailo_apps/python/standalone_apps/carwash_lpr/ingest.py` |
| Create | `hailo_apps/python/standalone_apps/carwash_lpr/carwash_lpr.py` |
| Create | `hailo_apps/python/standalone_apps/carwash_lpr/README.md` |
| Create | `tests/standalone/test_carwash_lpr/__init__.py` |
| Create | `tests/standalone/test_carwash_lpr/test_storage.py` |
| Create | `tests/standalone/test_carwash_lpr/test_voter.py` |
| Create | `tests/standalone/test_carwash_lpr/test_session.py` |
| Create | `tests/standalone/test_carwash_lpr/test_fallback.py` |
| Create | `tests/standalone/test_carwash_lpr/test_inference.py` |

---

## Task 1: Scaffold

**Files:**
- Create: `hailo_apps/python/standalone_apps/carwash_lpr/__init__.py`
- Create: `tests/standalone/__init__.py`
- Create: `tests/standalone/test_carwash_lpr/__init__.py`

- [ ] **Step 1: Create app package**

```bash
mkdir -p hailo_apps/python/standalone_apps/carwash_lpr
touch hailo_apps/python/standalone_apps/carwash_lpr/__init__.py
```

- [ ] **Step 2: Create test package**

```bash
mkdir -p tests/standalone/test_carwash_lpr
touch tests/standalone/__init__.py
touch tests/standalone/test_carwash_lpr/__init__.py
```

- [ ] **Step 3: Verify import path works**

```bash
source setup_env.sh && python3 -c "import hailo_apps.python.standalone_apps.carwash_lpr; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add hailo_apps/python/standalone_apps/carwash_lpr/__init__.py \
        tests/standalone/__init__.py \
        tests/standalone/test_carwash_lpr/__init__.py
git commit -m "feat(carwash-lpr): scaffold package directories"
```

---

## Task 2: `storage.py` — SQLite + Snapshots

**Files:**
- Create: `hailo_apps/python/standalone_apps/carwash_lpr/storage.py`
- Create: `tests/standalone/test_carwash_lpr/test_storage.py`

- [ ] **Step 1: Write failing tests**

Create `tests/standalone/test_carwash_lpr/test_storage.py`:

```python
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

    # find_open_session returns row within window
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
    # Create 10001 dummy files
    for i in range(10001):
        (snaps / f"old_{i:05d}.jpg").write_bytes(b"x")
    store = ResultStorage(db_path=str(tmp_path / "p.db"), snapshot_dir=str(snaps))
    store.close()
    assert len(list(snaps.glob("*.jpg"))) <= 10000
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
source setup_env.sh && python3 -m pytest tests/standalone/test_carwash_lpr/test_storage.py -v 2>&1 | head -20
```
Expected: `ImportError` or `ModuleNotFoundError` (storage.py doesn't exist yet)

- [ ] **Step 3: Implement `storage.py`**

Create `hailo_apps/python/standalone_apps/carwash_lpr/storage.py`:

```python
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

import cv2

from hailo_apps.python.core.common.hailo_logger import get_logger

logger = get_logger(__name__)


class ResultStorage:
    def __init__(self, db_path: str, snapshot_dir: str):
        self.db_path = db_path
        self._snapshot_dir = Path(snapshot_dir)
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._prune_snapshots()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._setup_db()

    def _setup_db(self):
        with self._lock:
            self._conn.executescript("""
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;
                CREATE TABLE IF NOT EXISTS plate_reads (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id     TEXT NOT NULL,
                    timestamp     TEXT NOT NULL,
                    plate_string  TEXT NOT NULL,
                    confidence    REAL NOT NULL,
                    snapshot_path TEXT,
                    source        TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    ingress_read_id   INTEGER REFERENCES plate_reads(id),
                    egress_read_id    INTEGER REFERENCES plate_reads(id),
                    plate_string      TEXT NOT NULL,
                    egress_plate      TEXT,
                    entry_time        TEXT,
                    exit_time         TEXT,
                    duration_sec      INTEGER,
                    status            TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_status_entry
                    ON sessions(status, entry_time);
            """)
            self._conn.commit()

    def _prune_snapshots(self):
        files = sorted(self._snapshot_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime)
        excess = len(files) - 10000
        if excess > 0:
            for p in files[:excess]:
                p.unlink(missing_ok=True)
            logger.info(f"Pruned {excess} old snapshots")

    def save_snapshot(self, frame, camera_id: str, plate_string: str) -> str:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"cam_{camera_id}_{ts}_{plate_string}.jpg"
        path = self._snapshot_dir / filename
        cv2.imwrite(str(path), frame)
        return str(path)

    def write_plate_read(self, camera_id: str, timestamp: str, plate_string: str,
                         confidence: float, snapshot_path: str, source: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO plate_reads (camera_id, timestamp, plate_string, confidence, snapshot_path, source) "
                "VALUES (?,?,?,?,?,?)",
                (camera_id, timestamp, plate_string, confidence, snapshot_path, source),
            )
            self._conn.commit()
            return cur.lastrowid

    def create_session(self, ingress_read_id: int, plate_string: str, entry_time: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO sessions (ingress_read_id, plate_string, entry_time, status) "
                "VALUES (?,?,?,'open')",
                (ingress_read_id, plate_string, entry_time),
            )
            self._conn.commit()
            return cur.lastrowid

    def find_open_session_by_entry(self, min_entry: str, max_entry: str):
        """Return (session_id, plate_string, entry_time) or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT id, plate_string, entry_time FROM sessions "
                "WHERE status='open' AND entry_time >= ? AND entry_time <= ? "
                "ORDER BY entry_time ASC LIMIT 1",
                (min_entry, max_entry),
            ).fetchone()
        return row

    def confirm_session(self, session_id: int, egress_read_id: int,
                        exit_time: str, duration_sec: int):
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET egress_read_id=?, exit_time=?, duration_sec=?, "
                "status='confirmed' WHERE id=?",
                (egress_read_id, exit_time, duration_sec, session_id),
            )
            self._conn.commit()

    def review_session(self, session_id: int, egress_read_id: int, egress_plate: str,
                       exit_time: str, duration_sec: int):
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET egress_read_id=?, egress_plate=?, exit_time=?, "
                "duration_sec=?, status='needs_review' WHERE id=?",
                (egress_read_id, egress_plate, exit_time, duration_sec, session_id),
            )
            self._conn.commit()

    def write_egress_only(self, plate_string: str, timestamp: str,
                          confidence: float, snapshot_path: str, source: str) -> int:
        read_id = self.write_plate_read("egress", timestamp, plate_string, confidence,
                                        snapshot_path, source)
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions (egress_read_id, plate_string, exit_time, status) "
                "VALUES (?,?,?,'egress_only')",
                (read_id, plate_string, timestamp),
            )
            self._conn.commit()
        return read_id

    def close(self):
        self._conn.close()
```

- [ ] **Step 4: Run tests — expect pass**

```bash
source setup_env.sh && python3 -m pytest tests/standalone/test_carwash_lpr/test_storage.py -v
```
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add hailo_apps/python/standalone_apps/carwash_lpr/storage.py \
        tests/standalone/test_carwash_lpr/test_storage.py
git commit -m "feat(carwash-lpr): storage — SQLite WAL + snapshot save"
```

---

## Task 3: `voter.py` — TemporalVoter

**Files:**
- Create: `hailo_apps/python/standalone_apps/carwash_lpr/voter.py`
- Create: `tests/standalone/test_carwash_lpr/test_voter.py`

- [ ] **Step 1: Write failing tests**

Create `tests/standalone/test_carwash_lpr/test_voter.py`:

```python
import time
import numpy as np
import pytest

from hailo_apps.python.standalone_apps.carwash_lpr.voter import TemporalVoter, VotedPlate


def _dummy_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_no_emit_below_threshold():
    voter = TemporalVoter(window_size=10, emit_threshold=6, dedup_seconds=30,
                          fallback_threshold=0.60)
    results = []
    for _ in range(5):
        voted = voter.add("ABC123", 0.90, _dummy_frame())
        if voted:
            results.append(voted)
    assert results == []


def test_emit_at_threshold():
    voter = TemporalVoter(window_size=10, emit_threshold=6, dedup_seconds=30,
                          fallback_threshold=0.60)
    result = None
    for _ in range(6):
        result = voter.add("ABC123", 0.90, _dummy_frame())
    assert isinstance(result, VotedPlate)
    assert result.plate_string == "ABC123"
    assert result.confidence >= 0.6


def test_dedup_suppresses_same_plate(monkeypatch):
    voter = TemporalVoter(window_size=10, emit_threshold=6, dedup_seconds=30,
                          fallback_threshold=0.60)
    # First emission
    for _ in range(6):
        voter.add("ABC123", 0.90, _dummy_frame())
    # Second round — should be suppressed within dedup window
    voter._window.clear()
    results = []
    for _ in range(6):
        r = voter.add("ABC123", 0.90, _dummy_frame())
        if r:
            results.append(r)
    assert results == []


def test_dedup_allows_after_window(monkeypatch):
    voter = TemporalVoter(window_size=10, emit_threshold=6, dedup_seconds=1,
                          fallback_threshold=0.60)
    for _ in range(6):
        voter.add("ABC123", 0.90, _dummy_frame())
    time.sleep(1.1)
    voter._window.clear()
    result = None
    for _ in range(6):
        result = voter.add("ABC123", 0.90, _dummy_frame())
    assert result is not None


def test_needs_fallback_flag_when_low_confidence():
    voter = TemporalVoter(window_size=10, emit_threshold=6, dedup_seconds=30,
                          fallback_threshold=0.80)
    result = None
    for _ in range(6):
        result = voter.add("ABC123", 0.70, _dummy_frame())
    assert result is not None
    assert result.needs_fallback is True


def test_best_frame_is_highest_confidence_frame():
    voter = TemporalVoter(window_size=10, emit_threshold=6, dedup_seconds=30,
                          fallback_threshold=0.60)
    frames = [np.full((480, 640, 3), i, dtype=np.uint8) for i in range(6)]
    confidences = [0.70, 0.80, 0.95, 0.72, 0.68, 0.75]
    result = None
    for i in range(6):
        result = voter.add("ABC123", confidences[i], frames[i])
    assert result is not None
    assert np.array_equal(result.best_frame, frames[2])  # index 2 = 0.95
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
source setup_env.sh && python3 -m pytest tests/standalone/test_carwash_lpr/test_voter.py -v 2>&1 | head -10
```
Expected: `ImportError`

- [ ] **Step 3: Implement `voter.py`**

Create `hailo_apps/python/standalone_apps/carwash_lpr/voter.py`:

```python
import time
from collections import deque
from dataclasses import dataclass, field
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
```

- [ ] **Step 4: Run tests — expect pass**

```bash
source setup_env.sh && python3 -m pytest tests/standalone/test_carwash_lpr/test_voter.py -v
```
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add hailo_apps/python/standalone_apps/carwash_lpr/voter.py \
        tests/standalone/test_carwash_lpr/test_voter.py
git commit -m "feat(carwash-lpr): TemporalVoter — sliding window emit + dedup"
```

---

## Task 4: `session.py` — SessionTracker

**Files:**
- Create: `hailo_apps/python/standalone_apps/carwash_lpr/session.py`
- Create: `tests/standalone/test_carwash_lpr/test_session.py`

- [ ] **Step 1: Write failing tests**

Create `tests/standalone/test_carwash_lpr/test_session.py`:

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
source setup_env.sh && python3 -m pytest tests/standalone/test_carwash_lpr/test_session.py -v 2>&1 | head -10
```
Expected: `ImportError`

- [ ] **Step 3: Implement `session.py`**

Create `hailo_apps/python/standalone_apps/carwash_lpr/session.py`:

```python
from datetime import datetime, timedelta

import numpy as np

from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.standalone_apps.carwash_lpr.storage import ResultStorage

logger = get_logger(__name__)


class SessionTracker:
    def __init__(self, storage: ResultStorage, tunnel_min_minutes: int,
                 tunnel_max_minutes: int):
        self._storage = storage
        self._tunnel_min = tunnel_min_minutes
        self._tunnel_max = tunnel_max_minutes

    def handle_ingress(self, plate_string: str, confidence: float,
                       frame: np.ndarray, timestamp: str, source: str):
        snapshot_path = self._storage.save_snapshot(frame, "ingress", plate_string)
        read_id = self._storage.write_plate_read(
            camera_id="ingress",
            timestamp=timestamp,
            plate_string=plate_string,
            confidence=confidence,
            snapshot_path=snapshot_path,
            source=source,
        )
        self._storage.create_session(
            ingress_read_id=read_id,
            plate_string=plate_string,
            entry_time=timestamp,
        )
        logger.info(f"Ingress confirmed: {plate_string} at {timestamp}")

    def handle_egress(self, plate_string: str, confidence: float,
                      frame: np.ndarray, timestamp: str, source: str):
        # Compute lookup window relative to egress timestamp
        egress_dt = datetime.fromisoformat(timestamp)
        min_entry = (egress_dt - timedelta(minutes=self._tunnel_max)).isoformat()
        max_entry = (egress_dt - timedelta(minutes=self._tunnel_min)).isoformat()

        session = self._storage.find_open_session_by_entry(min_entry, max_entry)

        snapshot_path = self._storage.save_snapshot(frame, "egress", plate_string)
        read_id = self._storage.write_plate_read(
            camera_id="egress",
            timestamp=timestamp,
            plate_string=plate_string,
            confidence=confidence,
            snapshot_path=snapshot_path,
            source=source,
        )

        if session is None:
            self._storage.write_egress_only(
                plate_string=plate_string,
                timestamp=timestamp,
                confidence=confidence,
                snapshot_path=snapshot_path,
                source=source,
            )
            logger.warning(f"Egress-only (no matching ingress session): {plate_string}")
            return

        session_id, ingress_plate, entry_time = session
        entry_dt = datetime.fromisoformat(entry_time)
        duration_sec = int((egress_dt - entry_dt).total_seconds())

        if plate_string == ingress_plate:
            self._storage.confirm_session(session_id, read_id, timestamp, duration_sec)
            logger.info(f"Session confirmed: {plate_string}, {duration_sec}s in tunnel")
        else:
            self._storage.review_session(session_id, read_id, plate_string, timestamp, duration_sec)
            logger.warning(
                f"Session mismatch: ingress={ingress_plate}, egress={plate_string}"
            )
```

- [ ] **Step 4: Run tests — expect pass**

```bash
source setup_env.sh && python3 -m pytest tests/standalone/test_carwash_lpr/test_session.py -v
```
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add hailo_apps/python/standalone_apps/carwash_lpr/session.py \
        tests/standalone/test_carwash_lpr/test_session.py
git commit -m "feat(carwash-lpr): SessionTracker — ingress/egress lifecycle"
```

---

## Task 5: `fallback.py` — PaddleOCR Wrapper

**Files:**
- Create: `hailo_apps/python/standalone_apps/carwash_lpr/fallback.py`
- Create: `tests/standalone/test_carwash_lpr/test_fallback.py`

- [ ] **Step 1: Write failing tests**

Create `tests/standalone/test_carwash_lpr/test_fallback.py`:

```python
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from hailo_apps.python.standalone_apps.carwash_lpr.fallback import PaddleOCRFallback


def _crop():
    return np.zeros((128, 256, 3), dtype=np.uint8)


def test_returns_none_on_exception():
    fallback = PaddleOCRFallback.__new__(PaddleOCRFallback)
    fallback._run_ocr = MagicMock(side_effect=Exception("paddle error"))
    result = fallback.read_plate(_crop())
    assert result is None


def test_returns_plate_string_and_confidence():
    fallback = PaddleOCRFallback.__new__(PaddleOCRFallback)
    fallback._run_ocr = MagicMock(return_value=[("ABC123", 0.88)])
    result = fallback.read_plate(_crop())
    assert result is not None
    plate, conf = result
    assert plate == "ABC123"
    assert conf == pytest.approx(0.88)


def test_returns_none_on_empty_result():
    fallback = PaddleOCRFallback.__new__(PaddleOCRFallback)
    fallback._run_ocr = MagicMock(return_value=[])
    result = fallback.read_plate(_crop())
    assert result is None
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
source setup_env.sh && python3 -m pytest tests/standalone/test_carwash_lpr/test_fallback.py -v 2>&1 | head -10
```
Expected: `ImportError`

- [ ] **Step 3: Implement `fallback.py`**

Create `hailo_apps/python/standalone_apps/carwash_lpr/fallback.py`:

```python
from typing import Optional, Tuple

import numpy as np

from hailo_apps.python.core.common.hailo_logger import get_logger

logger = get_logger(__name__)

# Lazily imported — only load PaddleOCR dependencies on first use
_paddle_infer = None


class PaddleOCRFallback:
    def __init__(self, det_hef_path: str, ocr_hef_path: str):
        self._det_hef = det_hef_path
        self._ocr_hef = ocr_hef_path

    def _run_ocr(self, crop: np.ndarray):
        global _paddle_infer
        if _paddle_infer is None:
            from hailo_apps.python.standalone_apps.paddle_ocr.paddle_ocr import (
                run_inference_pipeline,
            )
            # Store reference so we can call it
            _paddle_infer = run_inference_pipeline
        # Call paddle_ocr_utils decode directly on the crop
        from hailo_apps.python.standalone_apps.paddle_ocr.paddle_ocr_utils import (
            decode_ocr_results,
        )
        from hailo_apps.python.core.common.hailo_inference import HailoInfer
        import queue, threading

        ocr_model = HailoInfer(self._ocr_hef)
        result_holder = []
        done = threading.Event()

        def callback(completion_info, bindings_list, _input_batch):
            if not completion_info.exception:
                buf = bindings_list[0].output().get_buffer()
                result_holder.extend(decode_ocr_results(buf, _PLATE_CHARS))
            done.set()

        h, w = ocr_model.get_input_shape()[:2]
        import cv2
        resized = cv2.resize(crop, (w, h))
        ocr_model.run([resized], callback)
        done.wait(timeout=5.0)
        return result_holder

    def read_plate(self, crop: np.ndarray) -> Optional[Tuple[str, float]]:
        try:
            results = self._run_ocr(crop)
            if not results:
                return None
            return results[0]
        except Exception as exc:
            logger.warning(f"PaddleOCR fallback failed: {exc}")
            return None


# Alphanumeric character set for US/CA plates (index 0 = CTC blank)
_PLATE_CHARS = [""] + list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
```

- [ ] **Step 4: Run tests — expect pass**

```bash
source setup_env.sh && python3 -m pytest tests/standalone/test_carwash_lpr/test_fallback.py -v
```
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add hailo_apps/python/standalone_apps/carwash_lpr/fallback.py \
        tests/standalone/test_carwash_lpr/test_fallback.py
git commit -m "feat(carwash-lpr): PaddleOCR fallback wrapper"
```

---

## Task 6: `inference.py` — Hailo Plate Detector + OCR

**Files:**
- Create: `hailo_apps/python/standalone_apps/carwash_lpr/inference.py`
- Create: `tests/standalone/test_carwash_lpr/test_inference.py`

- [ ] **Step 1: Write failing tests (mock HailoInfer)**

Create `tests/standalone/test_carwash_lpr/test_inference.py`:

```python
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from hailo_apps.python.standalone_apps.carwash_lpr.inference import PlateInference, PlateRead


def _make_det_output(boxes):
    """
    Simulate Hailo NMS detector output: list-of-classes, one class (license_plate).
    Each detection: [y1, x1, y2, x2, score] normalized 0-1.
    """
    return [np.array(boxes, dtype=np.float32)]


def _make_ocr_output(char_indices):
    """
    Simulate Hailo OCR output: shape [1, seq_len, num_chars].
    char_indices: list of ints (character indices into _PLATE_CHARS).
    """
    num_chars = 37  # blank + 36 alphanumeric
    seq = np.zeros((1, len(char_indices), num_chars), dtype=np.float32)
    for t, idx in enumerate(char_indices):
        seq[0, t, idx] = 1.0
    return seq


def test_no_detections_returns_empty():
    inf = PlateInference.__new__(PlateInference)
    inf._det_model = MagicMock()
    inf._ocr_model = MagicMock()
    inf._det_input_hw = (640, 640)
    inf._ocr_input_hw = (128, 256)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with patch.object(inf, '_run_detector', return_value=[]):
        reads = inf.run(frame)
    assert reads == []


def test_single_plate_detected_and_decoded():
    inf = PlateInference.__new__(PlateInference)
    inf._det_input_hw = (640, 640)
    inf._ocr_input_hw = (128, 256)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = [0.1, 0.2, 0.3, 0.5]  # [y1, x1, y2, x2] normalized

    # A=1, B=2, C=3 → indices 11,12,13 (0=blank, 1-10=digits, 11=A...)
    # _PLATE_CHARS = [""] + list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    # A=index 11, B=12, C=13, 1=1, 2=2, 3=3
    char_seq = [11, 12, 13, 1, 2, 3]  # "ABC123"
    ocr_out = _make_ocr_output(char_seq)

    with patch.object(inf, '_run_detector', return_value=[(bbox, 0.92)]):
        with patch.object(inf, '_run_ocr', return_value=ocr_out):
            reads = inf.run(frame)

    assert len(reads) == 1
    assert reads[0].plate_string == "ABC123"
    assert reads[0].confidence == pytest.approx(0.92)


def test_multiple_plates_sorted_left_to_right():
    inf = PlateInference.__new__(PlateInference)
    inf._det_input_hw = (640, 640)
    inf._ocr_input_hw = (128, 256)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Two plates: right one (x1=0.6) and left one (x1=0.1)
    # [y1, x1, y2, x2]
    bbox_right = [0.1, 0.6, 0.3, 0.9]
    bbox_left  = [0.1, 0.1, 0.3, 0.4]

    char_seq_right = [24, 25, 26, 1, 2, 3]  # "NOP123" (N=24, O=25, P=26)
    char_seq_left  = [11, 12, 13, 1, 2, 3]  # "ABC123"

    call_count = [0]
    def mock_ocr(crop):
        seq = char_seq_left if call_count[0] == 0 else char_seq_right
        call_count[0] += 1
        return _make_ocr_output(seq)

    with patch.object(inf, '_run_detector', return_value=[
        (bbox_right, 0.90), (bbox_left, 0.88)
    ]):
        with patch.object(inf, '_run_ocr', side_effect=mock_ocr):
            reads = inf.run(frame)

    # After L→R sort by x1: left (x1=0.1) first, then right (x1=0.6)
    assert len(reads) == 2
    assert reads[0].plate_string == "ABC123"
    assert reads[1].plate_string == "NOP123"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
source setup_env.sh && python3 -m pytest tests/standalone/test_carwash_lpr/test_inference.py -v 2>&1 | head -10
```
Expected: `ImportError`

- [ ] **Step 3: Implement `inference.py`**

Create `hailo_apps/python/standalone_apps/carwash_lpr/inference.py`:

```python
import threading
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from hailo_apps.python.core.common.hailo_inference import HailoInfer
from hailo_apps.python.core.common.hailo_logger import get_logger

logger = get_logger(__name__)

# blank + "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_PLATE_CHARS = [""] + list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


@dataclass
class PlateRead:
    plate_string: str
    confidence: float
    bbox: List[float]       # [y1, x1, y2, x2] normalized
    crop_frame: np.ndarray
    full_frame: np.ndarray


class PlateInference:
    def __init__(self, detector_hef: str, ocr_hef: str):
        self._det_model = HailoInfer(detector_hef)
        self._ocr_model = HailoInfer(ocr_hef)
        # Shape is (H, W, C)
        det_shape = self._det_model.get_input_shape()
        ocr_shape = self._ocr_model.get_input_shape()
        self._det_input_hw = (det_shape[0], det_shape[1])
        self._ocr_input_hw = (ocr_shape[0], ocr_shape[1])
        logger.info(f"Detector input: {self._det_input_hw}, OCR input: {self._ocr_input_hw}")

    def _preprocess(self, frame: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
        h, w = target_hw
        return cv2.resize(frame, (w, h))

    def _run_detector(self, frame: np.ndarray) -> List[Tuple[List[float], float]]:
        """Returns list of ([y1,x1,y2,x2], score) normalized."""
        processed = self._preprocess(frame, self._det_input_hw)
        result_holder = []
        done = threading.Event()

        def callback(completion_info, bindings_list, _input_batch):
            if completion_info.exception:
                logger.error(f"Detector error: {completion_info.exception}")
            else:
                raw = bindings_list[0].output().get_buffer()
                # raw: list of per-class arrays, shape [N, 5] = [y1,x1,y2,x2,score]
                for class_detections in raw:
                    for det in class_detections:
                        bbox = det[:4].tolist()
                        score = float(det[4])
                        if score > 0.3:
                            result_holder.append((bbox, score))
            done.set()

        self._det_model.run([processed], callback)
        done.wait(timeout=5.0)
        return result_holder

    def _run_ocr(self, crop: np.ndarray) -> np.ndarray:
        """Returns raw OCR model output array shape [1, seq_len, num_chars]."""
        processed = self._preprocess(crop, self._ocr_input_hw)
        result_holder = []
        done = threading.Event()

        def callback(completion_info, bindings_list, _input_batch):
            if completion_info.exception:
                logger.error(f"OCR error: {completion_info.exception}")
            else:
                result_holder.append(bindings_list[0].output().get_buffer())
            done.set()

        self._ocr_model.run([processed], callback)
        done.wait(timeout=5.0)
        return result_holder[0] if result_holder else np.zeros((1, 1, 37), dtype=np.float32)

    @staticmethod
    def _decode_ocr(raw: np.ndarray) -> str:
        """CTC greedy decode: argmax per timestep, remove blanks and repeats."""
        if raw.ndim == 3:
            raw = raw[0]  # (seq_len, num_chars)
        indices = raw.argmax(axis=1)
        chars = []
        prev = -1
        for idx in indices:
            if idx != 0 and idx != prev:  # 0 = CTC blank
                chars.append(_PLATE_CHARS[idx])
            prev = idx
        return "".join(chars)

    def _crop_bbox(self, frame: np.ndarray, bbox: List[float]) -> np.ndarray:
        h, w = frame.shape[:2]
        y1, x1, y2, x2 = bbox
        r = frame[
            max(0, int(y1 * h)):min(h, int(y2 * h)),
            max(0, int(x1 * w)):min(w, int(x2 * w)),
        ]
        return r if r.size > 0 else np.zeros((10, 10, 3), dtype=np.uint8)

    def run(self, frame: np.ndarray) -> List[PlateRead]:
        detections = self._run_detector(frame)
        if not detections:
            return []

        # Sort left-to-right by x1 (index 1 of bbox)
        detections.sort(key=lambda d: d[0][1])

        reads = []
        for bbox, score in detections:
            crop = self._crop_bbox(frame, bbox)
            raw_ocr = self._run_ocr(crop)
            plate_str = self._decode_ocr(raw_ocr)
            if plate_str:
                reads.append(PlateRead(
                    plate_string=plate_str,
                    confidence=score,
                    bbox=bbox,
                    crop_frame=crop,
                    full_frame=frame,
                ))
        return reads
```

- [ ] **Step 4: Run tests — expect pass**

```bash
source setup_env.sh && python3 -m pytest tests/standalone/test_carwash_lpr/test_inference.py -v
```
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add hailo_apps/python/standalone_apps/carwash_lpr/inference.py \
        tests/standalone/test_carwash_lpr/test_inference.py
git commit -m "feat(carwash-lpr): PlateInference — Hailo detector + OCR + CTC decode"
```

---

## Task 7: `ingest.py` — GStreamer RTSP Ingest

**Files:**
- Create: `hailo_apps/python/standalone_apps/carwash_lpr/ingest.py`

- [ ] **Step 1: Verify GStreamer Python bindings available**

```bash
source setup_env.sh && python3 -c "import gi; gi.require_version('Gst', '1.0'); from gi.repository import Gst; print('GStreamer OK')"
```
Expected: `GStreamer OK`

- [ ] **Step 2: Implement `ingest.py`**

Create `hailo_apps/python/standalone_apps/carwash_lpr/ingest.py`:

```python
import queue
import threading
import time

import numpy as np

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

from hailo_apps.python.core.common.hailo_logger import get_logger

logger = get_logger(__name__)

Gst.init(None)


class RTSPIngest:
    def __init__(self, rtsp_url: str, camera_id: str):
        self.camera_id = camera_id
        self._url = rtsp_url
        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._running = threading.Event()
        self._pipeline = None
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name=f"ingest-{camera_id}")

    def start(self):
        self._running.set()
        self._thread.start()

    def stop(self):
        self._running.clear()
        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)

    def _build_pipeline(self) -> Gst.Pipeline:
        pipeline_str = (
            f"rtspsrc location={self._url} latency=200 protocols=tcp name=src "
            f"! decodebin ! videoconvert ! video/x-raw,format=BGR "
            f"! appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false"
        )
        pipeline = Gst.parse_launch(pipeline_str)
        sink = pipeline.get_by_name("sink")
        sink.connect("new-sample", self._on_new_sample)
        return pipeline

    def _on_new_sample(self, sink) -> Gst.FlowReturn:
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR
        buf = sample.get_buffer()
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR
        frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape((height, width, 3)).copy()
        buf.unmap(map_info)
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass  # discard stale frame — keep newest
        return Gst.FlowReturn.OK

    def _run_loop(self):
        backoff = 2.0
        while self._running.is_set():
            try:
                self._pipeline = self._build_pipeline()
                self._pipeline.set_state(Gst.State.PLAYING)
                bus = self._pipeline.get_bus()
                while self._running.is_set():
                    msg = bus.timed_pop_filtered(
                        500 * Gst.MSECOND,
                        Gst.MessageType.ERROR | Gst.MessageType.EOS,
                    )
                    if msg:
                        if msg.type == Gst.MessageType.ERROR:
                            err, debug = msg.parse_error()
                            logger.error(f"[{self.camera_id}] GStreamer error: {err} — {debug}")
                        break
                self._pipeline.set_state(Gst.State.NULL)
            except Exception as exc:
                logger.error(f"[{self.camera_id}] Ingest exception: {exc}")

            if self._running.is_set():
                logger.info(f"[{self.camera_id}] Reconnecting in {backoff:.0f}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            else:
                break
        logger.info(f"[{self.camera_id}] Ingest stopped")
```

- [ ] **Step 3: Smoke-test with a test RTSP stream (optional — skip if no test stream)**

```bash
source setup_env.sh && python3 -c "
from hailo_apps.python.standalone_apps.carwash_lpr.ingest import RTSPIngest
import time
ing = RTSPIngest('rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4', 'test')
ing.start()
time.sleep(5)
frame = ing.frame_queue.get(timeout=3)
print('Got frame:', frame.shape)
ing.stop()
"
```
Expected: `Got frame: (H, W, 3)` — or a connection error if no network (acceptable).

- [ ] **Step 4: Commit**

```bash
git add hailo_apps/python/standalone_apps/carwash_lpr/ingest.py
git commit -m "feat(carwash-lpr): RTSPIngest — GStreamer appsink + exponential backoff reconnect"
```

---

## Task 8: `carwash_lpr.py` — Entry Point + Worker Loop

**Files:**
- Create: `hailo_apps/python/standalone_apps/carwash_lpr/carwash_lpr.py`

- [ ] **Step 1: Implement `carwash_lpr.py`**

Create `hailo_apps/python/standalone_apps/carwash_lpr/carwash_lpr.py`:

```python
#!/usr/bin/env python3
import argparse
import queue
import signal
import threading
from datetime import datetime, timezone

import numpy as np

from hailo_apps.python.core.common.hailo_logger import get_logger, init_logging
from hailo_apps.python.standalone_apps.carwash_lpr.fallback import PaddleOCRFallback
from hailo_apps.python.standalone_apps.carwash_lpr.inference import PlateInference
from hailo_apps.python.standalone_apps.carwash_lpr.ingest import RTSPIngest
from hailo_apps.python.standalone_apps.carwash_lpr.session import SessionTracker
from hailo_apps.python.standalone_apps.carwash_lpr.storage import ResultStorage
from hailo_apps.python.standalone_apps.carwash_lpr.voter import TemporalVoter

logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Car Wash License Plate Recognition")
    p.add_argument("--ingress", required=True, help="RTSP URL for ingress camera")
    p.add_argument("--egress",  required=True, help="RTSP URL for egress camera")
    p.add_argument("--detector-hef", required=True,
                   help="Path to yolov8n_relu6_lp HEF file")
    p.add_argument("--ocr-hef", required=True,
                   help="Path to yolov8n_relu6_lp_ocr HEF file")
    p.add_argument("--db", default="plates.db", help="SQLite database path")
    p.add_argument("--snapshot-dir", default="snapshots", help="Directory for JPEG snapshots")
    p.add_argument("--tunnel-min", type=int, default=3,
                   help="Minimum tunnel time in minutes (default: 3)")
    p.add_argument("--tunnel-max", type=int, default=8,
                   help="Maximum tunnel time in minutes (default: 8)")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def _make_voter(camera_id: str) -> TemporalVoter:
    if camera_id == "ingress":
        return TemporalVoter(window_size=15, emit_threshold=10,
                             dedup_seconds=45, fallback_threshold=0.70)
    return TemporalVoter(window_size=10, emit_threshold=6,
                         dedup_seconds=30, fallback_threshold=0.60)


def inference_worker(camera_id: str, frame_queue: queue.Queue,
                     inference: PlateInference, voter: TemporalVoter,
                     session_tracker: SessionTracker, fallback: PaddleOCRFallback,
                     running: threading.Event):
    logger.info(f"[{camera_id}] Inference worker started")
    while running.is_set():
        try:
            frame = frame_queue.get(timeout=5)
        except queue.Empty:
            continue

        reads = inference.run(frame)
        if not reads:
            continue

        # Use the highest-confidence read from this frame
        best = max(reads, key=lambda r: r.confidence)
        voted = voter.add(best.plate_string, best.confidence, best.full_frame)

        if voted is None:
            continue

        plate_str = voted.plate_string
        confidence = voted.confidence
        source = "hailo_ocr"

        if voted.needs_fallback:
            fb_result = fallback.read_plate(best.crop_frame)
            if fb_result:
                plate_str, confidence = fb_result
                source = "paddleocr"
                logger.info(f"[{camera_id}] Fallback improved: {plate_str} ({confidence:.2f})")

        timestamp = datetime.now(timezone.utc).isoformat()

        if camera_id == "ingress":
            session_tracker.handle_ingress(plate_str, confidence, voted.best_frame,
                                           timestamp, source)
        else:
            session_tracker.handle_egress(plate_str, confidence, voted.best_frame,
                                          timestamp, source)

    logger.info(f"[{camera_id}] Inference worker stopped")


def main():
    args = parse_args()
    init_logging(debug=args.debug)

    running = threading.Event()
    running.set()

    def handle_signal(sig, frame):
        logger.info("Shutdown signal received")
        running.clear()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    storage = ResultStorage(db_path=args.db, snapshot_dir=args.snapshot_dir)
    inference = PlateInference(detector_hef=args.detector_hef, ocr_hef=args.ocr_hef)
    fallback = PaddleOCRFallback(det_hef_path=args.detector_hef, ocr_hef_path=args.ocr_hef)
    session_tracker = SessionTracker(storage, args.tunnel_min, args.tunnel_max)

    ingress_ingest = RTSPIngest(args.ingress, "ingress")
    egress_ingest  = RTSPIngest(args.egress,  "egress")

    ingress_voter = _make_voter("ingress")
    egress_voter  = _make_voter("egress")

    workers = [
        threading.Thread(
            target=inference_worker,
            args=("ingress", ingress_ingest.frame_queue, inference,
                  ingress_voter, session_tracker, fallback, running),
            daemon=True, name="worker-ingress",
        ),
        threading.Thread(
            target=inference_worker,
            args=("egress", egress_ingest.frame_queue, inference,
                  egress_voter, session_tracker, fallback, running),
            daemon=True, name="worker-egress",
        ),
    ]

    try:
        ingress_ingest.start()
        egress_ingest.start()
        for w in workers:
            w.start()
        logger.info("Car wash LPR running. Ctrl-C to stop.")
        for w in workers:
            w.join()
    finally:
        ingress_ingest.stop()
        egress_ingest.stop()
        storage.close()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI help works**

```bash
source setup_env.sh && python3 -m hailo_apps.python.standalone_apps.carwash_lpr.carwash_lpr --help
```
Expected: Usage message with `--ingress`, `--egress`, `--detector-hef`, `--ocr-hef`

- [ ] **Step 3: Commit**

```bash
git add hailo_apps/python/standalone_apps/carwash_lpr/carwash_lpr.py
git commit -m "feat(carwash-lpr): main entry point + inference worker loop + signal handling"
```

---

## Task 9: README

**Files:**
- Create: `hailo_apps/python/standalone_apps/carwash_lpr/README.md`

- [ ] **Step 1: Write README**

Create `hailo_apps/python/standalone_apps/carwash_lpr/README.md`:

```markdown
# Car Wash License Plate Recognition

Two-camera RTSP → Hailo-8L license plate detection and session tracking.
Runs unattended for 10-hour shifts. Stores reads to SQLite and JPEG snapshots.

## Requirements

- Hailo-8L device
- HEF files:
  - `yolov8n_relu6_lp--640x640_quant_hailort_hailo8l_1.hef`
  - `yolov8n_relu6_lp_ocr--256x128_quant_hailort_hailo8l_1.hef`
- Two RTSP IP cameras (ingress + egress)

## Quick Start

```bash
source setup_env.sh
python3 -m hailo_apps.python.standalone_apps.carwash_lpr.carwash_lpr \
    --ingress rtsp://192.168.1.10/stream \
    --egress  rtsp://192.168.1.11/stream \
    --detector-hef /path/to/yolov8n_relu6_lp--640x640_quant_hailort_hailo8l_1.hef \
    --ocr-hef      /path/to/yolov8n_relu6_lp_ocr--256x128_quant_hailort_hailo8l_1.hef \
    --db           /data/plates.db \
    --snapshot-dir /data/snapshots
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--ingress` | required | RTSP URL for ingress camera |
| `--egress` | required | RTSP URL for egress camera |
| `--detector-hef` | required | Path to plate detector HEF |
| `--ocr-hef` | required | Path to plate OCR HEF |
| `--db` | `plates.db` | SQLite database path |
| `--snapshot-dir` | `snapshots/` | JPEG snapshot directory |
| `--tunnel-min` | `3` | Min tunnel time in minutes |
| `--tunnel-max` | `8` | Max tunnel time in minutes |
| `--debug` | false | Verbose logging |

## Database Schema

**`plate_reads`** — every confirmed plate detection  
**`sessions`** — ingress/egress lifecycle per car

Session statuses:
- `open` — car entered, not yet exited
- `confirmed` — egress plate matched ingress
- `needs_review` — ingress and egress plates differ
- `egress_only` — egress read with no matching ingress

## Querying Results

```sql
-- All sessions from today
SELECT * FROM sessions WHERE entry_time >= date('now');

-- Plates that need review
SELECT s.plate_string, s.egress_plate, s.entry_time
FROM sessions s WHERE s.status = 'needs_review';

-- Confirmed sessions for POS lookup
SELECT plate_string, entry_time FROM sessions
WHERE status = 'confirmed' ORDER BY entry_time DESC;
```
```

- [ ] **Step 2: Run all unit tests one final time**

```bash
source setup_env.sh && python3 -m pytest tests/standalone/test_carwash_lpr/ -v
```
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add hailo_apps/python/standalone_apps/carwash_lpr/README.md
git commit -m "feat(carwash-lpr): README with quickstart, options, and SQL query examples"
```

---

## Self-Review Checklist

- [x] **storage.py**: `write_egress_only` inserts a sessions row (not just a plate_reads row) — consistent with spec's `egress_only` status
- [x] **session.py**: uses `find_open_session_by_entry` (ISO string comparison) — consistent with storage.py method name
- [x] **voter.py**: `VotedPlate.needs_fallback` flag — consistent with `carwash_lpr.py` check `if voted.needs_fallback`
- [x] **inference.py**: `_run_detector` returns `(bbox, score)` tuples — consistent with `run()` usage
- [x] **carwash_lpr.py**: shares single `PlateInference` instance between both workers — correct (HailoInfer uses SHARED VDevice internally)
- [x] All imports are absolute (`hailo_apps.python...`)
- [x] Signal handling uses `threading.Event` flag — not direct sys.exit in handler
- [x] No `print()` — all operational messages use `logger`
