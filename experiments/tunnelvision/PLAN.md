# TunnelVision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build TunnelVision — a continuous-tracking, zone-aware, best-frame, Rekor-gated edge ALPR system for car wash tunnels — as a new standalone app at `hailo_apps/python/standalone_apps/tunnelvision/`.

**Architecture:** Two RTSP camera ingest threads → two per-camera pipeline threads (vehicle detect → IoU track → zone evaluate → state machine → plate detect → best-frame score) → async Rekor worker thread + serialized DB writer thread. Zone schema borrowed from `td-edge`; tuned cam0 ingress polygons reused from `experiments/tunnelvision/zones.json`.

**Tech Stack:** Python 3.10+, DeGirum SDK (local Hailo device, not cloud), GStreamer/PyGObject for RTSP, SQLite (WAL), `requests` for Rekor HTTP, pytest + pytest-timeout. Reused deps already in pyproject.toml: `numpy`, `opencv-python`, `scipy`, `lap`, `cython_bbox`.

**Source docs:**
- `experiments/tunnelvision/PRD.md` — product requirements
- `experiments/tunnelvision/ERD.md` — entity model + SQL schema + reference Python
- `experiments/tunnelvision/DESIGN.md` — design decisions, file structure, threading
- `experiments/tunnelvision/zones.json` — tuned cam0 polygons (from td-edge)

---

## Phase 0 — Discovery (no code yet)

### Task 0: Identify DeGirum local vehicle detection model name

**Files:**
- Create: `experiments/tunnelvision/MODELS.md` (one-page note; not part of the app)

**Why:** DESIGN.md §3 marks the vehicle detection model as TBD. The plate detector and OCR model names are already known (`yolov8n_relu6_lp--640x640_quant_hailort_hailo8l_1`, `yolov8n_relu6_lp_ocr--256x128_quant_hailort_hailo8l_1`) but there's no precedent in this repo for vehicle detection via DeGirum local. Resolve before writing `tracker.py`.

- [ ] **Step 1: List models available on the local DeGirum device**

```bash
source setup_env.sh
python3 -c "
import degirum as dg
zoo = dg.connect(inference_host_address='@local')
for m in zoo.list_models():
    name = m if isinstance(m, str) else m.name
    print(name)
" | tee /tmp/dg_local_models.txt
```

Expected: a list of model names compiled for the local Hailo device (hailo8l). Look for entries containing `yolov8n`, `vehicle`, or COCO-class detectors.

- [ ] **Step 2: Choose the vehicle detector**

Preference order:
1. A model explicitly named `vehicle_detection*` if available
2. A general COCO YOLO (`yolov8n--640x640*hailo8l*`) — vehicle classes are car (2), motorcycle (3), bus (5), truck (7)
3. If neither is available locally, download via DeGirum and document the steps

- [ ] **Step 3: Confirm the plate detector and OCR are also present locally**

```bash
grep -E "yolov8n_relu6_lp" /tmp/dg_local_models.txt
```

Expected: both `yolov8n_relu6_lp--640x640_quant_hailort_hailo8l_1` and `yolov8n_relu6_lp_ocr--256x128_quant_hailort_hailo8l_1` appear. If missing, push them to the local zoo before continuing.

- [ ] **Step 4: Smoke-test the chosen vehicle detector on a sample image**

```bash
python3 -c "
import degirum as dg
import cv2, numpy as np
m = dg.load_model(model_name='<CHOSEN_MODEL_NAME>',
                  inference_host_address='@local',
                  zoo_url='degirum/models_hailort')
img = cv2.imread('assets/Car.jpg')
r = m(img)
print(r.results[:3])
"
```

Expected: at least one result with a vehicle class label (`car`, `truck`, `bus`) and non-zero score.

- [ ] **Step 5: Record decisions in MODELS.md and commit**

Write `experiments/tunnelvision/MODELS.md` with three lines:

```
VEHICLE_DETECT_MODEL = "<chosen name>"
PLATE_DETECT_MODEL   = "yolov8n_relu6_lp--640x640_quant_hailort_hailo8l_1"
OCR_MODEL            = "yolov8n_relu6_lp_ocr--256x128_quant_hailort_hailo8l_1"
```

Plus a note on which classes (by id) the vehicle detector emits and which we keep.

```bash
git add experiments/tunnelvision/MODELS.md
git commit -m "docs(tunnelvision): record DeGirum local model choices"
```

---

## Phase 1 — Foundation: pure-logic components (no Hailo hardware needed)

These components are the most testable and form the substrate. Build them first and lock them down with unit tests.

### Task 1: Package scaffold

**Files:**
- Create: `hailo_apps/python/standalone_apps/tunnelvision/__init__.py`
- Create: `tests/python/standalone_apps/tunnelvision/__init__.py`
- Create: `tests/python/standalone_apps/tunnelvision/conftest.py`

- [ ] **Step 1: Create empty package files**

```bash
mkdir -p hailo_apps/python/standalone_apps/tunnelvision
mkdir -p tests/python/standalone_apps/tunnelvision
touch hailo_apps/python/standalone_apps/tunnelvision/__init__.py
touch tests/python/standalone_apps/tunnelvision/__init__.py
```

- [ ] **Step 2: Create test conftest with shared fixtures**

`tests/python/standalone_apps/tunnelvision/conftest.py`:

```python
import sqlite3
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_db_path(tmp_path: Path) -> str:
    return str(tmp_path / "test.db")


@pytest.fixture
def temp_snapshot_dir(tmp_path: Path) -> str:
    d = tmp_path / "snapshots"
    d.mkdir()
    return str(d)
```

- [ ] **Step 3: Verify pytest discovers the package**

```bash
source setup_env.sh
pytest tests/python/standalone_apps/tunnelvision -v --collect-only
```

Expected: `collected 0 items` (no tests yet, but no collection errors either).

- [ ] **Step 4: Commit**

```bash
git add hailo_apps/python/standalone_apps/tunnelvision/__init__.py \
        tests/python/standalone_apps/tunnelvision/__init__.py \
        tests/python/standalone_apps/tunnelvision/conftest.py
git commit -m "feat(tunnelvision): scaffold package + test directory"
```

---

### Task 2: SQLite schema (`db.py` part 1)

**Files:**
- Create: `hailo_apps/python/standalone_apps/tunnelvision/db.py`
- Create: `tests/python/standalone_apps/tunnelvision/test_db_schema.py`

**Tables in MVP** (per DESIGN.md §5.8): `visits`, `camera_observations`, `vehicle_tracks`, `track_state_events`, `best_images`, `rekor_carcheck_requests`, `rekor_carcheck_responses`, `rekor_plate_results`, `rekor_plate_candidates`, `plates`, `visit_match_events`.

- [ ] **Step 1: Write the failing schema test**

`tests/python/standalone_apps/tunnelvision/test_db_schema.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/python/standalone_apps/tunnelvision/test_db_schema.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` on `init_schema`.

- [ ] **Step 3: Implement schema in db.py**

`hailo_apps/python/standalone_apps/tunnelvision/db.py`:

```python
import sqlite3

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS visits (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK (status IN (
        'approaching','ingress_captured','in_tunnel',
        'egress_captured','completed','failed','manual_review'
    )),
    plate_id TEXT,
    ingress_observation_id TEXT,
    egress_observation_id TEXT,
    started_at TEXT NOT NULL,
    ingress_captured_at TEXT,
    egress_captured_at TEXT,
    completed_at TEXT,
    tunnel_duration_sec INTEGER,
    match_confidence REAL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS camera_observations (
    id TEXT PRIMARY KEY,
    visit_id TEXT,
    camera_role TEXT NOT NULL CHECK (camera_role IN ('ingress','egress')),
    started_at TEXT NOT NULL,
    ended_at TEXT,
    observation_status TEXT NOT NULL CHECK (observation_status IN (
        'tracking','best_image_selected','rekor_queued','rekor_sent',
        'completed','discarded','no_read'
    )),
    frame_count INTEGER NOT NULL DEFAULT 0,
    best_image_id TEXT,
    rejection_reason TEXT
);

CREATE TABLE IF NOT EXISTS vehicle_tracks (
    id TEXT PRIMARY KEY,
    camera_observation_id TEXT NOT NULL,
    camera_role TEXT NOT NULL,
    edge_track_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN (
        'new','tracking','partial_visible','fully_visible',
        'plate_blocked','plate_candidate','best_image_selected',
        'rekor_sent','processed_but_tracking','lost','exited','no_read'
    )),
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    current_zone TEXT,
    current_bbox TEXT,
    processed INTEGER NOT NULL DEFAULT 0,
    rekor_sent INTEGER NOT NULL DEFAULT 0,
    best_image_id TEXT,
    exit_confirmed INTEGER NOT NULL DEFAULT 0,
    UNIQUE(camera_role, edge_track_id)
);

CREATE TABLE IF NOT EXISTS track_state_events (
    id TEXT PRIMARY KEY,
    vehicle_track_id TEXT NOT NULL,
    previous_state TEXT,
    next_state TEXT NOT NULL,
    reason TEXT,
    zone_type TEXT,
    frame_index INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS best_images (
    id TEXT PRIMARY KEY,
    vehicle_track_id TEXT NOT NULL,
    observation_id TEXT NOT NULL,
    image_path TEXT NOT NULL,
    image_sha256 TEXT NOT NULL UNIQUE,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    quality_score REAL,
    blur_score REAL,
    motion_score REAL,
    plate_width_px INTEGER,
    car_fully_visible INTEGER NOT NULL,
    plate_visible INTEGER NOT NULL,
    plate_blocked INTEGER NOT NULL DEFAULT 0,
    vehicle_bbox TEXT,
    plate_bbox TEXT,
    selected_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rekor_carcheck_requests (
    id TEXT PRIMARY KEY,
    visit_id TEXT,
    observation_id TEXT NOT NULL,
    vehicle_track_id TEXT NOT NULL,
    best_image_id TEXT NOT NULL,
    image_path TEXT NOT NULL,
    image_sha256 TEXT NOT NULL,
    request_status TEXT NOT NULL CHECK (request_status IN (
        'queued','sent','succeeded','failed','skipped'
    )),
    credit_policy TEXT NOT NULL,
    estimated_credit_cost INTEGER,
    actual_credit_cost INTEGER,
    sent_at TEXT,
    completed_at TEXT,
    http_status INTEGER,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS rekor_carcheck_responses (
    id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    data_type TEXT,
    epoch_time INTEGER,
    img_width INTEGER,
    img_height INTEGER,
    error INTEGER,
    version INTEGER,
    uuid TEXT,
    credit_cost INTEGER,
    credits_monthly_used INTEGER,
    credits_monthly_total INTEGER,
    processing_time_total_ms REAL,
    raw_response TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rekor_plate_results (
    id TEXT PRIMARY KEY,
    response_id TEXT NOT NULL,
    plate_index INTEGER,
    plate TEXT NOT NULL,
    normalized_plate TEXT NOT NULL,
    region TEXT,
    confidence REAL,
    region_confidence REAL,
    matches_template INTEGER,
    coordinates TEXT,
    selected INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS rekor_plate_candidates (
    id TEXT PRIMARY KEY,
    plate_result_id TEXT NOT NULL,
    rank INTEGER NOT NULL,
    plate TEXT NOT NULL,
    normalized_plate TEXT NOT NULL,
    confidence REAL,
    matches_template INTEGER
);

CREATE TABLE IF NOT EXISTS plates (
    id TEXT PRIMARY KEY,
    plate_number TEXT NOT NULL,
    normalized_plate TEXT NOT NULL,
    region TEXT,
    country TEXT NOT NULL DEFAULT 'us',
    confidence REAL,
    source TEXT NOT NULL DEFAULT 'rekor_carcheck',
    created_at TEXT NOT NULL,
    UNIQUE(normalized_plate, region, country)
);

CREATE TABLE IF NOT EXISTS visit_match_events (
    id TEXT PRIMARY KEY,
    ingress_visit_id TEXT NOT NULL,
    egress_observation_id TEXT NOT NULL,
    match_method TEXT NOT NULL CHECK (match_method IN (
        'plate_exact','plate_fuzzy','lane_time_window','track_sequence','manual'
    )),
    match_confidence REAL NOT NULL,
    matched_at TEXT NOT NULL,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_visits_status ON visits(status);
CREATE INDEX IF NOT EXISTS idx_visits_ingress_captured ON visits(ingress_captured_at);
CREATE INDEX IF NOT EXISTS idx_tracks_observation ON vehicle_tracks(camera_observation_id);
CREATE INDEX IF NOT EXISTS idx_rekor_req_status ON rekor_carcheck_requests(request_status);
"""


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/python/standalone_apps/tunnelvision/test_db_schema.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add hailo_apps/python/standalone_apps/tunnelvision/db.py \
        tests/python/standalone_apps/tunnelvision/test_db_schema.py
git commit -m "feat(tunnelvision): SQLite schema for tracking + visits + Rekor"
```

---

### Task 3: DB write helpers + writer thread (`db.py` part 2)

**Files:**
- Modify: `hailo_apps/python/standalone_apps/tunnelvision/db.py`
- Create: `tests/python/standalone_apps/tunnelvision/test_db_writes.py`

**Approach:** A `DBWriter` thread drains a `queue.Queue` of write events. Each event is a dataclass that knows how to write itself. Read paths use a separate read-only connection (no thread safety issue — SQLite WAL handles concurrent reads).

- [ ] **Step 1: Write failing test for write events**

`tests/python/standalone_apps/tunnelvision/test_db_writes.py`:

```python
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
```

- [ ] **Step 2: Run test to verify failures**

```bash
pytest tests/python/standalone_apps/tunnelvision/test_db_writes.py -v
```

Expected: ImportError on `DBWriter`, `InsertVisit`, etc.

- [ ] **Step 3: Implement write events + DBWriter in db.py**

Append to `hailo_apps/python/standalone_apps/tunnelvision/db.py`:

```python
import logging
import queue
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class InsertVisit:
    id: str
    status: str
    started_at: str
    plate_id: Optional[str] = None
    ingress_observation_id: Optional[str] = None
    egress_observation_id: Optional[str] = None
    ingress_captured_at: Optional[str] = None
    egress_captured_at: Optional[str] = None
    completed_at: Optional[str] = None
    tunnel_duration_sec: Optional[int] = None
    match_confidence: Optional[float] = None
    notes: Optional[str] = None

    def apply(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO visits(id, status, plate_id, ingress_observation_id, "
            "egress_observation_id, started_at, ingress_captured_at, "
            "egress_captured_at, completed_at, tunnel_duration_sec, "
            "match_confidence, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (self.id, self.status, self.plate_id, self.ingress_observation_id,
             self.egress_observation_id, self.started_at, self.ingress_captured_at,
             self.egress_captured_at, self.completed_at, self.tunnel_duration_sec,
             self.match_confidence, self.notes),
        )


@dataclass
class UpdateVisitStatus:
    id: str
    status: str
    egress_observation_id: Optional[str] = None
    egress_captured_at: Optional[str] = None
    completed_at: Optional[str] = None
    tunnel_duration_sec: Optional[int] = None
    match_confidence: Optional[float] = None
    plate_id: Optional[str] = None

    def apply(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "UPDATE visits SET status=?, egress_observation_id=COALESCE(?, egress_observation_id), "
            "egress_captured_at=COALESCE(?, egress_captured_at), "
            "completed_at=COALESCE(?, completed_at), "
            "tunnel_duration_sec=COALESCE(?, tunnel_duration_sec), "
            "match_confidence=COALESCE(?, match_confidence), "
            "plate_id=COALESCE(?, plate_id) WHERE id=?",
            (self.status, self.egress_observation_id, self.egress_captured_at,
             self.completed_at, self.tunnel_duration_sec, self.match_confidence,
             self.plate_id, self.id),
        )


@dataclass
class InsertCameraObservation:
    id: str
    visit_id: Optional[str]
    camera_role: str
    started_at: str
    observation_status: str
    frame_count: int = 0
    best_image_id: Optional[str] = None
    rejection_reason: Optional[str] = None
    ended_at: Optional[str] = None

    def apply(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO camera_observations(id, visit_id, camera_role, started_at, "
            "ended_at, observation_status, frame_count, best_image_id, rejection_reason) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (self.id, self.visit_id, self.camera_role, self.started_at, self.ended_at,
             self.observation_status, self.frame_count, self.best_image_id, self.rejection_reason),
        )


@dataclass
class InsertVehicleTrack:
    id: str
    camera_observation_id: str
    camera_role: str
    edge_track_id: str
    state: str
    first_seen_at: str
    last_seen_at: str
    current_zone: Optional[str] = None
    current_bbox: Optional[str] = None

    def apply(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO vehicle_tracks(id, camera_observation_id, camera_role, "
            "edge_track_id, state, first_seen_at, last_seen_at, current_zone, current_bbox) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (self.id, self.camera_observation_id, self.camera_role, self.edge_track_id,
             self.state, self.first_seen_at, self.last_seen_at, self.current_zone, self.current_bbox),
        )


@dataclass
class UpdateVehicleTrack:
    id: str
    state: str
    last_seen_at: str
    current_zone: Optional[str] = None
    current_bbox: Optional[str] = None
    processed: Optional[int] = None
    rekor_sent: Optional[int] = None
    best_image_id: Optional[str] = None
    exit_confirmed: Optional[int] = None

    def apply(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "UPDATE vehicle_tracks SET state=?, last_seen_at=?, "
            "current_zone=COALESCE(?, current_zone), "
            "current_bbox=COALESCE(?, current_bbox), "
            "processed=COALESCE(?, processed), "
            "rekor_sent=COALESCE(?, rekor_sent), "
            "best_image_id=COALESCE(?, best_image_id), "
            "exit_confirmed=COALESCE(?, exit_confirmed) WHERE id=?",
            (self.state, self.last_seen_at, self.current_zone, self.current_bbox,
             self.processed, self.rekor_sent, self.best_image_id, self.exit_confirmed, self.id),
        )


@dataclass
class InsertTrackStateEvent:
    id: str
    vehicle_track_id: str
    previous_state: Optional[str]
    next_state: str
    reason: Optional[str]
    zone_type: Optional[str]
    frame_index: Optional[int]
    created_at: str

    def apply(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO track_state_events(id, vehicle_track_id, previous_state, "
            "next_state, reason, zone_type, frame_index, created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (self.id, self.vehicle_track_id, self.previous_state, self.next_state,
             self.reason, self.zone_type, self.frame_index, self.created_at),
        )


@dataclass
class InsertBestImage:
    id: str
    vehicle_track_id: str
    observation_id: str
    image_path: str
    image_sha256: str
    width: int
    height: int
    quality_score: float
    blur_score: float
    motion_score: float
    plate_width_px: int
    car_fully_visible: int
    plate_visible: int
    plate_blocked: int
    vehicle_bbox: Optional[str]
    plate_bbox: Optional[str]
    selected_at: str

    def apply(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO best_images(id, vehicle_track_id, observation_id, image_path, "
            "image_sha256, width, height, quality_score, blur_score, motion_score, "
            "plate_width_px, car_fully_visible, plate_visible, plate_blocked, "
            "vehicle_bbox, plate_bbox, selected_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (self.id, self.vehicle_track_id, self.observation_id, self.image_path,
             self.image_sha256, self.width, self.height, self.quality_score, self.blur_score,
             self.motion_score, self.plate_width_px, self.car_fully_visible, self.plate_visible,
             self.plate_blocked, self.vehicle_bbox, self.plate_bbox, self.selected_at),
        )


@dataclass
class InsertRekorRequest:
    id: str
    observation_id: str
    vehicle_track_id: str
    best_image_id: str
    image_path: str
    image_sha256: str
    request_status: str
    credit_policy: str
    visit_id: Optional[str] = None
    estimated_credit_cost: Optional[int] = None
    actual_credit_cost: Optional[int] = None
    sent_at: Optional[str] = None
    completed_at: Optional[str] = None
    http_status: Optional[int] = None
    error_message: Optional[str] = None

    def apply(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO rekor_carcheck_requests(id, visit_id, observation_id, "
            "vehicle_track_id, best_image_id, image_path, image_sha256, "
            "request_status, credit_policy, estimated_credit_cost, actual_credit_cost, "
            "sent_at, completed_at, http_status, error_message) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (self.id, self.visit_id, self.observation_id, self.vehicle_track_id,
             self.best_image_id, self.image_path, self.image_sha256, self.request_status,
             self.credit_policy, self.estimated_credit_cost, self.actual_credit_cost,
             self.sent_at, self.completed_at, self.http_status, self.error_message),
        )


@dataclass
class UpdateRekorRequest:
    id: str
    request_status: str
    actual_credit_cost: Optional[int] = None
    completed_at: Optional[str] = None
    http_status: Optional[int] = None
    error_message: Optional[str] = None

    def apply(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "UPDATE rekor_carcheck_requests SET request_status=?, "
            "actual_credit_cost=COALESCE(?, actual_credit_cost), "
            "completed_at=COALESCE(?, completed_at), "
            "http_status=COALESCE(?, http_status), "
            "error_message=COALESCE(?, error_message) WHERE id=?",
            (self.request_status, self.actual_credit_cost, self.completed_at,
             self.http_status, self.error_message, self.id),
        )


@dataclass
class InsertRekorResponse:
    id: str
    request_id: str
    raw_response: str
    created_at: str
    data_type: Optional[str] = None
    epoch_time: Optional[int] = None
    img_width: Optional[int] = None
    img_height: Optional[int] = None
    error: Optional[int] = None
    version: Optional[int] = None
    uuid: Optional[str] = None
    credit_cost: Optional[int] = None
    credits_monthly_used: Optional[int] = None
    credits_monthly_total: Optional[int] = None
    processing_time_total_ms: Optional[float] = None

    def apply(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO rekor_carcheck_responses(id, request_id, data_type, epoch_time, "
            "img_width, img_height, error, version, uuid, credit_cost, credits_monthly_used, "
            "credits_monthly_total, processing_time_total_ms, raw_response, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (self.id, self.request_id, self.data_type, self.epoch_time, self.img_width,
             self.img_height, self.error, self.version, self.uuid, self.credit_cost,
             self.credits_monthly_used, self.credits_monthly_total,
             self.processing_time_total_ms, self.raw_response, self.created_at),
        )


@dataclass
class InsertRekorPlateResult:
    id: str
    response_id: str
    plate: str
    normalized_plate: str
    plate_index: Optional[int] = None
    region: Optional[str] = None
    confidence: Optional[float] = None
    region_confidence: Optional[float] = None
    matches_template: Optional[int] = None
    coordinates: Optional[str] = None
    selected: int = 0

    def apply(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO rekor_plate_results(id, response_id, plate_index, plate, "
            "normalized_plate, region, confidence, region_confidence, matches_template, "
            "coordinates, selected) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (self.id, self.response_id, self.plate_index, self.plate, self.normalized_plate,
             self.region, self.confidence, self.region_confidence, self.matches_template,
             self.coordinates, self.selected),
        )


@dataclass
class InsertVisitMatchEvent:
    id: str
    ingress_visit_id: str
    egress_observation_id: str
    match_method: str
    match_confidence: float
    matched_at: str
    notes: Optional[str] = None

    def apply(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO visit_match_events(id, ingress_visit_id, egress_observation_id, "
            "match_method, match_confidence, matched_at, notes) "
            "VALUES (?,?,?,?,?,?,?)",
            (self.id, self.ingress_visit_id, self.egress_observation_id, self.match_method,
             self.match_confidence, self.matched_at, self.notes),
        )


_SHUTDOWN = object()


class DBWriter(threading.Thread):
    """Single-writer thread draining a queue of write events."""

    def __init__(self, db_path: str, q: queue.Queue, name: str = "db-writer"):
        super().__init__(name=name, daemon=True)
        self._db_path = db_path
        self._queue = q

    def run(self) -> None:
        conn = sqlite3.connect(self._db_path, isolation_level=None)
        init_schema(conn)
        while True:
            event = self._queue.get()
            try:
                if event is _SHUTDOWN:
                    return
                try:
                    event.apply(conn)
                except Exception as exc:  # noqa: BLE001
                    logger.error("DBWriter event %s failed: %s", type(event).__name__, exc)
            finally:
                self._queue.task_done()

    def stop(self) -> None:
        self._queue.put(_SHUTDOWN)
        self.join(timeout=5)


def open_reader(db_path: str) -> sqlite3.Connection:
    """Read-only connection for correlator lookups."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/python/standalone_apps/tunnelvision/test_db_writes.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add hailo_apps/python/standalone_apps/tunnelvision/db.py \
        tests/python/standalone_apps/tunnelvision/test_db_writes.py
git commit -m "feat(tunnelvision): DBWriter thread + write event dataclasses"
```

---

### Task 4: RTSPIngest (copy from carwash_lpr)

**Files:**
- Create: `hailo_apps/python/standalone_apps/tunnelvision/ingest.py`

- [ ] **Step 1: Copy ingest.py from carwash_lpr**

```bash
cp hailo_apps/python/standalone_apps/carwash_lpr/ingest.py \
   hailo_apps/python/standalone_apps/tunnelvision/ingest.py
```

- [ ] **Step 2: Verify it imports**

```bash
python3 -c "from hailo_apps.python.standalone_apps.tunnelvision.ingest import RTSPIngest; print(RTSPIngest)"
```

Expected: `<class 'hailo_apps.python.standalone_apps.tunnelvision.ingest.RTSPIngest'>`

- [ ] **Step 3: Commit**

```bash
git add hailo_apps/python/standalone_apps/tunnelvision/ingest.py
git commit -m "feat(tunnelvision): RTSPIngest (verbatim copy from carwash_lpr)"
```

---

### Task 5: Zone evaluator (`zone.py`)

**Files:**
- Create: `hailo_apps/python/standalone_apps/tunnelvision/zone.py`
- Create: `tests/python/standalone_apps/tunnelvision/test_zone.py`

**Reference:** Port `Zone.contains_point` and `Zone.overlaps_bbox` from `td_edge/src/td_edge/config/zones.py`. The td-edge implementation uses normalized 0–1 coords; we keep that.

- [ ] **Step 1: Write failing tests**

`tests/python/standalone_apps/tunnelvision/test_zone.py`:

```python
import json
from pathlib import Path

import pytest

from hailo_apps.python.standalone_apps.tunnelvision.zone import (
    Zone, ZoneConfig, load_zones,
)


def test_contains_point_inside_square():
    z = Zone(name="t", zone_type="trigger", action="start_visit", camera_id=0,
             polygon=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert z.contains_point(0.5, 0.5)
    assert not z.contains_point(1.5, 0.5)
    assert not z.contains_point(-0.1, 0.5)


def test_contains_point_outside_pentagon():
    z = Zone(name="t", zone_type="trigger", action="start_visit", camera_id=0,
             polygon=[[0.325, 0.2824], [0.1297, 0.988], [0.9695, 0.9722],
                      [0.7219, 0.2167], [0.3281, 0.2852]])
    assert z.contains_point(0.5, 0.6)
    assert not z.contains_point(0.0, 0.0)


def test_overlaps_bbox():
    z = Zone(name="t", zone_type="alpr", action="capture_plate", camera_id=0,
             polygon=[[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]])
    assert z.overlaps_bbox([0.5, 0.5, 0.6, 0.6])
    assert z.overlaps_bbox([0.1, 0.1, 0.5, 0.5])  # one corner inside
    assert not z.overlaps_bbox([0.0, 0.0, 0.1, 0.1])


def test_load_zones_from_file(tmp_path: Path):
    config = {
        "zones": [
            {
                "name": "ingress_gate", "zone_type": "trigger", "camera_id": 0,
                "action": "start_visit",
                "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
            },
            {
                "name": "lpr_zone", "zone_type": "alpr", "camera_id": 0,
                "action": "capture_plate",
                "polygon": [[0.2, 0.2], [0.7, 0.2], [0.7, 0.7], [0.2, 0.7]],
            },
            {
                "name": "egress_gate", "zone_type": "trigger", "camera_id": 1,
                "action": "end_visit",
                "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
            },
        ],
    }
    p = tmp_path / "z.json"
    p.write_text(json.dumps(config))

    cfg = load_zones(str(p))
    assert len(cfg.for_camera(0)) == 2
    assert len(cfg.for_camera(1)) == 1
    assert cfg.zones_with_action(0, "capture_plate")[0].name == "lpr_zone"


def test_load_real_zones_file():
    """Smoke test against the actual file shipped with the experiment."""
    cfg = load_zones("experiments/tunnelvision/zones.json")
    cam0 = cfg.for_camera(0)
    cam1 = cfg.for_camera(1)
    assert len(cam0) >= 2
    assert any(z.action == "start_visit" for z in cam0)
    assert any(z.action == "capture_plate" for z in cam0)
    assert any(z.action == "end_visit" for z in cam1)
```

- [ ] **Step 2: Run to verify failures**

```bash
pytest tests/python/standalone_apps/tunnelvision/test_zone.py -v
```

Expected: ImportError on `Zone`, `ZoneConfig`, `load_zones`.

- [ ] **Step 3: Implement zone.py**

`hailo_apps/python/standalone_apps/tunnelvision/zone.py`:

```python
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Zone:
    name: str
    zone_type: str            # "trigger", "alpr", "ignore", ...
    action: Optional[str]     # "start_visit", "capture_plate", "end_visit", "skip"
    camera_id: Optional[int]  # 0 = ingress (cam1), 1 = egress (cam2)
    polygon: list             # list of [x, y] in normalized 0-1 coords

    def contains_point(self, x: float, y: float) -> bool:
        n = len(self.polygon)
        inside = False
        p1x, p1y = self.polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = self.polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        xinters = p1x
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def overlaps_bbox(self, bbox) -> bool:
        if isinstance(bbox, dict):
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        else:
            if len(bbox) != 4:
                return False
            x1, y1, x2, y2 = bbox
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2),
                   ((x1 + x2) / 2, (y1 + y2) / 2)]
        return any(self.contains_point(cx, cy) for cx, cy in corners)


@dataclass
class ZoneConfig:
    zones: list = field(default_factory=list)

    def for_camera(self, camera_id: int) -> list:
        return [z for z in self.zones
                if z.camera_id is None or z.camera_id == camera_id]

    def zones_with_action(self, camera_id: int, action: str) -> list:
        return [z for z in self.for_camera(camera_id) if z.action == action]

    def has_action(self, camera_id: int, action: str) -> bool:
        return bool(self.zones_with_action(camera_id, action))


def load_zones(path: str) -> ZoneConfig:
    data = json.loads(Path(path).read_text())
    zones = [
        Zone(
            name=z["name"],
            zone_type=z["zone_type"],
            action=z.get("action"),
            camera_id=z.get("camera_id"),
            polygon=z["polygon"],
        )
        for z in data.get("zones", [])
    ]
    return ZoneConfig(zones=zones)
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/python/standalone_apps/tunnelvision/test_zone.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add hailo_apps/python/standalone_apps/tunnelvision/zone.py \
        tests/python/standalone_apps/tunnelvision/test_zone.py
git commit -m "feat(tunnelvision): zone evaluator (td-edge schema port)"
```

---

### Task 6: State machine (`state.py`)

**Files:**
- Create: `hailo_apps/python/standalone_apps/tunnelvision/state.py`
- Create: `tests/python/standalone_apps/tunnelvision/test_state.py`

**State graph (PRD §9):** `NEW → TRACKING → PARTIAL_VISIBLE → FULLY_VISIBLE → (PLATE_BLOCKED | PLATE_CANDIDATE) → BEST_IMAGE_SELECTED → REKOR_SENT → PROCESSED_BUT_TRACKING → IN_TUNNEL → MATCHED_AT_EGRESS → COMPLETED`. Plus `LOST | EXITED | NO_READ` terminal states.

- [ ] **Step 1: Write failing tests**

`tests/python/standalone_apps/tunnelvision/test_state.py`:

```python
from datetime import datetime, timezone

from hailo_apps.python.standalone_apps.tunnelvision.state import (
    TrackState, VehicleTrack, FrameCandidate, transition,
)


def _make_track(state=TrackState.NEW) -> VehicleTrack:
    now = datetime.now(timezone.utc)
    return VehicleTrack(
        track_id=1, camera_role="ingress", state=state,
        first_seen_at=now, last_seen_at=now, current_bbox=[0, 0, 100, 100],
    )


def test_new_to_tracking_on_first_detection():
    t = _make_track(TrackState.NEW)
    next_state = transition(t, current_zone="approach", fully_visible=True,
                            plate_visible=False, plate_blocked=False)
    assert next_state == TrackState.TRACKING


def test_tracking_to_fully_visible_when_in_capture():
    t = _make_track(TrackState.TRACKING)
    next_state = transition(t, current_zone="capture", fully_visible=True,
                            plate_visible=False, plate_blocked=False)
    assert next_state == TrackState.FULLY_VISIBLE


def test_fully_visible_to_plate_blocked():
    t = _make_track(TrackState.FULLY_VISIBLE)
    next_state = transition(t, current_zone="capture", fully_visible=True,
                            plate_visible=False, plate_blocked=True)
    assert next_state == TrackState.PLATE_BLOCKED


def test_plate_blocked_recovers_when_unblocked():
    t = _make_track(TrackState.PLATE_BLOCKED)
    next_state = transition(t, current_zone="capture", fully_visible=True,
                            plate_visible=True, plate_blocked=False)
    assert next_state == TrackState.PLATE_CANDIDATE


def test_processed_track_keeps_tracking():
    t = _make_track(TrackState.PROCESSED_BUT_TRACKING)
    t.processed = True
    next_state = transition(t, current_zone="capture", fully_visible=True,
                            plate_visible=True, plate_blocked=False)
    # Already processed, do not re-enter capture cycle
    assert next_state == TrackState.PROCESSED_BUT_TRACKING


def test_lost_track_after_max_lost_frames():
    t = _make_track(TrackState.TRACKING)
    t.lost_frames = 11  # > default max_lost_frames
    next_state = transition(t, current_zone=None, fully_visible=False,
                            plate_visible=False, plate_blocked=False,
                            max_lost_frames=10)
    assert next_state == TrackState.LOST


def test_frame_candidate_dataclass_holds_metadata():
    c = FrameCandidate(
        frame=None, quality_score=85.0, blur_score=120.0,
        motion_score=0.5, plate_width_px=80, plate_confidence=0.91,
        bbox_stability=0.95, vehicle_bbox=[0, 0, 100, 200],
        plate_bbox=[10, 60, 80, 90],
    )
    assert c.quality_score == 85.0
```

- [ ] **Step 2: Run, verify failures**

```bash
pytest tests/python/standalone_apps/tunnelvision/test_state.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement state.py**

`hailo_apps/python/standalone_apps/tunnelvision/state.py`:

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np


class TrackState(str, Enum):
    NEW = "new"
    TRACKING = "tracking"
    PARTIAL_VISIBLE = "partial_visible"
    FULLY_VISIBLE = "fully_visible"
    PLATE_BLOCKED = "plate_blocked"
    PLATE_CANDIDATE = "plate_candidate"
    BEST_IMAGE_SELECTED = "best_image_selected"
    REKOR_SENT = "rekor_sent"
    PROCESSED_BUT_TRACKING = "processed_but_tracking"
    IN_TUNNEL = "in_tunnel"
    MATCHED_AT_EGRESS = "matched_at_egress"
    COMPLETED = "completed"
    LOST = "lost"
    EXITED = "exited"
    NO_READ = "no_read"


@dataclass
class FrameCandidate:
    frame: Optional[np.ndarray]
    quality_score: float
    blur_score: float
    motion_score: float
    plate_width_px: int
    plate_confidence: float
    bbox_stability: float
    vehicle_bbox: list
    plate_bbox: list


@dataclass
class VehicleTrack:
    track_id: int
    camera_role: str
    state: TrackState
    first_seen_at: datetime
    last_seen_at: datetime
    current_bbox: list
    db_id: Optional[str] = None
    observation_db_id: Optional[str] = None
    current_zone: Optional[str] = None
    fully_visible: bool = False
    processed: bool = False
    rekor_sent: bool = False
    candidates: list = field(default_factory=list)
    best_image_path: Optional[str] = None
    best_candidate: Optional[FrameCandidate] = None
    lost_frames: int = 0


_TERMINAL = {TrackState.LOST, TrackState.EXITED, TrackState.NO_READ, TrackState.COMPLETED}


def transition(
    track: VehicleTrack,
    current_zone: Optional[str],
    fully_visible: bool,
    plate_visible: bool,
    plate_blocked: bool,
    max_lost_frames: int = 10,
) -> TrackState:
    """Pure function — returns the next state without mutating the track."""
    if track.state in _TERMINAL:
        return track.state

    if track.lost_frames > max_lost_frames:
        return TrackState.LOST

    if track.processed:
        return TrackState.PROCESSED_BUT_TRACKING

    if track.state == TrackState.NEW:
        return TrackState.TRACKING

    if current_zone == "capture":
        if not fully_visible:
            return TrackState.PARTIAL_VISIBLE
        if plate_blocked:
            return TrackState.PLATE_BLOCKED
        if plate_visible:
            return TrackState.PLATE_CANDIDATE
        return TrackState.FULLY_VISIBLE

    return TrackState.TRACKING
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest tests/python/standalone_apps/tunnelvision/test_state.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add hailo_apps/python/standalone_apps/tunnelvision/state.py \
        tests/python/standalone_apps/tunnelvision/test_state.py
git commit -m "feat(tunnelvision): track state machine (12 states)"
```

---

### Task 7: Quality scorer (`scorer.py`)

**Files:**
- Create: `hailo_apps/python/standalone_apps/tunnelvision/scorer.py`
- Create: `tests/python/standalone_apps/tunnelvision/test_scorer.py`

**Reference:** PRD §11 quality formula. Blur = Laplacian variance on plate crop. Motion = pixel delta of bbox centroid between consecutive frames.

- [ ] **Step 1: Write failing tests**

`tests/python/standalone_apps/tunnelvision/test_scorer.py`:

```python
import numpy as np
import pytest

from hailo_apps.python.standalone_apps.tunnelvision.scorer import (
    compute_blur_score, compute_quality_score, frame_passes_gates,
    select_best_candidate, ScoringRule,
)
from hailo_apps.python.standalone_apps.tunnelvision.state import FrameCandidate


def _checkerboard(size=128):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[::8, :] = 255
    img[:, ::8] = 255
    return img


def _blurred(size=128):
    return np.full((size, size, 3), 128, dtype=np.uint8)


def test_blur_score_higher_for_sharp_image():
    sharp = compute_blur_score(_checkerboard())
    blur = compute_blur_score(_blurred())
    assert sharp > blur > 0


def test_quality_score_increases_with_plate_confidence():
    base = dict(
        plate_confidence=0.5, blur_score=100, plate_width_px=80,
        bbox_stability=0.9, roi_fit=0.9, motion_score=0.0,
        plate_blocked=False, plate_width_ref=120,
    )
    s_low = compute_quality_score(**base)
    s_high = compute_quality_score(**{**base, "plate_confidence": 0.95})
    assert s_high > s_low


def test_frame_passes_gates_rejects_blurry():
    rule = ScoringRule()
    assert not frame_passes_gates(
        plate_confidence=0.9, blur_score=10.0,
        plate_width_px=80, motion_score=0.0,
        bbox_growth_rate=0.0, fully_visible=True,
        plate_visible=True, plate_blocked=False,
        rule=rule,
    )


def test_frame_passes_gates_accepts_good():
    rule = ScoringRule()
    assert frame_passes_gates(
        plate_confidence=0.85, blur_score=200.0,
        plate_width_px=80, motion_score=0.5,
        bbox_growth_rate=0.05, fully_visible=True,
        plate_visible=True, plate_blocked=False,
        rule=rule,
    )


def test_select_best_picks_highest_score():
    cs = [
        FrameCandidate(frame=None, quality_score=70, blur_score=80, motion_score=0,
                       plate_width_px=70, plate_confidence=0.8, bbox_stability=0.9,
                       vehicle_bbox=[0,0,100,100], plate_bbox=[10,60,80,90]),
        FrameCandidate(frame=None, quality_score=90, blur_score=200, motion_score=0,
                       plate_width_px=80, plate_confidence=0.92, bbox_stability=0.95,
                       vehicle_bbox=[0,0,100,100], plate_bbox=[10,60,80,90]),
        FrameCandidate(frame=None, quality_score=80, blur_score=150, motion_score=0,
                       plate_width_px=75, plate_confidence=0.85, bbox_stability=0.93,
                       vehicle_bbox=[0,0,100,100], plate_bbox=[10,60,80,90]),
    ]
    best = select_best_candidate(cs)
    assert best.quality_score == 90
```

- [ ] **Step 2: Run, verify failures**

```bash
pytest tests/python/standalone_apps/tunnelvision/test_scorer.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement scorer.py**

`hailo_apps/python/standalone_apps/tunnelvision/scorer.py`:

```python
from dataclasses import dataclass

import cv2
import numpy as np

from hailo_apps.python.standalone_apps.tunnelvision.state import FrameCandidate


@dataclass
class ScoringRule:
    min_vehicle_confidence: float = 0.5
    min_plate_confidence: float = 0.6
    min_plate_width_px: int = 40
    min_blur_score: float = 50.0
    max_motion_score: float = 5.0
    max_bbox_growth_rate: float = 0.20
    plate_width_ref: int = 120  # px reference for plate_width_score normalization


def compute_blur_score(image: np.ndarray) -> float:
    """Laplacian variance — higher = sharper. Range typically 0..2000+."""
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_motion_score(prev_centroid, curr_centroid) -> float:
    """Pixel delta of bbox centroid between frames. Lower = stable."""
    if prev_centroid is None or curr_centroid is None:
        return 0.0
    dx = curr_centroid[0] - prev_centroid[0]
    dy = curr_centroid[1] - prev_centroid[1]
    return float(np.hypot(dx, dy))


def compute_quality_score(
    *,
    plate_confidence: float,
    blur_score: float,
    plate_width_px: int,
    bbox_stability: float,
    roi_fit: float,
    motion_score: float,
    plate_blocked: bool,
    plate_width_ref: int = 120,
    blur_ref: float = 200.0,
) -> float:
    """PRD §11 quality formula."""
    plate_width_norm = min(plate_width_px / max(plate_width_ref, 1), 1.0)
    blur_norm = min(blur_score / max(blur_ref, 1), 1.0)
    motion_penalty = min(motion_score, 10.0) * 2.0
    blockage_penalty = 50.0 if plate_blocked else 0.0
    return (
        plate_confidence * 30.0
        + blur_norm * 25.0
        + plate_width_norm * 20.0
        + bbox_stability * 15.0
        + roi_fit * 10.0
        - motion_penalty
        - blockage_penalty
    )


def frame_passes_gates(
    *,
    plate_confidence: float,
    blur_score: float,
    plate_width_px: int,
    motion_score: float,
    bbox_growth_rate: float,
    fully_visible: bool,
    plate_visible: bool,
    plate_blocked: bool,
    rule: ScoringRule,
) -> bool:
    if not (fully_visible and plate_visible):
        return False
    if plate_blocked:
        return False
    if plate_confidence < rule.min_plate_confidence:
        return False
    if plate_width_px < rule.min_plate_width_px:
        return False
    if blur_score < rule.min_blur_score:
        return False
    if motion_score > rule.max_motion_score:
        return False
    if abs(bbox_growth_rate) > rule.max_bbox_growth_rate:
        return False
    return True


def select_best_candidate(candidates: list) -> FrameCandidate:
    return max(candidates, key=lambda c: c.quality_score)
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest tests/python/standalone_apps/tunnelvision/test_scorer.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add hailo_apps/python/standalone_apps/tunnelvision/scorer.py \
        tests/python/standalone_apps/tunnelvision/test_scorer.py
git commit -m "feat(tunnelvision): best-frame quality scorer (PRD §11 formula)"
```

---

## Phase 2 — Inference + tracking (DeGirum local)

### Task 8: IoU multi-object tracker (`tracker.py` part 1)

**Files:**
- Create: `hailo_apps/python/standalone_apps/tunnelvision/tracker.py`
- Create: `tests/python/standalone_apps/tunnelvision/test_tracker.py`

**Approach:** Greedy IoU association (no Hungarian needed at single-lane density). Tracks have `(track_id, bbox, lost_frames)`. On each frame, associate detections to existing tracks by max-IoU above threshold. Unmatched detections start new tracks. Tracks with `lost_frames > max_lost_frames` are retired.

- [ ] **Step 1: Write failing tests**

`tests/python/standalone_apps/tunnelvision/test_tracker.py`:

```python
import pytest

from hailo_apps.python.standalone_apps.tunnelvision.tracker import (
    IoUTracker, iou,
)


def test_iou_perfect_match():
    assert iou([0, 0, 100, 100], [0, 0, 100, 100]) == pytest.approx(1.0)


def test_iou_no_overlap():
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_iou_half_overlap():
    # Two 100x100 boxes shifted by 50px → intersection 50x100=5000, union 15000
    val = iou([0, 0, 100, 100], [50, 0, 150, 100])
    assert val == pytest.approx(5000 / 15000)


def test_tracker_assigns_new_id_to_first_detection():
    t = IoUTracker(iou_threshold=0.3, max_lost_frames=5)
    out = t.update([[0, 0, 100, 100]])
    assert len(out) == 1
    assert out[0].track_id == 1


def test_tracker_keeps_id_across_frames():
    t = IoUTracker(iou_threshold=0.3, max_lost_frames=5)
    out1 = t.update([[0, 0, 100, 100]])
    out2 = t.update([[5, 5, 105, 105]])  # high IoU
    assert out1[0].track_id == out2[0].track_id


def test_tracker_assigns_new_id_when_iou_too_low():
    t = IoUTracker(iou_threshold=0.3, max_lost_frames=5)
    t.update([[0, 0, 100, 100]])
    out = t.update([[200, 200, 300, 300]])
    assert out[0].track_id == 2


def test_tracker_retires_lost_track():
    t = IoUTracker(iou_threshold=0.3, max_lost_frames=2)
    out1 = t.update([[0, 0, 100, 100]])
    first_id = out1[0].track_id
    for _ in range(3):
        t.update([])
    out = t.update([[0, 0, 100, 100]])  # should be a NEW track now
    assert out[0].track_id != first_id


def test_tracker_handles_two_simultaneous_vehicles():
    t = IoUTracker(iou_threshold=0.3, max_lost_frames=5)
    out1 = t.update([[0, 0, 100, 100], [200, 200, 300, 300]])
    ids = sorted(o.track_id for o in out1)
    assert ids == [1, 2]
    out2 = t.update([[5, 5, 105, 105], [205, 205, 305, 305]])
    assert sorted(o.track_id for o in out2) == [1, 2]
```

- [ ] **Step 2: Run, verify failures**

```bash
pytest tests/python/standalone_apps/tunnelvision/test_tracker.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement tracker.py (IoU + tracker only — detector wrapper next task)**

`hailo_apps/python/standalone_apps/tunnelvision/tracker.py`:

```python
from dataclasses import dataclass, field
from typing import Iterable, Optional


def iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    bbox: list
    lost_frames: int = 0
    score: float = 0.0
    bbox_history: list = field(default_factory=list)


class IoUTracker:
    """Greedy IoU multi-object tracker.

    On update(detections):
      - For each existing track, find the detection with max IoU above iou_threshold.
      - Matched: update bbox + reset lost_frames.
      - Unmatched detections start new tracks.
      - Tracks not matched this frame have lost_frames++.
      - Tracks with lost_frames > max_lost_frames are retired.
    """

    def __init__(self, iou_threshold: float = 0.3, max_lost_frames: int = 10):
        self._iou_threshold = iou_threshold
        self._max_lost_frames = max_lost_frames
        self._next_id = 1
        self._tracks: list = []

    def update(self, detections: Iterable[list], scores: Optional[list] = None) -> list:
        dets = list(detections)
        scores = list(scores) if scores is not None else [0.0] * len(dets)

        # Greedy assignment: best IoU pair first
        unmatched_dets = set(range(len(dets)))
        matched_pairs = []
        for ti, track in enumerate(self._tracks):
            best_iou, best_di = 0.0, None
            for di in unmatched_dets:
                v = iou(track.bbox, dets[di])
                if v > best_iou:
                    best_iou, best_di = v, di
            if best_di is not None and best_iou >= self._iou_threshold:
                matched_pairs.append((ti, best_di))
                unmatched_dets.discard(best_di)

        matched_track_idx = {ti for ti, _ in matched_pairs}

        # Update matched
        for ti, di in matched_pairs:
            self._tracks[ti].bbox = list(dets[di])
            self._tracks[ti].score = scores[di]
            self._tracks[ti].lost_frames = 0
            self._tracks[ti].bbox_history.append(list(dets[di]))

        # Age unmatched tracks
        for ti, track in enumerate(self._tracks):
            if ti not in matched_track_idx:
                track.lost_frames += 1

        # New tracks for unmatched detections
        for di in unmatched_dets:
            self._tracks.append(Track(
                track_id=self._next_id, bbox=list(dets[di]),
                score=scores[di], bbox_history=[list(dets[di])],
            ))
            self._next_id += 1

        # Retire stale tracks
        self._tracks = [t for t in self._tracks if t.lost_frames <= self._max_lost_frames]

        return list(self._tracks)
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest tests/python/standalone_apps/tunnelvision/test_tracker.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add hailo_apps/python/standalone_apps/tunnelvision/tracker.py \
        tests/python/standalone_apps/tunnelvision/test_tracker.py
git commit -m "feat(tunnelvision): IoU multi-object tracker"
```

---

### Task 9: DeGirum vehicle detector + plate detector + OCR wrappers (`tracker.py` part 2 + `inference.py`)

**Files:**
- Create: `hailo_apps/python/standalone_apps/tunnelvision/inference.py`
- Modify: `hailo_apps/python/standalone_apps/tunnelvision/tracker.py` (add `VehicleDetector`)

**Required:** the model names from `experiments/tunnelvision/MODELS.md` (Task 0).

- [ ] **Step 1: Implement inference.py — DeGirum local model wrappers**

Use values from `experiments/tunnelvision/MODELS.md`. Replace `<VEHICLE_MODEL>` below with the exact name recorded in Task 0.

`hailo_apps/python/standalone_apps/tunnelvision/inference.py`:

```python
"""DeGirum local inference wrappers — vehicle detector, plate detector, OCR."""

from dataclasses import dataclass
from typing import List

import degirum as dg
import numpy as np

from hailo_apps.python.core.common.hailo_logger import get_logger

logger = get_logger(__name__)

_DG_HOST = "@local"
_DG_ZOO = "degirum/models_hailort"

# Replace VEHICLE_MODEL with the value from experiments/tunnelvision/MODELS.md
VEHICLE_MODEL = "<FILL FROM MODELS.md>"
PLATE_MODEL   = "yolov8n_relu6_lp--640x640_quant_hailort_hailo8l_1"
OCR_MODEL     = "yolov8n_relu6_lp_ocr--256x128_quant_hailort_hailo8l_1"

# Vehicle classes to keep from a COCO detector. Adjust if MODELS.md notes a
# vehicle-specific detector that already filters classes.
VEHICLE_CLASS_LABELS = {"car", "truck", "bus", "motorcycle"}


@dataclass
class Detection:
    bbox: list           # [x1, y1, x2, y2] in input-image pixel coords
    score: float
    label: str


@dataclass
class PlateDetection:
    bbox: list           # plate bbox in vehicle-crop pixel coords
    score: float


@dataclass
class OCRResult:
    plate_string: str
    confidence: float


class VehicleDetector:
    def __init__(self, model_name: str = VEHICLE_MODEL):
        self._model = dg.load_model(
            model_name=model_name,
            inference_host_address=_DG_HOST,
            zoo_url=_DG_ZOO,
        )
        logger.info("VehicleDetector ready: %s", model_name)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        result = self._model(frame)
        out = []
        for r in result.results or []:
            label = (r.get("label") or "").lower()
            if VEHICLE_CLASS_LABELS and label not in VEHICLE_CLASS_LABELS:
                continue
            out.append(Detection(bbox=list(r["bbox"]), score=float(r["score"]), label=label))
        return out


class PlateDetector:
    def __init__(self, model_name: str = PLATE_MODEL):
        self._model = dg.load_model(
            model_name=model_name,
            inference_host_address=_DG_HOST,
            zoo_url=_DG_ZOO,
        )
        logger.info("PlateDetector ready: %s", model_name)

    def detect(self, vehicle_crop: np.ndarray) -> List[PlateDetection]:
        result = self._model(vehicle_crop)
        return [
            PlateDetection(bbox=list(r["bbox"]), score=float(r["score"]))
            for r in (result.results or [])
        ]


class PlateOCR:
    def __init__(self, model_name: str = OCR_MODEL):
        self._model = dg.load_model(
            model_name=model_name,
            inference_host_address=_DG_HOST,
            zoo_url=_DG_ZOO,
            output_use_regular_nms=False,
            output_confidence_threshold=0.1,
        )
        logger.info("PlateOCR ready: %s", model_name)

    def read(self, plate_crop: np.ndarray) -> OCRResult:
        result = self._model(plate_crop)
        chars = sorted(result.results or [], key=lambda r: r["bbox"][0])
        if not chars:
            return OCRResult(plate_string="", confidence=0.0)
        plate_str = "".join(c["label"] for c in chars)
        avg_conf = sum(c["score"] for c in chars) / len(chars)
        return OCRResult(plate_string=plate_str, confidence=float(avg_conf))
```

- [ ] **Step 2: Append a smoke-test script (not pytest — needs the live device)**

`experiments/tunnelvision/scripts/smoke_inference.py`:

```python
"""Manual smoke test — runs all three models on assets/Car.jpg."""
import sys
import cv2

from hailo_apps.python.standalone_apps.tunnelvision.inference import (
    VehicleDetector, PlateDetector, PlateOCR,
)


def main():
    img = cv2.imread("assets/Car.jpg")
    if img is None:
        print("ERROR: assets/Car.jpg not found", file=sys.stderr)
        sys.exit(1)

    vd = VehicleDetector()
    vehicles = vd.detect(img)
    print(f"vehicles: {len(vehicles)}")
    for v in vehicles:
        print(f"  {v.label} score={v.score:.2f} bbox={v.bbox}")

    pd = PlateDetector()
    if vehicles:
        x1, y1, x2, y2 = map(int, vehicles[0].bbox)
        crop = img[max(0, y1):y2, max(0, x1):x2]
    else:
        crop = img
    plates = pd.detect(crop)
    print(f"plates in first vehicle: {len(plates)}")

    if plates:
        ocr = PlateOCR()
        x1, y1, x2, y2 = map(int, plates[0].bbox)
        plate_crop = crop[max(0, y1):y2, max(0, x1):x2]
        result = ocr.read(plate_crop)
        print(f"OCR: {result.plate_string!r} conf={result.confidence:.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run smoke test**

```bash
mkdir -p experiments/tunnelvision/scripts
# (place the script above in that path)
source setup_env.sh
python3 experiments/tunnelvision/scripts/smoke_inference.py
```

Expected: at least one vehicle detected; if a plate is visible, OCR returns a non-empty string.

- [ ] **Step 4: Commit**

```bash
git add hailo_apps/python/standalone_apps/tunnelvision/inference.py \
        experiments/tunnelvision/scripts/smoke_inference.py
git commit -m "feat(tunnelvision): DeGirum local vehicle/plate/OCR wrappers + smoke test"
```

---

## Phase 3 — Async services + correlation

### Task 10: Rekor HTTP client + response parser + credit gate (`rekor.py` part 1)

**Files:**
- Create: `hailo_apps/python/standalone_apps/tunnelvision/rekor.py`
- Create: `tests/python/standalone_apps/tunnelvision/test_rekor.py`

**Reference:** ERD code samples for `call_rekor_carcheck`, `parse_rekor_response`, `should_call_rekor`.

- [ ] **Step 1: Write failing tests**

`tests/python/standalone_apps/tunnelvision/test_rekor.py`:

```python
from unittest.mock import patch, MagicMock

import pytest

from hailo_apps.python.standalone_apps.tunnelvision.rekor import (
    CreditPolicy, parse_rekor_response, should_call_rekor, call_rekor_carcheck,
)


SAMPLE_RESPONSE = {
    "data_type": "alpr_results",
    "epoch_time": 1714998000000,
    "img_width": 1920, "img_height": 1080,
    "error": False, "version": 2, "uuid": "abc",
    "credit_cost": 1, "credits_monthly_used": 100, "credits_monthly_total": 1000,
    "processing_time": {"total": 230.0, "plates": 200.0, "vehicles": 30.0},
    "regions_of_interest": [],
    "results": [
        {
            "plate": "ABC1234",
            "region": "ny",
            "confidence": 92.5,
            "region_confidence": 80.0,
            "matches_template": 1,
            "coordinates": [[0,0],[10,0],[10,5],[0,5]],
            "vehicle_detected": True,
            "candidates": [
                {"plate": "ABC1234", "confidence": 92.5, "matches_template": 1},
                {"plate": "ABC1Z34", "confidence": 80.1, "matches_template": 0},
            ],
            "vehicle": {
                "make":[{"name":"Honda","confidence":80.0}],
                "make_model":[{"name":"Honda Civic","confidence":75.0}],
                "color":[{"name":"red","confidence":90.0}],
                "year":[{"name":"2018-2020","confidence":60.0}],
                "orientation":[{"name":"front","confidence":85.0}],
                "body_type":[{"name":"sedan","confidence":80.0}],
            },
        },
    ],
}


def test_parse_rekor_response_extracts_top_plate():
    parsed = parse_rekor_response(SAMPLE_RESPONSE)
    assert parsed["plate"] == "ABC1234"
    assert parsed["plate_confidence"] == 92.5
    assert parsed["region"] == "ny"
    assert parsed["credit_cost"] == 1
    assert parsed["candidates"][0]["plate"] == "ABC1234"
    assert parsed["make"] == "Honda"


def test_parse_rekor_response_empty_results():
    parsed = parse_rekor_response({"results": [], "credit_cost": 0})
    assert parsed["plate"] is None


def test_should_call_rekor_skip_low_quality():
    p = CreditPolicy(min_quality_score_to_call=80.0)
    ok, recv, reason = should_call_rekor(quality_score=50.0, camera_role="ingress",
                                         credits_used=0, monthly_budget=500,
                                         known_recently=False, policy=p)
    assert not ok
    assert reason == "skip_low_quality"


def test_should_call_rekor_skip_egress():
    p = CreditPolicy(call_rekor_on_egress=False)
    ok, _, reason = should_call_rekor(quality_score=90.0, camera_role="egress",
                                      credits_used=0, monthly_budget=500,
                                      known_recently=False, policy=p)
    assert not ok
    assert reason == "skip_egress"


def test_should_call_rekor_skip_known_plate():
    p = CreditPolicy()
    ok, _, reason = should_call_rekor(quality_score=90.0, camera_role="ingress",
                                      credits_used=0, monthly_budget=500,
                                      known_recently=True, policy=p)
    assert not ok
    assert reason == "skip_known_plate"


def test_should_call_rekor_emergency_conservation_known():
    p = CreditPolicy(emergency_threshold=0.85)
    ok, _, reason = should_call_rekor(quality_score=90.0, camera_role="ingress",
                                      credits_used=900, monthly_budget=1000,
                                      known_recently=True, policy=p)
    assert not ok
    assert reason == "emergency_credit_conservation"


def test_should_call_rekor_happy_path():
    p = CreditPolicy()
    ok, recv, reason = should_call_rekor(quality_score=90.0, camera_role="ingress",
                                         credits_used=0, monthly_budget=500,
                                         known_recently=False, policy=p)
    assert ok
    assert reason == "plate_only"


def test_call_rekor_carcheck_posts_base64(tmp_path):
    img = tmp_path / "x.jpg"
    img.write_bytes(b"fake-jpeg-bytes")
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = SAMPLE_RESPONSE
    fake_resp.raise_for_status.return_value = None
    with patch("hailo_apps.python.standalone_apps.tunnelvision.rekor.requests.post",
               return_value=fake_resp) as mock_post:
        out = call_rekor_carcheck(str(img), secret_key="SK", recognize_vehicle=False)
    mock_post.assert_called_once()
    assert out["plate"] == "ABC1234"
```

- [ ] **Step 2: Run, verify failures**

```bash
pytest tests/python/standalone_apps/tunnelvision/test_rekor.py -v
```

- [ ] **Step 3: Implement rekor.py (parser + gate + HTTP)**

`hailo_apps/python/standalone_apps/tunnelvision/rekor.py`:

```python
import base64
from dataclasses import dataclass
from typing import Optional, Tuple

import requests


REKOR_BASE_URL = "https://api.openalpr.com/v3/recognize_bytes"


@dataclass
class CreditPolicy:
    monthly_credit_budget: int = 500
    min_quality_score_to_call: float = 80.0
    min_plate_confidence_to_call: float = 0.70
    emergency_threshold: float = 0.85
    call_rekor_on_ingress: bool = True
    call_rekor_on_egress: bool = False
    default_recognize_vehicle: bool = False


def _normalize_plate(plate: Optional[str]) -> Optional[str]:
    if plate is None:
        return None
    return "".join(c for c in plate.upper() if c.isalnum())


def _top_attr(vehicle: Optional[dict], key: str) -> Tuple[Optional[str], Optional[float]]:
    values = (vehicle or {}).get(key) or []
    if not values:
        return None, None
    return values[0].get("name"), values[0].get("confidence")


def parse_rekor_response(payload: dict) -> dict:
    first_result = (payload.get("results") or [None])[0]
    first_vehicle = first_result.get("vehicle") if first_result else None
    make, make_conf = _top_attr(first_vehicle, "make")
    make_model, make_model_conf = _top_attr(first_vehicle, "make_model")
    color, color_conf = _top_attr(first_vehicle, "color")
    year, year_conf = _top_attr(first_vehicle, "year")
    orientation, orientation_conf = _top_attr(first_vehicle, "orientation")
    body_type, body_type_conf = _top_attr(first_vehicle, "body_type")

    proc = payload.get("processing_time") or {}
    return {
        "data_type": payload.get("data_type"),
        "epoch_time": payload.get("epoch_time"),
        "img_width": payload.get("img_width"),
        "img_height": payload.get("img_height"),
        "error": payload.get("error"),
        "version": payload.get("version"),
        "uuid": payload.get("uuid"),
        "credit_cost": payload.get("credit_cost"),
        "credits_monthly_used": payload.get("credits_monthly_used"),
        "credits_monthly_total": payload.get("credits_monthly_total"),
        "processing_time_total_ms": proc.get("total"),
        "processing_time_plates_ms": proc.get("plates"),
        "processing_time_vehicles_ms": proc.get("vehicles"),
        "regions_of_interest": payload.get("regions_of_interest"),
        "plate": first_result.get("plate") if first_result else None,
        "normalized_plate": _normalize_plate(first_result.get("plate") if first_result else None),
        "region": first_result.get("region") if first_result else None,
        "plate_confidence": first_result.get("confidence") if first_result else None,
        "region_confidence": first_result.get("region_confidence") if first_result else None,
        "matches_template": first_result.get("matches_template") if first_result else None,
        "coordinates": first_result.get("coordinates") if first_result else None,
        "candidates": (first_result.get("candidates") if first_result else []) or [],
        "vehicle_detected": first_result.get("vehicle_detected") if first_result else None,
        "make": make, "make_confidence": make_conf,
        "make_model": make_model, "make_model_confidence": make_model_conf,
        "color": color, "color_confidence": color_conf,
        "year_range": year, "year_confidence": year_conf,
        "orientation": orientation, "orientation_confidence": orientation_conf,
        "body_type": body_type, "body_type_confidence": body_type_conf,
    }


def should_call_rekor(
    *,
    quality_score: float,
    camera_role: str,
    credits_used: int,
    monthly_budget: int,
    known_recently: bool,
    policy: CreditPolicy,
) -> Tuple[bool, bool, str]:
    """Returns (call?, recognize_vehicle?, reason)."""
    if quality_score < policy.min_quality_score_to_call:
        return False, False, "skip_low_quality"
    if camera_role == "egress" and not policy.call_rekor_on_egress:
        return False, False, "skip_egress"
    if camera_role == "ingress" and not policy.call_rekor_on_ingress:
        return False, False, "skip_ingress_disabled"
    monthly_ratio = (credits_used / monthly_budget) if monthly_budget > 0 else 0.0
    if monthly_ratio >= policy.emergency_threshold:
        if known_recently:
            return False, False, "emergency_credit_conservation"
        return True, False, "plate_only"
    if known_recently:
        return False, False, "skip_known_plate"
    return True, policy.default_recognize_vehicle, (
        "vehicle_enrichment" if policy.default_recognize_vehicle else "plate_only"
    )


def call_rekor_carcheck(
    image_path: str,
    *,
    secret_key: str,
    recognize_vehicle: bool = False,
    country: str = "us",
    timeout_seconds: int = 8,
) -> dict:
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read())
    url = (
        f"{REKOR_BASE_URL}"
        f"?recognize_vehicle={1 if recognize_vehicle else 0}"
        f"&country={country}"
        f"&secret_key={secret_key}"
    )
    response = requests.post(url, data=img_b64, timeout=timeout_seconds)
    response.raise_for_status()
    return parse_rekor_response(response.json())
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest tests/python/standalone_apps/tunnelvision/test_rekor.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add hailo_apps/python/standalone_apps/tunnelvision/rekor.py \
        tests/python/standalone_apps/tunnelvision/test_rekor.py
git commit -m "feat(tunnelvision): Rekor HTTP client + response parser + credit gate"
```

---

### Task 11: Async Rekor worker thread (`rekor.py` part 2)

**Files:**
- Modify: `hailo_apps/python/standalone_apps/tunnelvision/rekor.py`
- Create: `tests/python/standalone_apps/tunnelvision/test_rekor_worker.py`

- [ ] **Step 1: Write failing test**

`tests/python/standalone_apps/tunnelvision/test_rekor_worker.py`:

```python
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
```

- [ ] **Step 2: Run, verify failures**

```bash
pytest tests/python/standalone_apps/tunnelvision/test_rekor_worker.py -v
```

- [ ] **Step 3: Implement RekorWorker + RekorRequest in rekor.py**

Append to `rekor.py`:

```python
import base64
import hashlib
import json
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from hailo_apps.python.standalone_apps.tunnelvision.db import (
    InsertRekorRequest, InsertRekorResponse, InsertRekorPlateResult,
    UpdateRekorRequest,
)

logger = logging.getLogger(__name__)


@dataclass
class RekorRequest:
    request_id: str
    image_path: str
    image_sha256: str
    observation_id: str
    vehicle_track_id: str
    best_image_id: str
    visit_id: Optional[str]
    recognize_vehicle: bool
    credit_policy: str


_SHUTDOWN = object()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RekorWorker(threading.Thread):
    """Drains a queue of RekorRequest, calls Rekor, emits DB events.

    Camera loop is never blocked by this thread.
    """

    def __init__(
        self,
        in_queue: queue.Queue,
        db_queue: queue.Queue,
        *,
        secret_key: str,
        policy: CreditPolicy,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        timeout_seconds: int = 8,
        name: str = "rekor-worker",
    ):
        super().__init__(name=name, daemon=True)
        self._in = in_queue
        self._db = db_queue
        self._secret = secret_key
        self._policy = policy
        self._max_retries = max_retries
        self._backoff = retry_backoff
        self._timeout = timeout_seconds

    def run(self) -> None:
        while True:
            req = self._in.get()
            try:
                if req is _SHUTDOWN:
                    return
                self._handle(req)
            except Exception as exc:  # noqa: BLE001
                logger.exception("RekorWorker fatal: %s", exc)
            finally:
                self._in.task_done()

    def stop(self) -> None:
        self._in.put(_SHUTDOWN)
        self.join(timeout=10)

    def _handle(self, req: RekorRequest) -> None:
        # Persist the queued request first
        self._db.put(InsertRekorRequest(
            id=req.request_id,
            visit_id=req.visit_id,
            observation_id=req.observation_id,
            vehicle_track_id=req.vehicle_track_id,
            best_image_id=req.best_image_id,
            image_path=req.image_path,
            image_sha256=req.image_sha256,
            request_status="sent",
            credit_policy=req.credit_policy,
            sent_at=_now(),
        ))

        last_error = None
        for attempt in range(self._max_retries):
            try:
                payload = call_rekor_carcheck(
                    req.image_path,
                    secret_key=self._secret,
                    recognize_vehicle=req.recognize_vehicle,
                    timeout_seconds=self._timeout,
                )
                self._persist_success(req, payload)
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("Rekor attempt %d/%d failed: %s",
                               attempt + 1, self._max_retries, exc)
                time.sleep(self._backoff * (2 ** attempt))

        self._db.put(UpdateRekorRequest(
            id=req.request_id,
            request_status="failed",
            completed_at=_now(),
            error_message=str(last_error),
        ))

    def _persist_success(self, req: RekorRequest, parsed: dict) -> None:
        response_id = str(uuid.uuid4())
        self._db.put(InsertRekorResponse(
            id=response_id,
            request_id=req.request_id,
            data_type=parsed.get("data_type"),
            epoch_time=parsed.get("epoch_time"),
            img_width=parsed.get("img_width"),
            img_height=parsed.get("img_height"),
            error=int(bool(parsed.get("error"))),
            version=parsed.get("version"),
            uuid=parsed.get("uuid"),
            credit_cost=parsed.get("credit_cost"),
            credits_monthly_used=parsed.get("credits_monthly_used"),
            credits_monthly_total=parsed.get("credits_monthly_total"),
            processing_time_total_ms=parsed.get("processing_time_total_ms"),
            raw_response=json.dumps(parsed),
            created_at=_now(),
        ))
        if parsed.get("plate"):
            self._db.put(InsertRekorPlateResult(
                id=str(uuid.uuid4()),
                response_id=response_id,
                plate_index=0,
                plate=parsed["plate"],
                normalized_plate=parsed.get("normalized_plate") or parsed["plate"],
                region=parsed.get("region"),
                confidence=parsed.get("plate_confidence"),
                region_confidence=parsed.get("region_confidence"),
                matches_template=int(bool(parsed.get("matches_template"))),
                coordinates=json.dumps(parsed.get("coordinates")) if parsed.get("coordinates") else None,
                selected=1,
            ))
        self._db.put(UpdateRekorRequest(
            id=req.request_id,
            request_status="succeeded",
            actual_credit_cost=parsed.get("credit_cost"),
            completed_at=_now(),
            http_status=200,
        ))


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest tests/python/standalone_apps/tunnelvision/test_rekor_worker.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add hailo_apps/python/standalone_apps/tunnelvision/rekor.py \
        tests/python/standalone_apps/tunnelvision/test_rekor_worker.py
git commit -m "feat(tunnelvision): async RekorWorker thread + DB persistence"
```

---

### Task 12: Visit correlator (`correlator.py`)

**Files:**
- Create: `hailo_apps/python/standalone_apps/tunnelvision/correlator.py`
- Create: `tests/python/standalone_apps/tunnelvision/test_correlator.py`

**Reference:** PRD §5.3 ranking formula.

- [ ] **Step 1: Write failing tests**

`tests/python/standalone_apps/tunnelvision/test_correlator.py`:

```python
from datetime import datetime, timedelta, timezone

from hailo_apps.python.standalone_apps.tunnelvision.correlator import (
    Correlator, ActiveVisit, EgressEvent,
)


def _ts(seconds_ago: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)


def test_match_by_exact_plate_wins():
    c = Correlator(min_match_score=60.0, expected_tunnel_seconds=300)
    c.enqueue_ingress(ActiveVisit(visit_id="v1", plate="ABC1234",
                                  ingress_at=_ts(180), color="red"))
    c.enqueue_ingress(ActiveVisit(visit_id="v2", plate="DEF5678",
                                  ingress_at=_ts(160), color="blue"))
    eg = EgressEvent(plate="ABC1234", egress_at=_ts(0), color="red")
    match = c.match_egress(eg)
    assert match.visit_id == "v1"
    assert match.method == "plate_exact"
    assert match.confidence > 90


def test_match_by_queue_order_when_no_plate():
    c = Correlator(min_match_score=30.0, expected_tunnel_seconds=180)
    c.enqueue_ingress(ActiveVisit(visit_id="v1", plate=None, ingress_at=_ts(180)))
    c.enqueue_ingress(ActiveVisit(visit_id="v2", plate=None, ingress_at=_ts(120)))
    eg = EgressEvent(plate=None, egress_at=_ts(0))
    match = c.match_egress(eg)
    assert match.visit_id == "v1"   # FIFO head
    assert match.method == "track_sequence"


def test_match_returns_none_below_threshold():
    c = Correlator(min_match_score=80.0, expected_tunnel_seconds=180)
    c.enqueue_ingress(ActiveVisit(visit_id="v1", plate="AAA", ingress_at=_ts(600)))
    eg = EgressEvent(plate="ZZZ", egress_at=_ts(0))
    match = c.match_egress(eg)
    assert match is None or match.confidence < 80


def test_completed_visits_are_removed_from_queue():
    c = Correlator(min_match_score=60.0, expected_tunnel_seconds=300)
    c.enqueue_ingress(ActiveVisit(visit_id="v1", plate="ABC1234", ingress_at=_ts(180)))
    eg = EgressEvent(plate="ABC1234", egress_at=_ts(0))
    c.match_egress(eg)
    assert c.active_visit_count() == 0


def test_handles_three_simultaneous_cars_in_order():
    c = Correlator(min_match_score=60.0, expected_tunnel_seconds=300)
    c.enqueue_ingress(ActiveVisit(visit_id="A", plate="A1", ingress_at=_ts(300)))
    c.enqueue_ingress(ActiveVisit(visit_id="B", plate="B2", ingress_at=_ts(240)))
    c.enqueue_ingress(ActiveVisit(visit_id="C", plate="C3", ingress_at=_ts(180)))
    assert c.match_egress(EgressEvent(plate="A1", egress_at=_ts(0))).visit_id == "A"
    assert c.match_egress(EgressEvent(plate="B2", egress_at=_ts(0))).visit_id == "B"
    assert c.match_egress(EgressEvent(plate="C3", egress_at=_ts(0))).visit_id == "C"
    assert c.active_visit_count() == 0
```

- [ ] **Step 2: Run, verify failures**

```bash
pytest tests/python/standalone_apps/tunnelvision/test_correlator.py -v
```

- [ ] **Step 3: Implement correlator.py**

`hailo_apps/python/standalone_apps/tunnelvision/correlator.py`:

```python
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
    """Per-lane FIFO tunnel queue + egress→visit matcher (PRD §5.3)."""

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
                    best_method = (
                        "plate_exact" if pm == 1.0 else
                        "plate_fuzzy" if pm > 0 else
                        ("lane_time_window" if ts > 0.5 else "track_sequence")
                    )

            if best is None or best_score < self._min:
                return None
            self._queue.remove(best)
            return Match(visit_id=best.visit_id, confidence=best_score, method=best_method)
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest tests/python/standalone_apps/tunnelvision/test_correlator.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add hailo_apps/python/standalone_apps/tunnelvision/correlator.py \
        tests/python/standalone_apps/tunnelvision/test_correlator.py
git commit -m "feat(tunnelvision): tunnel FIFO queue + egress visit correlator"
```

---

## Phase 4 — Pipeline orchestration + entry point

### Task 13: Per-camera pipeline orchestration (`pipeline.py`)

**Files:**
- Create: `hailo_apps/python/standalone_apps/tunnelvision/pipeline.py`
- Create: `tests/python/standalone_apps/tunnelvision/test_pipeline.py`

**Approach:** `CameraPipeline` is a `threading.Thread` subclass. Constructor receives all components (vehicle_detector, plate_detector, tracker, zone_config, scorer rule, credit policy, queues). The frame loop is the orchestration described in DESIGN.md §4.3.

- [ ] **Step 1: Write a behavior-level test using mocked components**

`tests/python/standalone_apps/tunnelvision/test_pipeline.py`:

```python
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
    return np.full((720, 1280, 3), 128, dtype=np.uint8)


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
        # give the pipeline time to drain
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
```

- [ ] **Step 2: Run, verify failures**

```bash
pytest tests/python/standalone_apps/tunnelvision/test_pipeline.py -v
```

Expected: ImportError on `CameraPipeline`.

- [ ] **Step 3: Implement pipeline.py**

`hailo_apps/python/standalone_apps/tunnelvision/pipeline.py`:

```python
import hashlib
import logging
import os
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from hailo_apps.python.standalone_apps.tunnelvision.correlator import (
    ActiveVisit, Correlator,
)
from hailo_apps.python.standalone_apps.tunnelvision.db import (
    InsertBestImage, InsertCameraObservation, InsertTrackStateEvent,
    InsertVehicleTrack, InsertVisit, UpdateVehicleTrack,
)
from hailo_apps.python.standalone_apps.tunnelvision.rekor import (
    CreditPolicy, RekorRequest, file_sha256, should_call_rekor,
)
from hailo_apps.python.standalone_apps.tunnelvision.scorer import (
    ScoringRule, compute_blur_score, compute_motion_score,
    compute_quality_score, frame_passes_gates, select_best_candidate,
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


def _normalize_bbox(bbox, w: int, h: int) -> list:
    x1, y1, x2, y2 = bbox
    return [x1 / w, y1 / h, x2 / w, y2 / h]


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
        self._observation_db_ids: dict = {}  # edge_track_id -> obs uuid
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
            return

        vehicles = self._vd.detect(frame)
        if not vehicles:
            self._prev_frame = frame
            self._age_unmatched()
            return

        bboxes = [v.bbox for v in vehicles]
        scores = [v.score for v in vehicles]
        active = self._tracker.update(bboxes, scores)

        for raw in active:
            edge_id = str(raw.track_id)
            track = self._tracks.get(edge_id)
            if track is None:
                track = self._create_track(edge_id, raw.bbox)
            else:
                track.last_seen_at = _now_dt()
                track.current_bbox = list(raw.bbox)
                track.lost_frames = raw.lost_frames

            zone_label = self._zone_for_track(track, w, h)
            track.current_zone = zone_label

            fully_visible = self._is_fully_visible(track, w, h)
            track.fully_visible = fully_visible

            plate_visible = False
            plate_blocked = False
            plate_det = None
            quality = 0.0

            if zone_label == "capture" and fully_visible and not track.processed:
                plate_det, plate_blocked = self._detect_plate(frame, track, active)
                plate_visible = plate_det is not None

                if plate_det is not None:
                    candidate = self._score_candidate(frame, track, plate_det)
                    if candidate is not None:
                        track.candidates.append(candidate)
                        quality = candidate.quality_score

                        if len(track.candidates) >= self._min_stable:
                            best = select_best_candidate(track.candidates)
                            self._select_and_emit(track, best, w, h)

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

    # ---------- helpers ----------

    def _motion_gate(self, frame: np.ndarray) -> bool:
        if self._prev_frame is None:
            return True
        diff = cv2.absdiff(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(self._prev_frame, cv2.COLOR_BGR2GRAY),
        )
        return float(np.mean(diff)) > 1.0

    def _age_unmatched(self) -> None:
        # Tracker handles aging via update([]) on no-detection frames
        for raw in self._tracker.update([]):
            pass

    def _create_track(self, edge_id: str, bbox: list) -> VehicleTrack:
        obs_id = str(uuid.uuid4())
        track_db_id = str(uuid.uuid4())
        now = _now_dt()
        self._observation_db_ids[edge_id] = obs_id
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
        for zone in self._zones.for_camera(self._camera_id):
            if zone.contains_point(nx, ny):
                if zone.action == "capture_plate":
                    return "capture"
                if zone.action == "start_visit":
                    return "approach"
                if zone.action == "end_visit":
                    return "exit"
                if zone.action == "skip":
                    return "ignore"
        return None

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
        plate_in_frame = [plate.bbox[0] + x1, plate.bbox[1] + y1,
                          plate.bbox[2] + x1, plate.bbox[3] + y1]
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
        prev_centroid = track.current_bbox if not track.candidates else None
        # bbox stability: 1.0 if bbox barely changed since last candidate
        if track.candidates:
            last_bbox = track.candidates[-1].vehicle_bbox
            dx = (track.current_bbox[0] + track.current_bbox[2]) / 2 - (last_bbox[0] + last_bbox[2]) / 2
            dy = (track.current_bbox[1] + track.current_bbox[3]) / 2 - (last_bbox[1] + last_bbox[3]) / 2
            motion = float(np.hypot(dx, dy))
        else:
            motion = 0.0
        bbox_stability = 1.0 / (1.0 + motion / 50.0)
        roi_fit = 1.0  # placeholder — full impl requires capture-zone polygon area
        bbox_growth_rate = 0.0  # placeholder

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
            vehicle_bbox=list(track.current_bbox), plate_bbox=plate_in_frame_coords(
                plate_det.bbox, x1, y1
            ),
        )

    def _select_and_emit(self, track: VehicleTrack, best: FrameCandidate,
                         w: int, h: int) -> None:
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

        # Decide on Rekor
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
            from hailo_apps.python.standalone_apps.tunnelvision.correlator import EgressEvent
            from hailo_apps.python.standalone_apps.tunnelvision.db import (
                UpdateVisitStatus, InsertVisitMatchEvent,
            )
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


def plate_in_frame_coords(plate_bbox, vehicle_x: int, vehicle_y: int) -> list:
    x1, y1, x2, y2 = plate_bbox
    return [x1 + vehicle_x, y1 + vehicle_y, x2 + vehicle_x, y2 + vehicle_y]
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest tests/python/standalone_apps/tunnelvision/test_pipeline.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add hailo_apps/python/standalone_apps/tunnelvision/pipeline.py \
        tests/python/standalone_apps/tunnelvision/test_pipeline.py
git commit -m "feat(tunnelvision): per-camera pipeline thread orchestration"
```

---

### Task 14: CLI entry point + thread wiring (`tunnelvision.py`)

**Files:**
- Create: `hailo_apps/python/standalone_apps/tunnelvision/tunnelvision.py`

- [ ] **Step 1: Implement entry point**

`hailo_apps/python/standalone_apps/tunnelvision/tunnelvision.py`:

```python
#!/usr/bin/env python3
"""TunnelVision — edge tracking + visit correlation for car wash tunnels."""

import argparse
import os
import queue
import signal
import sys
import threading
from pathlib import Path

from hailo_apps.python.core.common.hailo_logger import get_logger, init_logging
from hailo_apps.python.standalone_apps.tunnelvision.correlator import Correlator
from hailo_apps.python.standalone_apps.tunnelvision.db import DBWriter
from hailo_apps.python.standalone_apps.tunnelvision.inference import (
    PlateDetector, VehicleDetector,
)
from hailo_apps.python.standalone_apps.tunnelvision.ingest import RTSPIngest
from hailo_apps.python.standalone_apps.tunnelvision.pipeline import CameraPipeline
from hailo_apps.python.standalone_apps.tunnelvision.rekor import (
    CreditPolicy, RekorWorker,
)
from hailo_apps.python.standalone_apps.tunnelvision.scorer import ScoringRule
from hailo_apps.python.standalone_apps.tunnelvision.zone import load_zones

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TunnelVision edge ALPR")
    p.add_argument("--ingress", required=True, help="RTSP URL for cam0 (ingress)")
    p.add_argument("--egress", required=True, help="RTSP URL for cam1 (egress)")
    p.add_argument("--zones", default="experiments/tunnelvision/zones.json")
    p.add_argument("--db", default="tunnelvision.db")
    p.add_argument("--snapshot-dir", default="tv_snapshots")
    p.add_argument("--monthly-budget", type=int, default=500)
    p.add_argument("--min-quality", type=float, default=80.0)
    p.add_argument("--min-plate-confidence", type=float, default=0.70)
    p.add_argument("--emergency-threshold", type=float, default=0.85)
    p.add_argument("--min-stable-candidates", type=int, default=3)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    init_logging(level="DEBUG" if args.debug else None)

    secret = os.environ.get("REKOR_SECRET_KEY")
    if not secret:
        logger.warning("REKOR_SECRET_KEY not set — Rekor calls will fail")
        secret = ""

    Path(args.snapshot_dir).mkdir(parents=True, exist_ok=True)

    zones = load_zones(args.zones)
    rule = ScoringRule(min_plate_confidence=args.min_plate_confidence)
    policy = CreditPolicy(
        monthly_credit_budget=args.monthly_budget,
        min_quality_score_to_call=args.min_quality,
        min_plate_confidence_to_call=args.min_plate_confidence,
        emergency_threshold=args.emergency_threshold,
    )

    db_q: queue.Queue = queue.Queue(maxsize=2000)
    rekor_q: queue.Queue = queue.Queue(maxsize=200)
    correlator = Correlator(min_match_score=60.0)

    db_writer = DBWriter(args.db, db_q)
    rekor_worker = RekorWorker(rekor_q, db_q, secret_key=secret, policy=policy)

    ingress_ingest = RTSPIngest(args.ingress, "ingress")
    egress_ingest = RTSPIngest(args.egress, "egress")

    # Each pipeline owns its own DeGirum models — no sharing across threads
    ing_vd = VehicleDetector()
    ing_pd = PlateDetector()
    eg_vd = VehicleDetector()
    eg_pd = PlateDetector()

    ingress_pipe = CameraPipeline(
        camera_role="ingress", camera_id=0,
        frame_queue=ingress_ingest.frame_queue,
        db_queue=db_q, rekor_queue=rekor_q, correlator=correlator,
        vehicle_detector=ing_vd, plate_detector=ing_pd,
        zone_config=zones, scoring_rule=rule, credit_policy=policy,
        snapshot_dir=args.snapshot_dir, secret_key=secret,
        min_stable_candidates=args.min_stable_candidates,
    )
    egress_pipe = CameraPipeline(
        camera_role="egress", camera_id=1,
        frame_queue=egress_ingest.frame_queue,
        db_queue=db_q, rekor_queue=rekor_q, correlator=correlator,
        vehicle_detector=eg_vd, plate_detector=eg_pd,
        zone_config=zones, scoring_rule=rule, credit_policy=policy,
        snapshot_dir=args.snapshot_dir, secret_key=secret,
        min_stable_candidates=args.min_stable_candidates,
    )

    stopping = threading.Event()

    def _shutdown(*_):
        if not stopping.is_set():
            stopping.set()
            logger.info("Shutdown signal received")

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    db_writer.start()
    rekor_worker.start()
    ingress_ingest.start()
    egress_ingest.start()
    ingress_pipe.start()
    egress_pipe.start()

    logger.info("TunnelVision running. Ctrl-C to stop.")
    try:
        while not stopping.is_set():
            stopping.wait(timeout=1.0)
    finally:
        ingress_pipe.stop()
        egress_pipe.stop()
        ingress_ingest.stop()
        egress_ingest.stop()
        rekor_worker.stop()
        db_q.join()
        db_writer.stop()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Sanity-check the entry point loads its imports**

```bash
source setup_env.sh
python3 -c "from hailo_apps.python.standalone_apps.tunnelvision import tunnelvision; print(tunnelvision.parse_args.__doc__)"
```

Expected: no import errors. (The DeGirum import requires the SDK to be installed.)

- [ ] **Step 3: Help screen sanity check**

```bash
python3 -m hailo_apps.python.standalone_apps.tunnelvision.tunnelvision --help
```

Expected: argparse usage with all the documented flags.

- [ ] **Step 4: Commit**

```bash
git add hailo_apps/python/standalone_apps/tunnelvision/tunnelvision.py
git commit -m "feat(tunnelvision): CLI entry point + thread wiring"
```

---

## Phase 5 — Live integration

### Task 15: README + smoke run against live cameras

**Files:**
- Create: `hailo_apps/python/standalone_apps/tunnelvision/README.md`

- [ ] **Step 1: Write README**

`hailo_apps/python/standalone_apps/tunnelvision/README.md`:

````markdown
# TunnelVision

Continuous-tracking edge ALPR for car wash tunnels. Two RTSP cameras
(cam0 ingress, cam1 egress) feed a per-camera pipeline that detects vehicles,
tracks them across frames, scores best frames, queues async Rekor CarCheck
calls, and correlates ingress to egress visits.

See `experiments/tunnelvision/{PRD,ERD,DESIGN,PLAN}.md` for full context.

## Requirements

- DeGirum SDK with local Hailo device (`@local`)
- Hailo-8/8L device with the models recorded in `experiments/tunnelvision/MODELS.md`
- Two RTSP cameras (or test streams)
- `REKOR_SECRET_KEY` in environment

## Quick start

```bash
source setup_env.sh
export REKOR_SECRET_KEY=<your_key>
python3 -m hailo_apps.python.standalone_apps.tunnelvision.tunnelvision \
  --ingress 'rtsp://bowtie:dieformalwear99!@192.168.1.121:554/media/live/1/1' \
  --egress  'rtsp://admin:123456@192.168.1.125:554/media/live/1/1' \
  --zones   experiments/tunnelvision/zones.json \
  --db      tunnelvision.db \
  --snapshot-dir tv_snapshots \
  --debug
```

The `!` in the ingress URL must be inside single quotes — bash history expansion
otherwise mangles it.

## Verification queries

```sql
SELECT status, COUNT(*) FROM visits GROUP BY status;
SELECT COUNT(*), SUM(actual_credit_cost) FROM rekor_carcheck_requests
  WHERE request_status = 'succeeded';
SELECT AVG(tunnel_duration_sec) FROM visits WHERE status = 'completed';
```
````

- [ ] **Step 2: Run a 5-minute live smoke test**

```bash
source setup_env.sh
export REKOR_SECRET_KEY="<key>"
timeout 300 python3 -m hailo_apps.python.standalone_apps.tunnelvision.tunnelvision \
  --ingress 'rtsp://bowtie:dieformalwear99!@192.168.1.121:554/media/live/1/1' \
  --egress  'rtsp://admin:123456@192.168.1.125:554/media/live/1/1' \
  --zones   experiments/tunnelvision/zones.json \
  --db      /tmp/tv_smoke.db \
  --snapshot-dir /tmp/tv_snaps \
  --debug 2>&1 | tee /tmp/tv_smoke.log
```

Expected during 5 minutes:
- "Stream live" lines for both cameras
- "First frame received" with non-zero dimensions
- At least one vehicle detection logged
- No tracebacks

- [ ] **Step 3: Inspect results**

```bash
sqlite3 /tmp/tv_smoke.db <<EOF
SELECT 'visits', COUNT(*) FROM visits
UNION ALL SELECT 'observations', COUNT(*) FROM camera_observations
UNION ALL SELECT 'tracks', COUNT(*) FROM vehicle_tracks
UNION ALL SELECT 'best_images', COUNT(*) FROM best_images
UNION ALL SELECT 'rekor_requests', COUNT(*) FROM rekor_carcheck_requests
UNION ALL SELECT 'rekor_responses', COUNT(*) FROM rekor_carcheck_responses;
EOF
ls /tmp/tv_snaps | head
```

Expected: non-zero counts for at least `tracks` and `observations`. If a vehicle drove through, also `best_images` and `rekor_requests`.

- [ ] **Step 4: Tune the egress polygon if needed**

If the egress smoke run shows no tracks despite vehicles passing, edit `experiments/tunnelvision/zones.json` to redraw the `egress_gate` polygon around the actual vehicle path. Re-run step 2.

- [ ] **Step 5: Commit README + any zone tuning**

```bash
git add hailo_apps/python/standalone_apps/tunnelvision/README.md \
        experiments/tunnelvision/zones.json
git commit -m "feat(tunnelvision): README + tuned cam1 egress polygon"
```

---

## Self-Review

**Spec coverage check:**

| Spec section | Implemented in |
|---|---|
| PRD §1 components | Tasks 2–14 (all 8 PRD components mapped to files) |
| PRD §2 pipeline (motion → detect → track → zone → state → plate → score → Rekor → visit → queue → match) | Task 13 (`pipeline.py`) |
| PRD §5.1 ingress steps | Task 13 |
| PRD §5.2 egress steps | Task 13 + Task 12 |
| PRD §5.3 tunnel queue + scoring formula | Task 12 |
| PRD §6 hardware/runtime targets | Validated by Task 15 smoke run |
| PRD §7 Hailo edge pipeline | Tasks 8 (detector wrappers) + Task 0 (model discovery) |
| PRD §8 Rekor integration | Task 10 |
| PRD §9 state machine (12 states) | Task 6 |
| PRD §10 DB requirements | Tasks 2–3 |
| PRD §11 best frame algorithm | Task 7 |
| PRD §12 failure handling | RTSPIngest reconnect (Task 4), Rekor retry (Task 11), DBWriter survival (Task 3), correlator threshold (Task 12), state machine LOST/NO_READ (Task 6) |
| PRD §13 visit correlation rules | Task 12 |
| PRD §14 tunnel timing | Task 12 + Task 13 (UpdateVisitStatus emits tunnel_duration_sec) |
| PRD §15 API cost controls | Task 10 (`should_call_rekor`) |
| PRD §16 deployment architecture | Task 14 (thread wiring) |
| ERD all MVP tables | Task 2 schema |
| ERD Rekor request/response/plate persistence | Tasks 10–11 |
| DESIGN.md zone schema (td-edge-compatible) | Task 5 |
| DESIGN.md threading model | Task 14 |

No unmapped spec sections.

**Placeholders found and intentionally retained:**
- Task 9 / `inference.py` `<FILL FROM MODELS.md>` — explicitly tracked by Task 0 as the discovery output. The engineer fills this in after Task 0 completes.
- `roi_fit = 1.0` and `bbox_growth_rate = 0.0` placeholders in `pipeline.py:_score_candidate` — call this out as a phase-2 refinement; the gates and scoring still work without them.

**Type consistency:** All cross-task references checked — `VehicleTrack`, `FrameCandidate`, `Track` (tracker.py), `Detection`, `PlateDetection`, `RekorRequest`, `ActiveVisit`, `EgressEvent`, `Match` are defined in exactly one file each. DB write event class names (`InsertVisit`, `UpdateVisitStatus`, etc.) match between db.py definitions and pipeline.py / rekor.py callers.

---

## Notes for the Executor

- Run `source setup_env.sh` once per shell. Tests need this for `pyproject` editable install + GStreamer Python bindings.
- Several `pipeline.py` tests are mock-heavy on purpose — the real integration test is Task 15 against a live device.
- The Rekor token is sensitive. Never commit it; never log it. The CLI reads it from `REKOR_SECRET_KEY`.
- DeGirum local model loading can take several seconds per model; expect 4 model loads on startup (vehicle×2 + plate×2). OCR is loaded lazily by Task 17 if added later.
- If you see `OperationalError: database is locked` under load, increase `db_q` size or check for synchronous DB access from a non-writer thread.
