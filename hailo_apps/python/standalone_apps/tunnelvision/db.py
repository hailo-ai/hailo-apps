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


import logging
import queue
import threading
from dataclasses import dataclass
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
