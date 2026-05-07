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
