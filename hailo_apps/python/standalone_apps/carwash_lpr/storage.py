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
