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

        # Only write plate_read when a session was found
        read_id = self._storage.write_plate_read(
            camera_id="egress",
            timestamp=timestamp,
            plate_string=plate_string,
            confidence=confidence,
            snapshot_path=snapshot_path,
            source=source,
        )

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
