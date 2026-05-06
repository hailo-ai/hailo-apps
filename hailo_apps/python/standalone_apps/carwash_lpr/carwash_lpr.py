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
