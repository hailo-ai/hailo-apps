#!/usr/bin/env python3
"""TunnelVision — edge tracking + visit correlation for car wash tunnels."""

import argparse
import os
import queue
import signal
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
    p.add_argument("--zones", default="experiments/tunnelvision/zones-lalaland.json")
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
