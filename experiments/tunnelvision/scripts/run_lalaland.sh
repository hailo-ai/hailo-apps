#!/usr/bin/env bash
# TunnelVision — lalaland test (local cameras at 192.168.1.121 / 192.168.1.125)
#
# Camera role mapping for this site:
#   physical cam1 (192.168.1.121, bowtie creds)  → camera_id=0  → INGRESS
#   physical cam2 (192.168.1.125, admin creds)   → camera_id=1  → EGRESS
#
# Run from repo root:
#   bash experiments/tunnelvision/scripts/run_lalaland.sh
#
# Stop with Ctrl-C. Inspect tunnelvision-lalaland.db with sqlite3.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
# shellcheck disable=SC1091
source setup_env.sh

if [ -z "${REKOR_SECRET_KEY:-}" ]; then
  echo "WARNING: REKOR_SECRET_KEY not set — Rekor calls will fail (visit tracking still works)"
fi

mkdir -p tv_snapshots/lalaland

exec python3 -m hailo_apps.python.standalone_apps.tunnelvision.tunnelvision \
  --ingress 'rtsp://bowtie:dieformalwear99!@192.168.1.121:554/media/live/1/1' \
  --egress  'rtsp://admin:123456@192.168.1.125:554/media/live/1/1' \
  --zones   experiments/tunnelvision/zones-lalaland.json \
  --db      tunnelvision-lalaland.db \
  --snapshot-dir tv_snapshots/lalaland \
  --debug "$@"
