#!/usr/bin/env bash
# TunnelVision — jajamaica test (Advance Car Wash, Jamaica Queens)
#
# Camera role mapping for this site:
#   physical cam2 — WashiFi Axis at 192.168.35.121 (root/carwash)
#       → camera_id=0  → INGRESS
#       → reachable via eth1 (separate subnet from the egress NVR)
#   physical cam1 — NVR ch801 washexitpass at 192.168.1.50 (ai/ai123456)
#       → camera_id=1  → EGRESS
#       → reachable via the standard LAN (eth0)
#
# Run from repo root on a Pi configured for both subnets:
#   bash experiments/tunnelvision/scripts/run_jajamaica.sh
#
# Stop with Ctrl-C. Inspect tunnelvision-jajamaica.db with sqlite3.
#
# NOTE: zones-jajamaica.json contains PLACEHOLDER polygons (full-frame defaults).
# At first live test, tune the ingress_gate, lpr_zone, and egress_gate polygons
# against real WashiFi/NVR frames before relying on visit data.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
# shellcheck disable=SC1091
source setup_env.sh

if [ -z "${REKOR_SECRET_KEY:-}" ]; then
  echo "WARNING: REKOR_SECRET_KEY not set — Rekor calls will fail (visit tracking still works)"
fi

mkdir -p tv_snapshots/jajamaica

exec python3 -m hailo_apps.python.standalone_apps.tunnelvision.tunnelvision \
  --ingress 'rtsp://root:carwash@192.168.35.121/axis-media/media.amp' \
  --egress  'rtsp://ai:ai123456@192.168.1.50:554/Streaming/Channels/801/?transportmode=unicast' \
  --zones   experiments/tunnelvision/zones-jajamaica.json \
  --db      tunnelvision-jajamaica.db \
  --snapshot-dir tv_snapshots/jajamaica \
  --debug "$@"
