#!/usr/bin/env bash
# TunnelVision — jajamaica test via SSH port-forward through cloudflared
#
# Run this from a remote machine (e.g. your laptop) when you can't be on the
# Jamaica LAN directly. Requires an open SSH tunnel that forwards both camera
# ports to localhost. Open the tunnel in a separate terminal first:
#
#   ssh -o ProxyCommand='cloudflared access ssh --hostname %h' \
#       -L 8121:192.168.35.121:554 \
#       -L 8150:192.168.1.50:554 \
#       -N td-pi@ssh-metal-pi.tuxedodrive.dev
#
# (-N opens forward-only, no shell. Leave the terminal running for the duration
# of the test.)
#
# Then run this script from another terminal:
#
#   bash experiments/tunnelvision/scripts/run_jajamaica_via_tunnel.sh
#
# Stop the app with Ctrl-C, then close the SSH tunnel.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
# shellcheck disable=SC1091
source setup_env.sh

if [ -z "${REKOR_SECRET_KEY:-}" ]; then
  echo "WARNING: REKOR_SECRET_KEY not set — Rekor calls will fail (visit tracking still works)"
fi

# Pre-flight: confirm both forwards are open
python3 <<'PY'
import socket, sys
for host, port, label in [("localhost", 8121, "WashiFi ingress"),
                          ("localhost", 8150, "NVR egress")]:
    try:
        socket.create_connection((host, port), timeout=3).close()
        print(f"  OK   localhost:{port} ({label})")
    except Exception as e:
        print(f"  FAIL localhost:{port} ({label}): {e}")
        sys.exit(1)
PY

mkdir -p tv_snapshots/jajamaica

exec python3 -m hailo_apps.python.standalone_apps.tunnelvision.tunnelvision \
  --ingress 'rtsp://root:carwash@localhost:8121/axis-media/media.amp' \
  --egress  'rtsp://ai:ai123456@localhost:8150/Streaming/Channels/801/?transportmode=unicast' \
  --zones   experiments/tunnelvision/zones-jajamaica.json \
  --db      tunnelvision-jajamaica.db \
  --snapshot-dir tv_snapshots/jajamaica \
  --debug "$@"
