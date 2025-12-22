#!/usr/bin/env bash
# kill_first_hailo.sh — find the first /dev/hailo* and kill any process using it

set -uo pipefail

# Find all hailo devices, pick the first one
hailo_dev=$(ls /dev/hailo* 2>/dev/null | head -n1)
if [[ -z "$hailo_dev" ]]; then
    echo "❌ No /dev/hailo* device found."
    exit 1
fi

echo "🔍 Target device: $hailo_dev"

# Gather PIDs holding the device open
if command -v fuser &>/dev/null; then
    pids=$(fuser "$hailo_dev" 2>/dev/null | tr -d ':')
elif command -v lsof &>/dev/null; then
    pids=$(lsof -t "$hailo_dev" 2>/dev/null)
else
    echo "❌ Neither fuser nor lsof is installed." >&2
    exit 1
fi

if [[ -z "$pids" ]]; then
    echo "✅ No processes are using $hailo_dev."
    exit 0
fi

echo "⚠️  Found processes using $hailo_dev: $pids"
echo "➡️  Sending SIGTERM to $pids"
kill $pids

# Give them time to exit
sleep 2

# Force-kill any remaining
for pid in $pids; do
    if kill -0 $pid &>/dev/null; then
        echo "🚨 PID $pid still alive—sending SIGKILL"
        kill -KILL $pid
    fi
done

echo "✅ Done."
