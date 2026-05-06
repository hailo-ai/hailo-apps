# Car Wash License Plate Recognition

Two-camera RTSP → Hailo-8L license plate detection and session tracking.
Runs unattended for 10-hour shifts. Stores reads to SQLite and JPEG snapshots.

## Requirements

- Hailo-8L device
- HEF files:
  - `yolov8n_relu6_lp--640x640_quant_hailort_hailo8l_1.hef`
  - `yolov8n_relu6_lp_ocr--256x128_quant_hailort_hailo8l_1.hef`
- Two RTSP IP cameras (ingress + egress)

## Quick Start

```bash
source setup_env.sh
python3 -m hailo_apps.python.standalone_apps.carwash_lpr.carwash_lpr \
    --ingress rtsp://192.168.1.10/stream \
    --egress  rtsp://192.168.1.11/stream \
    --detector-hef /path/to/yolov8n_relu6_lp--640x640_quant_hailort_hailo8l_1.hef \
    --ocr-hef      /path/to/yolov8n_relu6_lp_ocr--256x128_quant_hailort_hailo8l_1.hef \
    --db           /data/plates.db \
    --snapshot-dir /data/snapshots
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--ingress` | required | RTSP URL for ingress camera |
| `--egress` | required | RTSP URL for egress camera |
| `--detector-hef` | required | Path to plate detector HEF |
| `--ocr-hef` | required | Path to plate OCR HEF |
| `--db` | `plates.db` | SQLite database path |
| `--snapshot-dir` | `snapshots/` | JPEG snapshot directory |
| `--tunnel-min` | `3` | Min tunnel time in minutes |
| `--tunnel-max` | `8` | Max tunnel time in minutes |
| `--debug` | false | Verbose logging |

## Database Schema

**`plate_reads`** — every confirmed plate detection  
**`sessions`** — ingress/egress lifecycle per car

Session statuses:
- `open` — car entered, not yet exited
- `confirmed` — egress plate matched ingress
- `needs_review` — ingress and egress plates differ
- `egress_only` — egress read with no matching ingress

## Querying Results

```sql
-- All sessions from today
SELECT * FROM sessions WHERE entry_time >= date('now');

-- Plates that need review
SELECT s.plate_string, s.egress_plate, s.entry_time
FROM sessions s WHERE s.status = 'needs_review';

-- Confirmed sessions for POS lookup
SELECT plate_string, entry_time FROM sessions
WHERE status = 'confirmed' ORDER BY entry_time DESC;
```
