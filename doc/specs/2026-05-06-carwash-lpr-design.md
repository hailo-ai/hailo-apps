# Car Wash License Plate Recognition — Design Spec

**Date:** 2026-05-06
**Branch:** edge-degirum
**Status:** Approved

---

## Overview

A 10-hour unattended license plate recognition system for a car wash tunnel. Two RTSP IP cameras — one at the ingress, one at the egress — feed a Python standalone app running on a Hailo-8L device. The ingress read is the authoritative plate event (to be consumed by a POS system separately via SQLite). The egress read confirms the car's exit and matches back to its ingress session.

**Non-goals (this build):**
- POS API integration (reads SQLite directly)
- State/region detection (future phase)
- Video clip recording (snapshot only)

---

## Hardware & Models

| Item | Value |
|---|---|
| Hailo device | Hailo-8L |
| Plate detector | `yolov8n_relu6_lp--640x640_quant_hailort_hailo8l_1` |
| Plate OCR | `yolov8n_relu6_lp_ocr--256x128_quant_hailort_hailo8l_1` |
| Cameras | 2× RTSP IP cameras |
| Run duration | 10 hours continuous |

---

## Architecture

### Data Flow

```
Camera 1 — Ingress (RTSP)               Camera 2 — Egress (RTSP)
        │                                       │
GStreamer rtspsrc + reconnect           GStreamer rtspsrc + reconnect
        │                                       │
     appsink                                 appsink
        │                                       │
  Queue(maxsize=1)                       Queue(maxsize=1)
  put_nowait → drop if full             put_nowait → drop if full
        │                                       │
Inference Worker (ingress)             Inference Worker (egress)
        └──────────────┬──────────────────────┘
                       │
             Shared Hailo VDevice
             (SHARED_VDEVICE_GROUP_ID)
                       │
              Plate Detector 640×640
                       │
                  Crop bboxes
                       │
               OCR model 256×128
                       │
              Sort detections L→R
                       │
              TemporalVoter (per camera)
                       │
           Low confidence? → PaddleOCR fallback
                       │
              SessionTracker
                       │
              ┌─────────┴─────────┐
           SQLite              JPEG snapshot
           plates.db           snapshots/
```

### No-Latency Guarantee

`queue.Queue(maxsize=1)` with `put_nowait()`. If the inference worker is busy, the ingest thread catches `queue.Full`, discards the old frame, and the new frame takes its place. The worker always processes the freshest available frame — backlog is structurally impossible.

---

## Components

### `ingest.py` — RTSPIngest

One instance per camera. Owns a GStreamer pipeline:
```
rtspsrc location=<url> → decodebin → videoconvert → video/x-raw,format=RGB → appsink
```

- `appsink` callback: `queue.put_nowait(frame)`, silently drops if full
- Bus watcher: on `EOS` or `ERROR`, set pipeline to `NULL`, reconnect with exponential backoff (2s → 4s → 8s, capped at 30s)
- Exposes: `start()`, `stop()`, `frame_queue`

### `inference.py` — PlateInference

Loaded once at startup, shared across both workers via `SHARED_VDEVICE_GROUP_ID`.

Per frame:
1. Run plate detector (640×640) → list of bounding boxes
2. Crop each bbox from original frame
3. Run OCR model (256×128) on each crop
4. Sort detections left-to-right by x-coordinate
5. Assemble characters into plate string
6. Return: `list[PlateRead(plate_string, confidence, crop_frame, full_frame)]`

### `voter.py` — TemporalVoter

Per-camera sliding window over recent reads.

| Parameter | Ingress | Egress |
|---|---|---|
| Window size | 15 | 10 |
| Emit threshold | 10/15 | 6/10 |
| PaddleOCR fallback threshold | < 0.70 | < 0.60 |
| Dedup window | 45 seconds | 30 seconds |

Emits a `VotedPlate(plate_string, confidence, best_frame)` when threshold is met. Suppresses re-emission of the same plate within the dedup window.

### `session.py` — SessionTracker

Manages the ingress → egress lifecycle.

**On ingress confirmation:**
- Write `plate_reads` row
- Save snapshot
- Create `sessions` row with `status='open'`, `entry_time=now`

**On egress confirmation:**
- Look up `sessions` where `status='open'` and `entry_time` is 3–8 minutes ago
- **Match** (strings equal): update `status='confirmed'`, set `exit_time`, `duration_sec`
- **Mismatch** (strings differ): update `status='needs_review'`, store `egress_plate`
- **No open session**: write `plate_reads` row with `status='egress_only'`

Lookup window default: 3 min (min) to 8 min (max). Configurable via CLI.

### `storage.py` — ResultStorage

- SQLite in WAL mode (`PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL`)
- Single write thread with an internal queue to avoid lock contention
- Saves full frame as JPEG (not crop only) — preserves context for `needs_review` manual inspection
- Snapshot filename: `snapshots/cam_{ingress|egress}_{YYYYMMDD_HHMMSS}_{plate}.jpg`
- Snapshot directory pruned at startup if > 10,000 files (oldest deleted first)

### `fallback.py` — PaddleOCR fallback

Thin wrapper around the existing `hailo_apps.python.standalone_apps.paddle_ocr` module. Called only when `TemporalVoter` confidence < threshold after the window fills. Returns `(plate_string, confidence)` or `None` on failure.

### `carwash_lpr.py` — Entry point

```
usage: carwash_lpr.py --ingress <rtsp://...> --egress <rtsp://...>
                       [--db plates.db]
                       [--snapshot-dir snapshots/]
                       [--arch hailo8l]
                       [--tunnel-min 3] [--tunnel-max 8]
                       [--debug]
```

- Starts both `RTSPIngest` threads
- Starts both inference worker threads
- Handles `SIGINT`/`SIGTERM` via `threading.Event(running)`
- `finally`: drain queues → close HailoInfer models → release VDevice → close SQLite

---

## SQLite Schema

```sql
CREATE TABLE plate_reads (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id     TEXT NOT NULL,   -- 'ingress' | 'egress'
    timestamp     TEXT NOT NULL,   -- ISO 8601
    plate_string  TEXT NOT NULL,
    confidence    REAL NOT NULL,
    snapshot_path TEXT,
    source        TEXT NOT NULL    -- 'hailo_ocr' | 'paddleocr'
);

CREATE TABLE sessions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ingress_read_id   INTEGER REFERENCES plate_reads(id),
    egress_read_id    INTEGER REFERENCES plate_reads(id),
    plate_string      TEXT NOT NULL,  -- ingress plate (authoritative)
    egress_plate      TEXT,           -- egress read (populated on mismatch)
    entry_time        TEXT,
    exit_time         TEXT,
    duration_sec      INTEGER,
    status            TEXT NOT NULL   -- 'open'|'confirmed'|'needs_review'|'egress_only'
);

CREATE INDEX idx_sessions_status_entry ON sessions(status, entry_time);
```

---

## 10-Hour Stability

### RTSP Disconnect
GStreamer `rtspsrc` reconnects automatically. Ingest thread additionally watches the bus; on `ERROR` it restarts the pipeline with exponential backoff (2s → 4s → 8s → 30s cap). Inference worker uses `queue.get(timeout=5)` and loops on `TimeoutError` — no crash on empty queue.

### Hailo Device Error
`HailoInfer` exceptions are caught in the inference worker. One reload attempt is made. If reload fails, the worker sets a `degraded` flag and logs `CRITICAL`. The other camera's worker continues independently.

### Memory
- Snapshot directory pruned at startup if > 10,000 files
- SQLite WAL mode prevents write blocking reads
- `deque(maxlen=N)` voters have bounded memory by design

### Signal Handling
`SIGINT`/`SIGTERM` sets shared `threading.Event`. Each thread checks `running.is_set()` in its loop. Main thread `finally` block handles full teardown in order: queues → models → VDevice → SQLite.

---

## File Structure

```
hailo_apps/python/standalone_apps/carwash_lpr/
    __init__.py
    carwash_lpr.py      # CLI entry point, orchestration, signal handling
    ingest.py           # GStreamer RTSP → appsink → Queue(1) + reconnect
    inference.py        # Hailo plate detector + crop + OCR
    voter.py            # TemporalVoter: sliding window per camera
    session.py          # SessionTracker: open/confirm/mismatch sessions
    storage.py          # SQLite (WAL) writes + JPEG snapshot saves
    fallback.py         # PaddleOCR wrapper
    README.md
```

---

## Out of Scope (Future Phases)

- State/region detection from plate crop
- POS API integration (webhook or shared queue)
- Web dashboard for reviewing `needs_review` sessions
- Multi-bay support (> 2 cameras)
