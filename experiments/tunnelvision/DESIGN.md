# TunnelVision — Design Specification

**Date:** 2026-05-07  
**Status:** Approved for implementation  
**Source docs:** `experiments/tunnelvision/PRD.md`, `experiments/tunnelvision/ERD.md`  
**App location:** `hailo_apps/python/standalone_apps/tunnelvision/`

---

## 1. Problem Statement

The existing `carwash_lpr` app is a proof-of-concept: it reads plates via DeGirum cloud, votes across N frames, and matches sessions by time window. It has no vehicle detection, no persistent track IDs, no zone awareness, and single-car session matching that breaks under simultaneous traffic.

TunnelVision replaces this with a continuous-tracking, zone-aware, best-frame, Rekor-gated system that handles multiple simultaneous cars, produces insurance-grade before/after evidence pairs, and runs for 10+ hour shifts without drift.

---

## 2. Cameras

| Role | IP | Credentials | Main stream URL |
|---|---|---|---|
| Ingress (cam1) | 192.168.1.121 | bowtie / dieformalwear99! | `rtsp://bowtie:dieformalwear99!@192.168.1.121:554/media/live/1/1` |
| Egress (cam2) | 192.168.1.125 | admin / 123456 | `rtsp://admin:123456@192.168.1.125:554/media/live/1/1` |

Always single-quote the ingress URL in shell — `!` in the password triggers bash history expansion under double quotes.

---

## 3. Inference Stack

**Backend:** DeGirum local (`@local`) — same SDK as `carwash_lpr` but pointed at the on-device HailoRT server instead of cloud.

**Models used:**

| Stage | Model name (DeGirum zoo) | Input |
|---|---|---|
| Vehicle detection | TBD — query local zoo at implementation start; target COCO YOLO detecting car/truck/bus | full frame |
| Plate detection | `yolov8n_relu6_lp--640x640_quant_hailort_hailo8l_1` | vehicle crop |
| OCR | `yolov8n_relu6_lp_ocr--256x128_quant_hailort_hailo8l_1` | plate crop |

Each camera pipeline owns its own model instances. DeGirum models are not thread-safe for concurrent `.predict()` calls — no sharing between ingress and egress workers.

**PaddleOCR fallback is dropped.** Rekor CarCheck is the enrichment layer for low-confidence reads.

---

## 4. Architecture

### 4.1 File Structure

```
hailo_apps/python/standalone_apps/tunnelvision/
├── __init__.py
├── tunnelvision.py      # CLI entry point, thread wiring, signal handling
├── ingest.py            # RTSPIngest — copied verbatim from carwash_lpr
├── tracker.py           # DeGirum vehicle detector + IoU/SORT multi-object tracker
├── zone.py              # Polygon zone evaluation (approach/capture/ignore/exit)
├── state.py             # VehicleTrack dataclass + 12-state state machine
├── scorer.py            # Best-frame quality scoring (PRD §11 formula)
├── pipeline.py          # Per-camera frame loop — orchestrates all components
├── rekor.py             # Async Rekor CarCheck queue + HTTP client + response parser
├── correlator.py        # Tunnel FIFO queue + ingress-egress visit matching
└── db.py                # SQLite schema + all read/write operations
```

### 4.2 Threading Model

```
ingest thread (cam1)  →  frame_queue_ingress  →  pipeline thread (cam1) ─┐
ingest thread (cam2)  →  frame_queue_egress   →  pipeline thread (cam2) ─┤
                                                                           ├→ rekor_queue → rekor worker thread
                                                                           └→ db_queue    → db writer thread
```

- **2 ingest threads** — GStreamer mainloop per camera, exponential-backoff reconnect
- **2 pipeline threads** — full per-frame processing loop, one per camera
- **1 Rekor worker thread** — drains `rekor_queue`, never blocks camera loop
- **1 DB writer thread** — serializes all SQLite writes; readers use a read-only connection

### 4.3 Per-Frame Processing Loop (`pipeline.py`)

```
receive frame from frame_queue
  → motion_gate()                     # skip if no motion delta
  → vehicle_detector.predict(frame)   # DeGirum vehicle detection
  → tracker.update(detections)        # IoU association → persistent track IDs
  → for each active track:
      zone_evaluator.update(track)    # which zone is the vehicle in?
      state_machine.transition(track) # advance state based on zone + visibility
      if track.processed: continue
      if not eligible_for_capture(track): continue
      plate_det = plate_detector.predict(vehicle_crop)
      if not plate_det: continue
      if plate_blocked(plate_det, all_tracks): → state = PLATE_BLOCKED; continue
      score = scorer.score(frame, track, plate_det)
      track.add_candidate(frame, score)
      if track.has_stable_candidates(min_frames=3):
          best = scorer.select_best(track.candidates)
          db_queue.put(SaveBestImage(track, best))
          rekor_queue.put(RekorRequest(track, best))  # if credit policy allows
          track.processed = True
          track.state = PROCESSED_BUT_TRACKING
          if camera_role == ingress:
              db_queue.put(CreateVisit(track))     # visit created immediately after best image
              correlator.enqueue_ingress(track)    # enters tunnel FIFO queue
          if camera_role == egress:
              match = correlator.match_egress(track)
              db_queue.put(CloseVisit(match))
```

---

## 5. Component Design

### 5.1 `ingest.py` — RTSPIngest

Copied verbatim from `carwash_lpr/ingest.py`. GStreamer pipeline:

```
rtspsrc location="{url}" latency=200 protocols=tcp
  → decodebin → videoconvert → video/x-raw,format=BGR
  → appsink emit-signals=true max-buffers=1 drop=true sync=false
```

`frame_queue` is `maxsize=1` — always drops stale frames, keeps newest.

### 5.2 `tracker.py` — Vehicle Detector + IoU Tracker

Two responsibilities in one file:

**Vehicle detector:** Calls DeGirum vehicle detection model on the full frame. Returns bounding boxes + class scores for car/truck/bus detections.

**IoU/SORT tracker:** Associates detections across frames by intersection-over-union. Assigns persistent `track_id` integers. Handles:
- Re-association after brief occlusion (configurable `max_lost_frames`, default 10)
- New track creation when IoU match fails
- Track retirement when lost beyond `max_lost_frames`

ByteTrack/DeepSORT noted as future upgrade if multi-lane density increases. IoU/SORT is sufficient for single-lane throughput.

**Key invariant:** Track IDs never reuse within a session. Once a track is retired, its ID is gone.

### 5.3 `zone.py` — Zone Evaluator

**Schema adopted from `td-edge`** for forward compatibility with TD-Core integration. Single JSON file, normalized 0–1 polygon coordinates, per-zone `camera_id` filter (0 = ingress / cam1, 1 = egress / cam2):

```json
{
  "zones": [
    {
      "name": "ingress_gate",
      "zone_type": "trigger",
      "camera_id": 0,
      "action": "start_visit",
      "polygon": [[0.325, 0.2824], [0.1297, 0.988], ...]
    },
    {
      "name": "lpr_zone",
      "zone_type": "alpr",
      "camera_id": 0,
      "action": "capture_plate",
      "polygon": [[0.0911, 0.1157], ...]
    }
  ]
}
```

**Initial config:** `experiments/tunnelvision/zones.json` — copied from `td-edge/config/zones.example.json`. Cam0 polygons (`ingress_gate`, `lpr_zone`) are already tuned for the live ingress camera. Cam1 polygon (`egress_gate`) is a placeholder — tuned at first live egress test.

**PRD zone vocabulary maps onto td-edge `(zone_type, action)` pairs:**

| PRD concept | td-edge representation |
|---|---|
| `approach` (start paying attention) | `(trigger, start_visit)` |
| `capture` (eligible for best-frame scoring) | `(alpr, capture_plate)` |
| `exit` (retire track after car leaves) | `(trigger, end_visit)` |
| `ignore` (do not create new captures) | `(ignore, skip)` — new pair, no td-edge equivalent |

`zone.py` exposes the same `Zone.contains_point()` and `Zone.overlaps_bbox()` API as `td_edge/config/zones.py` (ray-casting point-in-polygon, four-corner + center bbox overlap test). The state machine queries zones by `(zone_type, action)` pair, not by raw zone name. A vehicle can be inside multiple zones simultaneously (e.g. inside both `lpr_zone` and `ingress_gate`); precedence is determined by the state machine, not by the zone evaluator.

### 5.4 `state.py` — VehicleTrack + State Machine

**`VehicleTrack` dataclass fields (in-memory, mirrors `vehicle_tracks` table):**

```python
track_id: int                    # from IoU tracker
db_id: str                       # UUID assigned on first DB write
camera_role: str                 # 'ingress' | 'egress'
state: TrackState                # current state enum
current_zone: str | None
current_bbox: list[float]
first_seen_at: datetime
last_seen_at: datetime
fully_visible: bool
processed: bool                  # True after best image selected
rekor_sent: bool
candidates: list[FrameCandidate] # in-memory, not persisted
best_image_path: str | None
lost_frames: int                 # frames since last detection
```

**State machine (12 states from PRD §9):**

```
NEW → TRACKING → PARTIAL_VISIBLE → FULLY_VISIBLE → PLATE_BLOCKED
                                                  → PLATE_CANDIDATE → BEST_IMAGE_SELECTED
                                                                     → REKOR_SENT
                                                                     → PROCESSED_BUT_TRACKING → IN_TUNNEL
                                                                                               → MATCHED_AT_EGRESS
                                                                                               → COMPLETED
Any state → LOST → EXITED | NO_READ
```

Transitions are pure functions: `transition(track, zone, visibility_flags, plate_detection) → new_state`. State transitions are logged to `track_state_events` table.

### 5.5 `scorer.py` — Best Frame Quality Scorer

**Quality score formula (exact from PRD §11):**

```
quality_score =
    (plate_confidence × 30)
  + (blur_score × 25)
  + (plate_width_score × 20)
  + (bbox_stability × 15)
  + (roi_fit × 10)
  - motion_penalty
  - blockage_penalty
```

**Frame accepted as candidate only if all gates pass:**
- `vehicle_confidence ≥ min_vehicle_confidence`
- `plate_confidence ≥ min_plate_confidence`
- `vehicle_inside_capture_zone = true`
- `vehicle_fully_visible = true`
- `plate_inside_vehicle = true`
- `plate_blocked = false`
- `plate_width_px ≥ min_plate_width_px`
- `blur_score ≥ min_blur_score`
- `motion_score ≤ max_motion_score`
- `|bbox_growth_rate| ≤ max_bbox_growth_rate`

**`select_best(candidates)`** picks the highest `quality_score` among the stable candidate window (min 3 frames).

Blur score: Laplacian variance on the plate crop. Motion score: frame-to-frame bbox centroid delta in pixels.

### 5.6 `rekor.py` — Async Rekor Queue

**Credit gate (from PRD §15 + ERD code):**

```python
def should_call_rekor(best_image, track, policy, known_recently=False):
    if best_image.quality_score < policy.min_quality_score: return False, "skip_low_quality"
    if track.camera_role == "egress":                       return False, "skip_egress"
    if monthly_ratio >= policy.emergency_threshold:
        if known_recently:                                  return False, "emergency_conservation"
        return True, "plate_only"
    if known_recently:                                      return False, "skip_known_plate"
    return True, "plate_only"   # recognize_vehicle=False by default
```

**HTTP call:** POST to `https://api.openalpr.com/v3/recognize_bytes` with base64-encoded JPEG. `REKOR_SECRET_KEY` from environment. 8-second timeout. Retry on network error (max 3 attempts, exponential backoff).

**Response parser** normalizes the full JSON into fields for `rekor_carcheck_responses` and `rekor_plate_results`. Raw JSON also stored in `raw_response` column.

### 5.7 `correlator.py` — Tunnel Queue + Visit Matching

**Tunnel queue:** Per-lane FIFO ordered by `ingress_capture_time`. All active visits (status `in_tunnel`) live here.

**Egress matching score (exact from PRD §5.3):**

```
score =
    plate_match × 100
  + queue_order_match × 40
  + tunnel_time_similarity × 25
  + vehicle_similarity × 15
  + color_similarity × 5
```

Candidate visits: all `in_tunnel` visits where `not completed`. Best candidate above threshold (configurable, default 60) becomes the match. Below threshold → `manual_review`.

**Fallback when plate is unread at egress:** queue ordering + tunnel timing alone. The first in-queue visit whose expected exit window covers the current time is the match candidate.

### 5.8 `db.py` — SQLite Storage

WAL mode + NORMAL synchronous (same as `carwash_lpr`). Single writer thread via `db_queue`. Separate read-only `sqlite3.connect(check_same_thread=False)` for correlator lookups.

**MVP schema (9 operational tables):**

```
visits
camera_observations
vehicle_tracks
track_state_events          ← state machine audit log
best_images
rekor_carcheck_requests
rekor_carcheck_responses
rekor_plate_results
rekor_plate_candidates      ← needed for fuzzy correlator matching
plates
visit_match_events
```

**Deferred to phase 2:** `sites`, `lanes`, `devices`, `cameras`, `camera_zones` (replaced by config file), `edge_model_runs`, `rekor_vehicle_results`, `rekor_vehicle_attributes`, `vehicles`, `vehicle_plates`, `image_evidence`, `rekor_credit_policies`.

UUIDs stored as `TEXT`. Timestamps stored as ISO 8601 `TEXT`. JSONB columns stored as `TEXT` (JSON-encoded). No foreign key enforcement in SQLite (integrity guaranteed by application layer).

### 5.9 `tunnelvision.py` — Entry Point

```bash
python3 -m hailo_apps.python.standalone_apps.tunnelvision.tunnelvision \
  --ingress 'rtsp://bowtie:dieformalwear99!@192.168.1.121:554/media/live/1/1' \
  --egress  'rtsp://admin:123456@192.168.1.125:554/media/live/1/1' \
  --zones         experiments/tunnelvision/zones.json \
  --db            tunnelvision.db \
  --snapshot-dir  snapshots/ \
  --tunnel-min    3 \
  --tunnel-max    8 \
  --monthly-budget 500 \
  --min-quality   80.0 \
  --debug
```

`REKOR_SECRET_KEY` consumed from environment, never a CLI flag.

---

## 6. Data Model (SQLite)

Key tables only — full SQL in `db.py`.

### `visits`
```
id TEXT PK, status TEXT, plate_id TEXT,
ingress_observation_id TEXT, egress_observation_id TEXT,
started_at TEXT, ingress_captured_at TEXT, egress_captured_at TEXT,
completed_at TEXT, tunnel_duration_sec INT, match_confidence REAL, notes TEXT
```
Statuses: `approaching → ingress_captured → in_tunnel → egress_captured → completed | failed | manual_review`

### `vehicle_tracks`
```
id TEXT PK, camera_observation_id TEXT, camera_role TEXT,
edge_track_id TEXT, state TEXT,
first_seen_at TEXT, last_seen_at TEXT,
current_zone TEXT, current_bbox TEXT, processed INT, rekor_sent INT,
best_image_id TEXT, exit_confirmed INT
UNIQUE(camera_role, edge_track_id)
```

### `best_images`
```
id TEXT PK, vehicle_track_id TEXT, observation_id TEXT,
image_path TEXT, image_sha256 TEXT, width INT, height INT,
quality_score REAL, blur_score REAL, motion_score REAL, plate_width_px INT,
car_fully_visible INT, plate_visible INT, plate_blocked INT,
vehicle_bbox TEXT, plate_bbox TEXT, selected_at TEXT
UNIQUE(image_sha256)
```

### `rekor_carcheck_requests`
```
id TEXT PK, visit_id TEXT, observation_id TEXT, vehicle_track_id TEXT,
best_image_id TEXT, image_path TEXT, image_sha256 TEXT,
request_status TEXT, credit_policy TEXT,
estimated_credit_cost INT, actual_credit_cost INT,
sent_at TEXT, completed_at TEXT, http_status INT, error_message TEXT
```

---

## 7. Configuration

### Zone Config File

`experiments/tunnelvision/zones.json` — td-edge schema, normalized 0–1 polygon coordinates. Single file covers both cameras via per-zone `camera_id` (0 = ingress, 1 = egress). See §5.3 for the schema and the PRD-vocabulary mapping.

CLI flag: `--zones experiments/tunnelvision/zones.json` (single path, not per-camera).

Coordinates are normalized — no resolution scaling needed at runtime; each zone evaluator just multiplies by the current frame's `(width, height)`.

### Credit Policy (CLI args, phase 2 promotes to DB)

| Flag | Default | Description |
|---|---|---|
| `--monthly-budget` | 500 | Monthly Rekor credit limit |
| `--min-quality` | 80.0 | Minimum quality score to call Rekor |
| `--min-plate-confidence` | 0.70 | Minimum plate confidence gate |
| `--emergency-threshold` | 0.85 | Monthly ratio to enter conservation mode |

---

## 8. Failure Handling

| Failure | Behavior |
|---|---|
| RTSP disconnect | `RTSPIngest` reconnects with exponential backoff (2s → 30s max) |
| Internet loss | Rekor requests stay in `rekor_queue` in memory; `request_status = queued` persisted to DB; retried when connectivity restores |
| Rekor timeout (8s) | Retry up to 3×, exponential backoff; after 3 failures mark `request_status = failed` |
| Track lost before best image | `state = NO_READ`; visit created with `no_plate` flag; correlator falls back to queue ordering |
| Plate blocked entire observation | `state = PLATE_BLOCKED` throughout; Rekor skipped; correlator uses queue + timing |
| Egress match below threshold | `visit.status = manual_review`; both images saved; operator reviews via SQL query |
| DB writer queue full | Drop oldest write event with warning log (DB queue `maxsize=500`) |

---

## 9. Testing Strategy

### Local (current phase)
- Both cameras reachable at 192.168.1.121 and 192.168.1.125 on local network
- Run app on Pi, verify via SQLite queries and saved JPEGs
- Rekor CarCheck verified via `rekor_carcheck_responses` table — no TD-Core needed
- Confirm DeGirum local vehicle detection model name before first run

### Staging (next phase)
- Real car wash cameras in production positions
- Run 2-hour observation windows, verify visit correlation rate
- Wire TD-Core API to push finalized visits to Tuxedo backend
- Compare `match_confidence` distribution against thresholds

### Key verification queries
```sql
-- Visit correlation rate
SELECT status, COUNT(*) FROM visits GROUP BY status;

-- Rekor spend
SELECT SUM(actual_credit_cost) FROM rekor_carcheck_requests
WHERE request_status = 'succeeded';

-- Tunnel timing accuracy
SELECT AVG(tunnel_duration_sec), MIN(tunnel_duration_sec), MAX(tunnel_duration_sec)
FROM visits WHERE status = 'completed';

-- State machine audit for a track
SELECT previous_state, next_state, reason, created_at
FROM track_state_events WHERE vehicle_track_id = ?
ORDER BY created_at;
```

---

## 10. Implementation Order

Following ERD build order (§ Build Order):

1. **`db.py`** — schema + write operations (no logic, just persistence)
2. **`ingest.py`** — copy from carwash_lpr, verify RTSP connects
3. **`tracker.py`** — vehicle detector (confirm model name) + IoU tracker
4. **`zone.py`** — polygon evaluator + zone config loader
5. **`state.py`** — VehicleTrack dataclass + state machine transitions
6. **`scorer.py`** — quality scoring formula + candidate selection
7. **`pipeline.py`** — wire all components into per-frame loop
8. **`rekor.py`** — async queue + HTTP + response parser + credit gate
9. **`correlator.py`** — tunnel FIFO queue + egress matching
10. **`tunnelvision.py`** — CLI entry point + thread wiring + signal handling

---

## 11. Open Items

| # | Item | When to resolve |
|---|---|---|
| 1 | Vehicle detection model name on DeGirum local zoo | First task in implementation |
| 2 | Egress polygon (`egress_gate`) tuning — cam0 polygons already tuned in `zones.json` | First live egress test |
| 3 | Add `(ignore, skip)` zone pair to schema if/when needed — not in current `zones.json` | Only if false-trigger areas surface in field |
| 4 | TD-Core API endpoint + auth for visit sync | Staging phase |
| 5 | `REKOR_SECRET_KEY` value confirmed in environment | Before first Rekor call |

---

## 12. What This Is Not

Per PRD §17, explicitly excluded from this implementation:

- Full video recording
- Live streaming
- Demographic inference
- Advanced damage AI
- Thermal cameras
- Multi-lane / multi-site support (phase 2)
- TD-Core cloud sync (staging phase)
- ByteTrack/DeepSORT (upgrade path, not MVP)
