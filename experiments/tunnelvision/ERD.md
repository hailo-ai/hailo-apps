# TunnelVision — Entity Relationship Document (ERD)

## Core Design Principle

```
Tracking   = identity layer
Zones      = decision layer
Best image = evidence layer
Rekor      = paid recognition layer
```

Zones should not decide whether a car still exists — the tracker does that. Zones only decide whether a tracked car is eligible for best-frame scoring.

---

## Entity Relationship Diagram

```
SITES ||--o{ LANES : has
LANES ||--o{ CAMERAS : has
LANES ||--o{ VISITS : processes
DEVICES ||--o{ CAMERAS : runs
CAMERAS ||--o{ CAMERA_ZONES : defines
CAMERAS ||--o{ CAMERA_OBSERVATIONS : captures
VISITS ||--o{ CAMERA_OBSERVATIONS : has
CAMERA_OBSERVATIONS ||--o{ VEHICLE_TRACKS : tracks
VEHICLE_TRACKS ||--o{ TRACK_STATE_EVENTS : logs
VEHICLE_TRACKS ||--o{ DETECTION_FRAMES : evaluates
VEHICLE_TRACKS ||--o{ EDGE_MODEL_RUNS : runs
VEHICLE_TRACKS ||--o{ BEST_IMAGES : selects
BEST_IMAGES ||--o{ IMAGE_EVIDENCE : stores
BEST_IMAGES ||--o{ REKOR_CARCHECK_REQUESTS : sends
REKOR_CARCHECK_REQUESTS ||--|| REKOR_CARCHECK_RESPONSES : receives
REKOR_CARCHECK_RESPONSES ||--o{ REKOR_PLATE_RESULTS : contains
REKOR_PLATE_RESULTS ||--o{ REKOR_PLATE_CANDIDATES : contains
REKOR_CARCHECK_RESPONSES ||--o{ REKOR_VEHICLE_RESULTS : contains
REKOR_VEHICLE_RESULTS ||--o{ REKOR_VEHICLE_ATTRIBUTES : contains
VISITS }o--|| VEHICLES : resolves_to
VEHICLES ||--o{ PLATES : has
PLATES ||--o{ VISITS : identifies
VISITS ||--o{ VISIT_MATCH_EVENTS : matches
```

---

## 1. Location, Hardware, and Camera Schema

```sql
CREATE TABLE sites (
  id UUID PRIMARY KEY,
  name TEXT NOT NULL,
  timezone TEXT NOT NULL DEFAULT 'America/New_York',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE lanes (
  id UUID PRIMARY KEY,
  site_id UUID NOT NULL REFERENCES sites(id),
  name TEXT NOT NULL,
  is_active BOOLEAN NOT NULL DEFAULT true
);

CREATE TABLE devices (
  id UUID PRIMARY KEY,
  site_id UUID NOT NULL REFERENCES sites(id),
  lane_id UUID REFERENCES lanes(id),
  name TEXT NOT NULL,
  hardware_type TEXT NOT NULL DEFAULT 'raspberry_pi_5',
  accelerator_type TEXT NOT NULL, -- hailo_ai_hat_plus_13tops / 26tops
  status TEXT NOT NULL DEFAULT 'online',
  last_seen_at TIMESTAMPTZ
);

CREATE TABLE cameras (
  id UUID PRIMARY KEY,
  device_id UUID NOT NULL REFERENCES devices(id),
  lane_id UUID NOT NULL REFERENCES lanes(id),
  camera_role TEXT NOT NULL CHECK (camera_role IN ('ingress', 'egress')),
  stream_url TEXT NOT NULL,
  resolution_width INT NOT NULL,
  resolution_height INT NOT NULL,
  fps_target INT NOT NULL DEFAULT 10,
  is_active BOOLEAN NOT NULL DEFAULT true,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

---

## 2. Zones — Decision Gates Only

```sql
CREATE TABLE camera_zones (
  id UUID PRIMARY KEY,
  camera_id UUID NOT NULL REFERENCES cameras(id),
  zone_type TEXT NOT NULL CHECK (
    zone_type IN ('approach', 'capture', 'ignore', 'exit')
  ),
  polygon JSONB NOT NULL,
  expected_direction TEXT NOT NULL, -- left_to_right / right_to_left / bottom_to_top / top_to_bottom
  is_active BOOLEAN NOT NULL DEFAULT true
);
```

**Recommended zone usage:**

| Zone | Purpose |
|---|---|
| `approach` | start paying attention |
| `capture` | evaluate best-frame candidates |
| `ignore` | do not create new captures |
| `exit` | retire track after car leaves |

---

## 3. Visit Lifecycle

```sql
CREATE TABLE visits (
  id UUID PRIMARY KEY,
  site_id UUID NOT NULL REFERENCES sites(id),
  lane_id UUID NOT NULL REFERENCES lanes(id),
  status TEXT NOT NULL CHECK (
    status IN (
      'approaching',
      'ingress_captured',
      'in_tunnel',
      'egress_captured',
      'completed',
      'failed',
      'manual_review'
    )
  ),
  vehicle_id UUID REFERENCES vehicles(id),
  plate_id UUID REFERENCES plates(id),
  ingress_observation_id UUID,
  egress_observation_id UUID,
  started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  ingress_captured_at TIMESTAMPTZ,
  egress_captured_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  match_confidence NUMERIC(5,2),
  notes TEXT
);
```

---

## 4. Camera Observations

A camera observation is a session around one tracked vehicle, not a raw stream.

```sql
CREATE TABLE camera_observations (
  id UUID PRIMARY KEY,
  visit_id UUID REFERENCES visits(id),
  camera_id UUID NOT NULL REFERENCES cameras(id),
  camera_role TEXT NOT NULL CHECK (camera_role IN ('ingress', 'egress')),
  started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  ended_at TIMESTAMPTZ,
  observation_status TEXT NOT NULL CHECK (
    observation_status IN (
      'tracking',
      'best_image_selected',
      'rekor_queued',
      'rekor_sent',
      'completed',
      'discarded',
      'no_read'
    )
  ),
  frame_count INT NOT NULL DEFAULT 0,
  best_image_id UUID,
  rejection_reason TEXT
);
```

---

## 5. Continuous Vehicle Tracking

This is the critical table.

```sql
CREATE TABLE vehicle_tracks (
  id UUID PRIMARY KEY,
  camera_observation_id UUID NOT NULL REFERENCES camera_observations(id),
  camera_id UUID NOT NULL REFERENCES cameras(id),
  edge_track_id TEXT NOT NULL,
  state TEXT NOT NULL CHECK (
    state IN (
      'new',
      'tracking',
      'partial_visible',
      'fully_visible',
      'plate_blocked',
      'plate_candidate',
      'best_image_selected',
      'rekor_sent',
      'processed_but_tracking',
      'lost',
      'exited',
      'no_read'
    )
  ),
  first_seen_at TIMESTAMPTZ NOT NULL,
  last_seen_at TIMESTAMPTZ NOT NULL,
  current_zone TEXT,
  current_bbox JSONB,
  bbox_history JSONB,
  processed BOOLEAN NOT NULL DEFAULT false,
  rekor_sent BOOLEAN NOT NULL DEFAULT false,
  best_image_id UUID,
  lost_after_ms INT,
  exit_confirmed BOOLEAN NOT NULL DEFAULT false,
  UNIQUE(camera_id, edge_track_id)
);
```

**Blocked plate handling:**

```
Car B is visible but blocked
→ state = plate_blocked
→ keep tracking
→ do not call Rekor
→ when plate becomes visible, promote to plate_candidate
```

---

## 6. Track State Event Log

```sql
CREATE TABLE track_state_events (
  id UUID PRIMARY KEY,
  vehicle_track_id UUID NOT NULL REFERENCES vehicle_tracks(id),
  previous_state TEXT,
  next_state TEXT NOT NULL,
  reason TEXT,
  zone_type TEXT,
  frame_index INT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

**Example state transitions:**

```
tracking → partial_visible
partial_visible → fully_visible
fully_visible → plate_blocked
plate_blocked → plate_candidate
plate_candidate → best_image_selected
best_image_selected → processed_but_tracking
processed_but_tracking → exited
```

---

## 7. Detection Frame Metadata

Do not save all frames. Save metadata for candidate frames only.

```sql
CREATE TABLE detection_frames (
  id UUID PRIMARY KEY,
  vehicle_track_id UUID NOT NULL REFERENCES vehicle_tracks(id),
  camera_observation_id UUID NOT NULL REFERENCES camera_observations(id),
  frame_ts TIMESTAMPTZ NOT NULL,
  frame_index INT NOT NULL,
  vehicle_confidence NUMERIC(6,3),
  plate_confidence NUMERIC(6,3),
  vehicle_bbox JSONB,
  plate_bbox JSONB,
  vehicle_inside_capture_zone BOOLEAN NOT NULL DEFAULT false,
  vehicle_fully_visible BOOLEAN NOT NULL DEFAULT false,
  plate_inside_vehicle BOOLEAN NOT NULL DEFAULT false,
  plate_blocked BOOLEAN NOT NULL DEFAULT false,
  plate_width_px INT,
  blur_score NUMERIC(8,2),
  motion_score NUMERIC(8,2),
  bbox_growth_rate NUMERIC(8,4),
  quality_score NUMERIC(8,2),
  selected_as_best BOOLEAN NOT NULL DEFAULT false
);
```

---

## 8. Edge Model Run Logging

```sql
CREATE TABLE edge_model_runs (
  id UUID PRIMARY KEY,
  vehicle_track_id UUID NOT NULL REFERENCES vehicle_tracks(id),
  camera_observation_id UUID NOT NULL REFERENCES camera_observations(id),
  device_id UUID NOT NULL REFERENCES devices(id),
  model_stage TEXT NOT NULL CHECK (
    model_stage IN (
      'motion_gate',
      'vehicle_detection',
      'vehicle_tracking',
      'plate_detection',
      'plate_ocr_local',
      'quality_estimation',
      'best_frame_selection'
    )
  ),
  model_name TEXT NOT NULL,
  model_version TEXT,
  accelerator TEXT NOT NULL DEFAULT 'hailo',
  input_type TEXT NOT NULL, -- full_frame / vehicle_crop / plate_crop
  output JSONB NOT NULL,
  latency_ms INT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

**Recommended local model names:**

| Stage | Model |
|---|---|
| vehicle_detection | Hailo vehicle_detection / YOLO vehicle model |
| plate_detection | YOLOv8n plate detector compiled for Hailo |
| plate_ocr_local | YOLOv8n OCR or LPRNet-style model compiled for Hailo |
| quality_estimation | local blur, motion, plate-size, ROI-fit scoring |

---

## 9. Best Image Storage

One best image per observation.

```sql
CREATE TABLE best_images (
  id UUID PRIMARY KEY,
  vehicle_track_id UUID NOT NULL REFERENCES vehicle_tracks(id),
  observation_id UUID NOT NULL REFERENCES camera_observations(id),
  image_path TEXT NOT NULL,
  image_sha256 TEXT NOT NULL,
  width INT NOT NULL,
  height INT NOT NULL,
  selected_frame_id UUID REFERENCES detection_frames(id),
  selected_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  vehicle_bbox JSONB,
  plate_bbox JSONB,
  quality_score NUMERIC(8,2),
  blur_score NUMERIC(8,2),
  motion_score NUMERIC(8,2),
  plate_width_px INT,
  car_fully_visible BOOLEAN NOT NULL,
  plate_visible BOOLEAN NOT NULL,
  plate_blocked BOOLEAN NOT NULL DEFAULT false,
  UNIQUE(image_sha256)
);

CREATE TABLE image_evidence (
  id UUID PRIMARY KEY,
  visit_id UUID NOT NULL REFERENCES visits(id),
  best_image_id UUID NOT NULL REFERENCES best_images(id),
  evidence_type TEXT NOT NULL CHECK (
    evidence_type IN ('ingress_best', 'egress_best')
  ),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

---

## 10. Rekor Request Persistence

```sql
CREATE TABLE rekor_carcheck_requests (
  id UUID PRIMARY KEY,
  visit_id UUID REFERENCES visits(id),
  observation_id UUID NOT NULL REFERENCES camera_observations(id),
  vehicle_track_id UUID NOT NULL REFERENCES vehicle_tracks(id),
  best_image_id UUID NOT NULL REFERENCES best_images(id),
  endpoint TEXT NOT NULL DEFAULT '/v3/recognize_bytes',
  api_url TEXT NOT NULL DEFAULT 'https://api.openalpr.com/v3/recognize_bytes',
  country TEXT NOT NULL DEFAULT 'us',
  recognize_vehicle BOOLEAN NOT NULL DEFAULT false,
  return_image BOOLEAN NOT NULL DEFAULT false,
  topn INT NOT NULL DEFAULT 10,
  secret_key_ref TEXT NOT NULL,
  image_path TEXT NOT NULL,
  image_sha256 TEXT NOT NULL,
  image_base64_size_bytes INT,
  request_status TEXT NOT NULL CHECK (
    request_status IN ('queued', 'sent', 'succeeded', 'failed', 'skipped')
  ),
  credit_policy TEXT NOT NULL CHECK (
    credit_policy IN (
      'plate_only',
      'vehicle_enrichment',
      'skip_known_plate',
      'skip_low_quality',
      'skip_egress_matched',
      'emergency_credit_conservation'
    )
  ),
  estimated_credit_cost INT,
  actual_credit_cost INT,
  sent_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  http_status INT,
  error_message TEXT
);
```

---

## 11. Rekor Response Persistence

```sql
CREATE TABLE rekor_carcheck_responses (
  id UUID PRIMARY KEY,
  request_id UUID NOT NULL REFERENCES rekor_carcheck_requests(id),
  data_type TEXT,
  epoch_time BIGINT,
  img_width INT,
  img_height INT,
  error BOOLEAN,
  version INT,
  uuid TEXT,
  credit_cost INT,
  credits_monthly_used INT,
  credits_monthly_total INT,
  processing_time_plates_ms NUMERIC,
  processing_time_vehicles_ms NUMERIC,
  processing_time_total_ms NUMERIC,
  regions_of_interest JSONB,
  raw_response JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

---

## 12. Rekor Plate Result Tables

```sql
CREATE TABLE rekor_plate_results (
  id UUID PRIMARY KEY,
  response_id UUID NOT NULL REFERENCES rekor_carcheck_responses(id),
  plate_index INT,
  requested_topn INT,
  plate TEXT NOT NULL,
  normalized_plate TEXT NOT NULL,
  region TEXT,
  confidence NUMERIC(8,4),
  region_confidence NUMERIC(8,4),
  matches_template BOOLEAN,
  coordinates JSONB,
  vehicle_region JSONB,
  processing_time_ms NUMERIC,
  vehicle_detected BOOLEAN,
  selected BOOLEAN NOT NULL DEFAULT false
);

CREATE TABLE rekor_plate_candidates (
  id UUID PRIMARY KEY,
  plate_result_id UUID NOT NULL REFERENCES rekor_plate_results(id),
  rank INT NOT NULL,
  plate TEXT NOT NULL,
  normalized_plate TEXT NOT NULL,
  confidence NUMERIC(8,4),
  matches_template BOOLEAN
);
```

---

## 13. Rekor Vehicle Result Tables

```sql
CREATE TABLE rekor_vehicle_results (
  id UUID PRIMARY KEY,
  response_id UUID NOT NULL REFERENCES rekor_carcheck_responses(id),
  plate_result_id UUID REFERENCES rekor_plate_results(id),
  vehicle_region JSONB,
  top_make TEXT,
  top_make_confidence NUMERIC(8,4),
  top_make_model TEXT,
  top_make_model_confidence NUMERIC(8,4),
  top_year_range TEXT,
  top_year_confidence NUMERIC(8,4),
  top_color TEXT,
  top_color_confidence NUMERIC(8,4),
  top_orientation TEXT,
  top_orientation_confidence NUMERIC(8,4),
  top_body_type TEXT,
  top_body_type_confidence NUMERIC(8,4),
  raw_vehicle JSONB,
  selected BOOLEAN NOT NULL DEFAULT false
);

CREATE TABLE rekor_vehicle_attributes (
  id UUID PRIMARY KEY,
  vehicle_result_id UUID NOT NULL REFERENCES rekor_vehicle_results(id),
  attribute_type TEXT NOT NULL CHECK (
    attribute_type IN ('make', 'make_model', 'year', 'color', 'orientation', 'body_type')
  ),
  rank INT NOT NULL,
  name TEXT NOT NULL,
  confidence NUMERIC(8,4)
);
```

---

## 14. Resolved Vehicle and Plate Tables

```sql
CREATE TABLE vehicles (
  id UUID PRIMARY KEY,
  make TEXT,
  model TEXT,
  year_range TEXT,
  color TEXT,
  body_type TEXT,
  orientation TEXT,
  source TEXT NOT NULL DEFAULT 'rekor_carcheck',
  confidence NUMERIC(8,4),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE plates (
  id UUID PRIMARY KEY,
  plate_number TEXT NOT NULL,
  normalized_plate TEXT NOT NULL,
  region TEXT,
  country TEXT NOT NULL DEFAULT 'us',
  confidence NUMERIC(8,4),
  source TEXT NOT NULL DEFAULT 'rekor_carcheck',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(normalized_plate, region, country)
);

CREATE TABLE vehicle_plates (
  id UUID PRIMARY KEY,
  vehicle_id UUID NOT NULL REFERENCES vehicles(id),
  plate_id UUID NOT NULL REFERENCES plates(id),
  first_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(vehicle_id, plate_id)
);
```

---

## 15. Visit Matching

```sql
CREATE TABLE visit_match_events (
  id UUID PRIMARY KEY,
  ingress_visit_id UUID NOT NULL REFERENCES visits(id),
  egress_observation_id UUID NOT NULL REFERENCES camera_observations(id),
  match_method TEXT NOT NULL CHECK (
    match_method IN (
      'plate_exact',
      'plate_fuzzy',
      'lane_time_window',
      'track_sequence',
      'manual'
    )
  ),
  match_confidence NUMERIC(5,2) NOT NULL,
  matched_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  notes TEXT
);
```

---

## 16. Credit Policy

```sql
CREATE TABLE rekor_credit_policies (
  id UUID PRIMARY KEY,
  site_id UUID REFERENCES sites(id),
  monthly_credit_budget INT NOT NULL,
  credits_used_this_month INT NOT NULL DEFAULT 0,
  default_recognize_vehicle BOOLEAN NOT NULL DEFAULT false,
  call_rekor_on_ingress BOOLEAN NOT NULL DEFAULT true,
  call_rekor_on_egress BOOLEAN NOT NULL DEFAULT false,
  skip_if_known_plate_within_days INT NOT NULL DEFAULT 30,
  enrich_vehicle_only_for_new_plate BOOLEAN NOT NULL DEFAULT true,
  min_quality_score_to_call NUMERIC(8,2) NOT NULL DEFAULT 80,
  min_plate_confidence_to_call NUMERIC(6,3) NOT NULL DEFAULT 0.70,
  emergency_conservation_threshold NUMERIC(4,2) NOT NULL DEFAULT 0.85
);
```

---

## Code Specifications

### Rekor Python Call

```python
import base64
import json
import os
import requests

REKOR_SECRET_KEY = os.environ["REKOR_SECRET_KEY"]
REKOR_BASE_URL = "https://api.openalpr.com/v3/recognize_bytes"

def call_rekor_carcheck(
    image_path: str,
    *,
    recognize_vehicle: bool = False,
    country: str = "us",
    timeout_seconds: int = 8,
) -> dict:
    with open(image_path, "rb") as image_file:
        img_base64 = base64.b64encode(image_file.read())
    url = (
        f"{REKOR_BASE_URL}"
        f"?recognize_vehicle={1 if recognize_vehicle else 0}"
        f"&country={country}"
        f"&secret_key={REKOR_SECRET_KEY}"
    )
    response = requests.post(url, data=img_base64, timeout=timeout_seconds)
    response.raise_for_status()
    return response.json()
```

### Best-Frame Tracker Loop

```python
def process_camera_frame(frame, camera, tracker, credit_policy):
    if not motion_gate(frame, camera.zones):
        return
    vehicle_detections = run_hailo_vehicle_detector(frame)
    tracks = tracker.update(vehicle_detections)
    for track in tracks:
        if track.state in ("lost", "exited"):
            continue
        update_track_zone(track, camera.zones)
        update_track_visibility(track, frame)
        if track.processed:
            track.state = "processed_but_tracking"
            continue
        if not track_is_eligible_for_capture(track):
            continue
        vehicle_crop = crop(frame, track.current_bbox)
        plate_detection = run_hailo_plate_detector(vehicle_crop)
        if not plate_detection:
            track.state = "fully_visible" if track.fully_visible else "partial_visible"
            continue
        if plate_is_blocked(plate_detection, tracks):
            track.state = "plate_blocked"
            continue
        frame_score = score_candidate_frame(
            frame=frame,
            track=track,
            plate_detection=plate_detection,
        )
        persist_detection_frame(track, frame_score)
        if frame_score.quality_score >= credit_policy.min_quality_score_to_call:
            track.add_candidate(frame, frame_score)
            track.state = "plate_candidate"
        if track.has_stable_candidates(min_frames=3):
            best_image = save_best_candidate_jpeg(track)
            mark_track_best_image_selected(track, best_image)
            enqueue_rekor_if_needed(track, best_image, credit_policy)
            track.processed = True
            track.state = "processed_but_tracking"
```

### Capture Eligibility

```python
def track_is_eligible_for_capture(track) -> bool:
    return (
        track.current_zone == "capture"
        and track.fully_visible
        and not track.processed
        and not track.rekor_sent
        and track.state not in ("best_image_selected", "processed_but_tracking")
    )
```

### Frame Quality Scoring

```python
def should_select_frame(frame_metrics, rule) -> bool:
    return (
        frame_metrics.vehicle_confidence >= rule.min_vehicle_confidence
        and frame_metrics.plate_confidence >= rule.min_plate_confidence
        and frame_metrics.vehicle_inside_capture_zone
        and frame_metrics.vehicle_fully_visible
        and frame_metrics.plate_inside_vehicle
        and not frame_metrics.plate_blocked
        and frame_metrics.plate_width_px >= rule.min_plate_width_px
        and frame_metrics.blur_score >= rule.min_blur_score
        and frame_metrics.motion_score <= rule.max_motion_score
        and abs(frame_metrics.bbox_growth_rate) <= rule.max_bbox_growth_rate
    )
```

### Rekor Credit Gate

```python
def should_call_rekor(best_image, track, credit_policy, known_plate_recently_seen=False):
    if best_image.quality_score < credit_policy.min_quality_score_to_call:
        return False, False, "skip_low_quality"
    if track.camera_role == "egress" and not credit_policy.call_rekor_on_egress:
        return False, False, "skip_egress_matched"
    monthly_ratio = (
        credit_policy.credits_used_this_month / credit_policy.monthly_credit_budget
        if credit_policy.monthly_credit_budget else 0
    )
    if monthly_ratio >= credit_policy.emergency_conservation_threshold:
        if known_plate_recently_seen:
            return False, False, "emergency_credit_conservation"
        return True, False, "plate_only"
    if known_plate_recently_seen:
        return False, False, "skip_known_plate"
    recognize_vehicle = credit_policy.default_recognize_vehicle
    return True, recognize_vehicle, (
        "vehicle_enrichment" if recognize_vehicle else "plate_only"
    )
```

### Rekor Response Parser

```python
def parse_rekor_response(payload: dict) -> dict:
    first_result = (payload.get("results") or [None])[0]
    first_vehicle = first_result.get("vehicle") if first_result else None

    def top_value(vehicle: dict, key: str):
        values = (vehicle or {}).get(key) or []
        if not values:
            return None, None
        return values[0].get("name"), values[0].get("confidence")

    make, make_conf = top_value(first_vehicle, "make")
    make_model, make_model_conf = top_value(first_vehicle, "make_model")
    color, color_conf = top_value(first_vehicle, "color")
    year, year_conf = top_value(first_vehicle, "year")
    orientation, orientation_conf = top_value(first_vehicle, "orientation")
    body_type, body_type_conf = top_value(first_vehicle, "body_type")

    return {
        "data_type": payload.get("data_type"),
        "epoch_time": payload.get("epoch_time"),
        "img_width": payload.get("img_width"),
        "img_height": payload.get("img_height"),
        "error": payload.get("error"),
        "version": payload.get("version"),
        "credit_cost": payload.get("credit_cost"),
        "credits_monthly_used": payload.get("credits_monthly_used"),
        "credits_monthly_total": payload.get("credits_monthly_total"),
        "processing_time": payload.get("processing_time"),
        "regions_of_interest": payload.get("regions_of_interest"),
        "plate": first_result.get("plate") if first_result else None,
        "region": first_result.get("region") if first_result else None,
        "plate_confidence": first_result.get("confidence") if first_result else None,
        "region_confidence": first_result.get("region_confidence") if first_result else None,
        "matches_template": first_result.get("matches_template") if first_result else None,
        "coordinates": first_result.get("coordinates") if first_result else None,
        "candidates": first_result.get("candidates") if first_result else [],
        "vehicle_detected": first_result.get("vehicle_detected") if first_result else None,
        "make": make,
        "make_confidence": make_conf,
        "make_model": make_model,
        "make_model_confidence": make_model_conf,
        "color": color,
        "color_confidence": color_conf,
        "year_range": year,
        "year_confidence": year_conf,
        "orientation": orientation,
        "orientation_confidence": orientation_conf,
        "body_type": body_type,
        "body_type_confidence": body_type_conf,
    }
```

---

## Build Order

1. Implement `camera_zones` — approach, capture, ignore, exit
2. Implement `vehicle_tracks` — keep tracking every car until lost/exited
3. Implement `best_images` — save one JPEG per ingress/egress observation
4. Implement Rekor async queue — camera loop must never wait on API response
5. Persist Rekor full JSON — store normalized fields plus raw response
6. Default `recognize_vehicle=false` — use `recognize_vehicle=true` only for new vehicles or enrichment moments

---

## Bottom Line

The correct ERD is **track-first, zone-aware, image-only, Rekor-gated**.

That lets you handle blocked plates naturally, run 10 hours/day, keep Rekor calls near one per real car, and build a clean audit trail for Tuxedo Drive.
