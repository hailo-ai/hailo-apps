# TunnelVision Edge Tracking + Visit Correlation System

**Technical Product Requirements Document (PRD)**

| Field | Value |
|---|---|
| Product | TunnelVision Edge Visit Tracking System |
| Version | 2.0 |
| Platform | Raspberry Pi 5 + Hailo AI HAT+ + RTSP Cameras + Rekor CarCheck + Local Edge Processing |

## Goal

Build a highly reliable, edge-first ingress/egress vehicle tracking system for car wash tunnels that:

- Tracks vehicles continuously from ingress until out-of-frame
- Associates ingress and egress captures to the same visit
- Handles multiple simultaneous vehicles in the tunnel
- Minimizes Rekor credit usage
- Runs continuously for 10+ hours/day
- Maintains low-latency edge processing on Raspberry Pi 5
- Stores only best evidence images (not full video)
- Measures tunnel dwell time and throughput
- Produces insurance-grade before/after evidence pairs

---

## 1. System Overview

TunnelVision consists of:

| Component | Responsibility |
|---|---|
| Ingress Camera Worker | Detect incoming vehicles and create visits |
| Egress Camera Worker | Detect exiting vehicles and close visits |
| Vehicle Tracker | Maintain persistent track IDs while vehicles remain visible |
| Capture Engine | Select best frame per observation |
| Rekor Service | ALPR + optional vehicle enrichment |
| Visit Correlator | Match ingress and egress events |
| Tunnel Queue Engine | Track multiple cars simultaneously in tunnel |
| Edge Database | Local persistence and buffering |
| Cloud Sync | Push finalized visit data to Tuxedo backend |

---

## 2. Core Architecture

```
Ingress Camera
  ↓
Motion Gate
  ↓
Hailo Vehicle Detection
  ↓
Multi-Object Tracking
  ↓
Zone-Aware State Machine
  ↓
Plate Detection
  ↓
Best Frame Selection
  ↓
Save Single JPEG
  ↓
Optional Rekor Call
  ↓
Create Visit
  ↓
Tunnel Queue Tracking
  ↓
Vehicle enters tunnel
Vehicle exits tunnel
  ↓
Egress Camera
  ↓
Motion Gate
  ↓
Tracking + Best Frame
  ↓
Match against active tunnel visits
  ↓
Attach egress image to visit
  ↓
Calculate tunnel time
  ↓
Complete visit
```

---

## 3. Key Design Principles

### 3.1 Continuous Tracking

Vehicles are continuously tracked until they leave frame.

Zones are **NOT** lifecycle boundaries.

Zones only determine:
- When to evaluate captures
- When to ignore new captures
- When to retire tracks

### 3.2 One Best Image Per Observation

Store:
- 1 ingress image
- 1 egress image

Do NOT store full video.

### 3.3 Rekor as Paid Enrichment Layer

Rekor is not the tracker.

Rekor only receives:
- High-quality still JPEGs
- One call max per ingress track
- Egress only if matching fails

### 3.4 Tunnel Queue Awareness

Multiple vehicles may exist simultaneously in tunnel.

The system must:
- Preserve ordering
- Track expected tunnel duration
- Match ingress → egress reliably

---

## 4. User Stories

### Operator

- As an operator, I want ingress and egress images tied to the same visit.
- As an operator, I want to see how long a car spent in tunnel.
- As an operator, I want evidence images for damage disputes.
- As an operator, I want reliable ALPR even when cars block each other.

### Site Owner

- As an owner, I want throughput metrics.
- As an owner, I want low API costs.
- As an owner, I want stable 10-hour runtime.
- As an owner, I want visit history tied to vehicles.

---

## 5. Functional Requirements

### 5.1 Ingress Processing

**Trigger**

System begins processing when:
- Motion detected in approach zone AND
- Vehicle detector confirms vehicle

**Steps**

**Step 1 — Create Track**

Assign: `track_id`

Persist:
- `first_seen_at`
- `bbox`
- `direction`
- `zone`

**Step 2 — Continuous Tracking**

Track persists while:
- vehicle visible OR
- temporarily occluded

Track should survive:
- partial blockage
- bumper overlap
- temporary plate obstruction

**Step 3 — Eligibility for Capture**

Vehicle becomes eligible when:
- vehicle inside capture zone AND
- vehicle fully visible AND
- plate visible AND
- track stable

**Step 4 — Best Frame Selection**

| Metric | Purpose |
|---|---|
| plate confidence | readable plate |
| blur score | avoid motion blur |
| plate width | sufficient OCR size |
| bbox stability | avoid partial vehicle |
| motion score | reject moving blur |
| plate blockage | reject overlapping cars |

**Step 5 — Save Best Image**

Persist: `best_ingress.jpg`

Store:
- image hash
- quality score
- bbox metadata

**Step 6 — Rekor Call**

Only if:
- quality score passes threshold AND
- vehicle not recently recognized

### 5.2 Egress Processing

Egress mirrors ingress but:
- no new visit creation
- no default Rekor call
- attempts visit correlation

**Egress Steps**

Step 1 — Detect exiting vehicle → create `egress_track_id`

Step 2 — Best image selection → persist `best_egress.jpg`

Step 3 — Visit Matching

System attempts to match `egress_track` → `active tunnel visit`

| Signal | Weight |
|---|---|
| plate match | highest |
| lane | high |
| tunnel ordering | high |
| elapsed tunnel time | medium |
| vehicle attributes | medium |
| color | low |
| bbox size | low |

### 5.3 Tunnel Queue Engine

**Problem**

Multiple cars can exist simultaneously:

```
Car A enters
Car B enters
Car C enters
Car A exits
Car B exits
Car C exits
```

Need reliable ordering.

**Queue Model**

Each lane maintains: `active_tunnel_queue`

Queue ordered by: `ingress_capture_time`

**Tunnel State**

Each active visit contains:

| Field | Purpose |
|---|---|
| ingress_time | tunnel start |
| expected_exit_window | estimated exit |
| queue_position | relative ordering |
| estimated_tunnel_duration | rolling average |
| active_in_tunnel | currently washing |

**Matching Logic**

When egress occurs — candidate visits:

```
all active tunnel visits
WHERE lane matches
AND not completed
```

**Ranking**

```
score =
    plate_match * 100
  + queue_order_match * 40
  + tunnel_time_similarity * 25
  + vehicle_similarity * 15
  + color_similarity * 5
```

Best candidate above threshold becomes match.

---

## 6. Technical Requirements

### 6.1 Hardware

| Component | Requirement |
|---|---|
| Compute | Raspberry Pi 5 8GB |
| Accelerator | Hailo AI HAT+ 26 TOPS preferred |
| Cameras | RTSP IP cameras |
| Runtime | 10+ hours continuous |
| Cooling | Active cooling mandatory |
| Storage | High-endurance SSD preferred |

### 6.2 Camera Configuration

| Setting | Value |
|---|---|
| Resolution | 1080p |
| Processing resolution | 640–720p |
| FPS | 8–12 FPS |
| Codec | H264 |
| Transport | RTSP TCP |

### 6.3 Runtime Targets

| Metric | Target |
|---|---|
| Runtime stability | 10+ hrs |
| CPU usage | <70% sustained |
| Memory usage | <6GB |
| Rekor calls | ≤1 per ingress vehicle |
| Capture latency | <500ms |
| Missed captures | <2% |
| Duplicate reads | <1% |

---

## 7. Hailo Edge Pipeline

Hailo pipeline stages:

```
vehicle_detection
→ vehicle_tracking
→ plate_detection
→ quality_estimation
→ best_frame_selection
→ optional_local_ocr
```

### 7.1 Recommended Models

| Stage | Model |
|---|---|
| vehicle detection | Hailo YOLO vehicle detector |
| tracking | ByteTrack / DeepSORT |
| plate detection | YOLOv8n LP detector |
| OCR | optional local OCR |
| quality scoring | custom local logic |

---

## 8. Rekor Integration

```python
import base64
import requests

with open(image_path, "rb") as image_file:
    img_base64 = base64.b64encode(image_file.read())

url = (
    "https://api.openalpr.com/v3/recognize_bytes"
    "?recognize_vehicle=0"
    "&country=us"
    f"&secret_key={REKOR_SECRET_KEY}"
)
response = requests.post(url, data=img_base64)
payload = response.json()
```

**Rekor Response Fields Used**

| Field | Use |
|---|---|
| results[].plate | canonical plate |
| results[].region | state |
| results[].confidence | confidence |
| candidates[] | fallback matching |
| vehicle.make | enrichment |
| vehicle.make_model | enrichment |
| vehicle.color | enrichment |
| credit_cost | tracking spend |

---

## 9. State Machine

```
NEW
↓
TRACKING
↓
PARTIAL_VISIBLE
↓
FULLY_VISIBLE
↓
PLATE_BLOCKED
↓
PLATE_CANDIDATE
↓
BEST_IMAGE_SELECTED
↓
REKOR_SENT
↓
PROCESSED_BUT_TRACKING
↓
IN_TUNNEL
↓
MATCHED_AT_EGRESS
↓
COMPLETED
```

---

## 10. Database Requirements

Must persist:
- visits
- tracks
- ingress image
- egress image
- Rekor request/response
- queue ordering
- tunnel timing
- evidence metadata

See ERD companion document (`ERD.md`).

---

## 11. Best Frame Algorithm

**Candidate selection**

```
quality_score =
    (plate_confidence * 30)
  + (blur_score * 25)
  + (plate_width_score * 20)
  + (bbox_stability * 15)
  + (roi_fit * 10)
  - motion_penalty
  - blockage_penalty
```

**Frame accepted if:**

- vehicle_fully_visible
- and plate_visible
- and not plate_blocked
- and blur_score >= threshold
- and plate_width_px >= min_width

---

## 12. Failure Handling

**Internet loss**

Continue locally:
- save image
- queue Rekor later

**Rekor timeout**

Retry async. Never block camera loop.

**Lost track**

Mark: `no_read`

**Plate blocked entire time**

Fallback:
- vehicle-only correlation
- queue ordering
- tunnel timing

---

## 13. Visit Correlation Rules

| Match Strategy | Confidence |
|---|---|
| exact plate | highest |
| candidate plate | high |
| queue ordering | medium |
| tunnel timing | medium |
| vehicle appearance | low-medium |
| manual review | fallback |

---

## 14. Tunnel Timing

Each visit tracks:
- `ingress_time`
- `egress_time`
- `tunnel_duration_seconds`

**Analytics**

| Metric | Purpose |
|---|---|
| avg tunnel duration | throughput |
| cars/hour | utilization |
| queue depth | congestion |
| stalled vehicle detection | operational alerts |

---

## 15. API Cost Controls

| Rule | Action |
|---|---|
| known recent plate | skip Rekor |
| egress matched | skip Rekor |
| low-quality frame | skip Rekor |
| monthly credits high | conservation mode |

---

## 16. Deployment Architecture

```
Ingress Worker
  ↓
SQLite Queue
  ↓
Rekor Async Worker
  ↓
Visit Correlator
  ↓
Cloud Sync

Egress Worker
  ↓
Visit Correlator
  ↓
Finalize Visit
```

---

## 17. MVP Scope

**Included**

- ingress tracking
- egress tracking
- tunnel timing
- best-frame selection
- Rekor ALPR
- visit matching
- image evidence
- local persistence

**Excluded**

- full video recording
- live streaming
- demographic inference
- advanced damage AI
- thermal cameras
- audio diagnostics

---

## 18. Success Metrics

| KPI | Goal |
|---|---|
| successful ingress captures | >95% |
| successful visit correlation | >95% |
| duplicate Rekor calls | <1% |
| runtime stability | 10+ hrs |
| avg tunnel timing accuracy | ±3 sec |
| Rekor credits per car | ≤1 |
| ingress-egress mismatch rate | <2% |

---

## 19. Final Technical Recommendation

The correct architecture is:

```
continuous tracking
+ zone-aware capture eligibility
+ single best-image evidence
+ async Rekor enrichment
+ queue-aware tunnel correlation
```

Not:

```
motion trigger → immediate ALPR → forget car
```

The system should think in terms of **persistent vehicle identities moving through a tunnel lifecycle** rather than isolated camera events.
