# hailooverlay_community

A GStreamer overlay element for drawing post-processing results (bounding boxes,
labels, landmarks, tracking IDs, masks, sprites) on video frames carrying Hailo
ROI metadata. It is a locally modifiable fork of the upstream `hailooverlay`
element, designed to run on constrained hardware (RPi + Hailo-8L) with
zero-cost-when-disabled features.

## Supported Video Formats

RGB, RGBA, YUY2, NV12

Note: Sprite/stamp rendering only works on RGB and RGBA formats. On YUY2/NV12
the sprite is silently skipped.

## Building

```bash
source setup_env.sh
hailo-compile-postprocess
```

The shared library is installed to `/usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgsthailooverlay_community.so`.

Requires `libyaml-cpp-dev` for YAML config parsing.

Verify the element is available:

```bash
gst-inspect-1.0 hailooverlay_community
```

## Basic Usage

Drop-in replacement anywhere you would use `hailooverlay`:

```
... ! hailooverlay_community ! ...
```

With no properties set, the output is identical to the original element.

## Properties

### Drawing Style

| Property | Type | Default | Mutable | Description |
|---|---|---|---|---|
| `line-thickness` | int | 1 | READY | Thickness of bounding box lines. |
| `font-thickness` | int | 1 | READY | Thickness of rendered text. |
| `landmark-point-radius` | float | 3.0 | READY | Radius of landmark keypoints. |
| `text-font-scale` | float | 0.0 | PLAYING | Fixed font scale for detection labels. `0` = auto-scale based on bbox width (default behavior). |
| `text-background` | boolean | false | PLAYING | Draw a solid black rectangle behind detection label text for readability. |

### Visibility Toggles

All toggles default to `true` (draw everything) to preserve backward
compatibility. Set to `false` to hide the corresponding element.

| Property | Type | Default | Description |
|---|---|---|---|
| `show-bbox` | boolean | true | Draw bounding box rectangles. |
| `show-labels-text` | boolean | true | Draw detection label and confidence text. |
| `show-landmarks` | boolean | true | Draw landmark keypoints and skeleton lines. |
| `show-tracking-id` | boolean | true | Draw tracking / global ID text. |
| `show-confidence` | boolean | true | Append confidence percentage to labels. |

### Filtering

| Property | Type | Default | Description |
|---|---|---|---|
| `min-confidence` | float | 0.0 | Minimum detection confidence to display. Detections below this threshold are skipped entirely (bbox, text, sub-objects). Range: 0.0 - 1.0. |
| `show-labels` | string | `""` | Comma-separated whitelist of detection labels to show. Empty string = show all. When set, only detections whose label appears in this list are drawn. |
| `hide-labels` | string | `""` | Comma-separated blacklist of detection labels to hide. Empty string = hide none. Detections whose label appears in this list are skipped. |

`show-labels` is evaluated before `hide-labels`. If both are set, a detection
must be in the whitelist AND not in the blacklist to be drawn.

Label strings are trimmed of leading/trailing whitespace, so
`"person, car"` works the same as `"person,car"`.

### Custom Colors

| Property | Type | Default | Description |
|---|---|---|---|
| `use-custom-colors` | boolean | false | Read per-detection color from metadata instead of the built-in class_id color table. |

When enabled, the overlay looks for a `HailoClassification` object attached to
each detection with `classification_type == "overlay_color"`. The color is read
from this object in two ways (tried in order):

1. **Packed class_id** (fast path): If `class_id > 0`, it is interpreted as a
   packed `0xRRGGBB` integer.
2. **Label string** (fallback): The label is parsed as `"R,G,B"` with integer
   values 0-255.

If no `overlay_color` classification is found on a detection, the default
class_id color table is used as a fallback.

Classifications with type `overlay_color` or `overlay_sprite` are never rendered
as text labels, regardless of this setting.

**Color priority order** (highest to lowest):

1. `overlay_color` classification metadata (per-detection, from Python callback)
2. Style config per-label/per-class-id color (from YAML file)
3. Default color table (built-in 19-color table)

#### Setting Custom Colors from Python

In your postprocess callback or app logic, attach a classification to the
detection:

```python
import hailo

# Using packed 0xRRGGBB in class_id (fast path)
color_cls = hailo.HailoClassification(
    "overlay_color",  # type
    0x00FF00,         # class_id = packed 0xRRGGBB (green)
    "",               # label (unused when class_id > 0)
    0.0,              # confidence
)
detection.add_object(color_cls)

# Or using "R,G,B" label string (fallback)
color_cls = hailo.HailoClassification(
    "overlay_color",  # type
    0,                # class_id = 0 (triggers label parse)
    "0,255,0",        # label = "R,G,B"
    0.0,              # confidence
)
detection.add_object(color_cls)
```

### Style Config (Per-Class Overrides via YAML)

| Property | Type | Default | Description |
|---|---|---|---|
| `style-config` | string | `""` | Path to a YAML file with per-class style overrides. Empty = disabled. |

The style config provides automatic per-class visual customization without
requiring Python callbacks. Each detection label (or numeric class_id) can be
mapped to a set of overrides.

#### YAML Format

```yaml
styles:
  person:
    color: [0, 200, 255]       # cyan bbox
    text_color: [255, 255, 255] # white text
    show_landmarks: true
    show_bbox: true
    show_label: true

  car:
    color: [255, 100, 0]       # orange bbox
    show_landmarks: false
    line_thickness: 2

  cat:
    color: [255, 0, 255]       # magenta
    sprite_key: cat_icon        # draw sprite on bbox (requires sprite-config)

  # Numeric class_id keys also work
  0:
    color: [0, 255, 0]
```

#### Available Per-Class Overrides

| Field | Type | Default | Description |
|---|---|---|---|
| `color` | [R, G, B] | inherit | Bounding box color. Overridden by `overlay_color` metadata. |
| `text_color` | [R, G, B] | inherit | Label text color. Defaults to bbox color. |
| `line_thickness` | int | inherit | Override global `line-thickness` for this class. |
| `show_bbox` | bool | inherit | Override global `show-bbox` for this class. |
| `show_label` | bool | inherit | Override global `show-labels-text` for this class. |
| `show_landmarks` | bool | inherit | Override global `show-landmarks` for this class. |
| `sprite_key` | string | none | Sprite key (from `sprite-config`) to draw on the bbox. |

Label matches take priority over numeric class_id matches.

A sample config is at: `examples/overlay_style.yaml`

### Sprite/Stamp System

| Property | Type | Default | Description |
|---|---|---|---|
| `sprite-config` | string | `""` | Path to a YAML file mapping sprite keys to PNG image paths. Empty = disabled. |

Sprites are PNG images (with alpha channel) that can be drawn on top of a
detection's bounding box. They are loaded from disk on first use and cached at
multiple resolutions (bucketed to 8px increments) for efficient reuse.

#### YAML Format

```yaml
sprites:
  thumbs_up: /path/to/thumbs_up.png
  warning: /path/to/warning.png
  cat_icon: /path/to/cat.png
```

#### Triggering Sprites

Sprites can be triggered in two ways:

**1. Per-detection metadata** (from Python callback):

```python
sprite_cls = hailo.HailoClassification(
    "overlay_sprite",  # type
    0,                 # class_id (unused)
    "thumbs_up",       # label = sprite key
    0.0,               # confidence
)
detection.add_object(sprite_cls)
```

**2. Per-class style config** (automatic, from YAML):

In your `style-config` YAML, set `sprite_key` for a detection label:

```yaml
styles:
  cat:
    sprite_key: cat_icon
```

Per-detection metadata takes priority over style config sprite keys.

#### Rendering Details

- The sprite is **letterbox-fit** into the detection bbox, preserving aspect
  ratio, centered in the box.
- **Alpha blending**: Transparent pixels (alpha=0) are skipped, opaque
  pixels (alpha=255) are copied directly, semi-transparent pixels are
  blended.
- Sprites are cached per size bucket (rounded to nearest 8px). On RPi with
  8 sprite keys and ~20 size buckets each, memory usage is roughly 2.5MB.
- Only works on **RGB and RGBA** video formats. On YUY2/NV12 frames the
  sprite is silently skipped.

A sample config is at: `examples/sprites.yaml`

### Stats Overlay

| Property | Type | Default | Description |
|---|---|---|---|
| `stats-overlay` | boolean | false | Display an FPS counter and object count in the top-left corner. |

When enabled, a black background rectangle with white text is drawn showing:

```
FPS: 30 | Objects: 5
```

FPS is calculated from a 30-frame sliding window using monotonic timestamps.
Object count is the number of `HailoDetection` objects in the current frame's
ROI.

### Other

| Property | Type | Default | Description |
|---|---|---|---|
| `face-blur` | boolean | false | Blur detected faces (label == "face"). |
| `local-gallery` | boolean | false | Display identified/unidentified ROIs from a local gallery with global IDs. |
| `mask-overlay-n-threads` | uint | 0 | Number of OpenCV threads for parallel mask drawing. 0 = OpenCV default. |

## Pipeline Examples

### Basic detection with defaults

```bash
gst-launch-1.0 \
  filesrc location=video.mp4 ! decodebin ! videoconvert ! \
  hailonet hef-path=yolov5m.hef ! \
  hailofilter so-path=libyolo_hailortpp_postprocess.so ! \
  hailooverlay_community ! \
  videoconvert ! autovideosink
```

### Show only high-confidence person detections, no landmarks

```bash
... ! hailooverlay_community \
      min-confidence=0.7 \
      show-labels="person" \
      show-landmarks=false \
    ! ...
```

### Hide cars, keep everything else

```bash
... ! hailooverlay_community hide-labels="car, truck" ! ...
```

### Bounding boxes only (no text, no landmarks, no IDs)

```bash
... ! hailooverlay_community \
      show-labels-text=false \
      show-landmarks=false \
      show-tracking-id=false \
    ! ...
```

### Text with dark background for outdoor scenes

```bash
... ! hailooverlay_community \
      text-background=true \
      text-font-scale=0.5 \
    ! ...
```

### Per-class styles from YAML (no Python needed)

```bash
... ! hailooverlay_community \
      style-config=/path/to/overlay_style.yaml \
    ! ...
```

### Sprites on detections

```bash
... ! hailooverlay_community \
      sprite-config=/path/to/sprites.yaml \
      style-config=/path/to/overlay_style.yaml \
    ! ...
```

The style config maps labels to sprite keys; the sprite config maps keys to
PNG files. Both must be set for style-based sprites to work.

### Custom colors from Python callback

```bash
... ! hailooverlay_community use-custom-colors=true ! ...
```

Then in your Python postprocess callback, attach `overlay_color` classifications
to detections as shown in the [Custom Colors](#setting-custom-colors-from-python)
section above.

### FPS and object count display

```bash
... ! hailooverlay_community stats-overlay=true ! ...
```

### Combine multiple features

```bash
... ! hailooverlay_community \
      min-confidence=0.5 \
      show-bbox=true \
      show-labels-text=true \
      show-landmarks=false \
      text-background=true \
      text-font-scale=0.6 \
      use-custom-colors=true \
      stats-overlay=true \
      style-config=/path/to/overlay_style.yaml \
      sprite-config=/path/to/sprites.yaml \
      hide-labels="background" \
    ! ...
```

## Python Demo: Pose Estimation with Overlay Features

A complete demo is provided at:

```
hailo_apps/python/pipeline_apps/pose_estimation/pose_estimation_overlay_demo.py
```

This demo uses the pose estimation pipeline with `hailooverlay_community` to:

- **Color persons by pose**: Arms raised = green, crouching = yellow, default = cyan
- **Text backgrounds** for readability
- **Stats overlay** showing live FPS and object count
- **Min-confidence filter** at 0.5

### Running the Demo

```bash
python -m hailo_apps.python.pipeline_apps.pose_estimation.pose_estimation_overlay_demo
```

With a YAML style config:

```bash
python -m hailo_apps.python.pipeline_apps.pose_estimation.pose_estimation_overlay_demo \
    --style-config hailo_apps/postprocess/cpp/overlay_community/examples/overlay_style.yaml
```

### How It Works

The callback analyzes COCO pose keypoints for each person detection:

| Pose | Condition | Color |
|---|---|---|
| Arms raised | Both wrists above both shoulders | Green (0x00FF00) |
| Crouching | Hip-to-ankle distance < 15% of bbox height | Yellow (0xFFFF00) |
| Default | All other poses | Cyan (0x00C8FF) |

The color is packed as `0xRRGGBB` into the `class_id` field of a
`HailoClassification(type="overlay_color")` and attached to the detection.
The overlay reads it via the existing `use-custom-colors` mechanism.

## Backward Compatibility

All new properties default to values that produce identical output to the
original element:

- All visibility toggles default to `true`
- `min-confidence=0.0` = no filtering
- `show-labels=""` and `hide-labels=""` = no label filtering
- `use-custom-colors=false` = standard class_id color table
- `text-background=false` and `stats-overlay=false` = off
- `text-font-scale=0.0` = auto-scale (original behavior)
- `style-config=""` and `sprite-config=""` = disabled (no YAML loaded)

Existing pipelines require no changes.

## Architecture

### Source Files

| File | Purpose |
|---|---|
| `gsthailooverlay_community.hpp` | GObject struct with all property fields and internal state. |
| `gsthailooverlay_community.cpp` | GStreamer element registration, property handling, `transform_ip` entry point. |
| `overlay.hpp` | `OverlayParams` struct and `draw_all` / `draw_stats_overlay` / `face_blur` declarations. |
| `overlay.cpp` | All drawing logic: detections, classifications, landmarks, masks, custom colors, sprites, filtering, stats. |
| `sprite_cache.hpp` | Sprite loading, resizing, and caching. Header-only, included from overlay.cpp. |
| `style_config.hpp` | Per-class style config parsing and lookup. Header-only, included from overlay.cpp. |
| `examples/overlay_style.yaml` | Sample per-class style config. |
| `examples/sprites.yaml` | Sample sprite key-to-file mapping. |

### OverlayParams

All drawing parameters are passed through a single `OverlayParams` struct
rather than individual function arguments. This keeps the `draw_all` signature
stable as new features are added. The struct includes opaque `void*` pointers
for `SpriteCache` and `StyleConfig` (nullptr when disabled).

### Performance Notes

- All new features are guarded by boolean/nullptr checks and have zero cost when
  disabled (the default).
- Label filtering uses `std::unordered_set` for O(1) lookups. The sets are
  built once when the property is set, not per-frame.
- Style config uses `std::unordered_map` for O(1) per-label lookups. The config
  is parsed once on property set, not per-frame.
- The stats overlay uses a fixed-size 30-slot ring buffer with no heap
  allocation per frame.
- Custom color lookup iterates only the sub-objects of each detection (typically
  0-3 items), with an early exit on the first `overlay_color` match.
- Sprites are cached at bucketed sizes (rounded to 8px), loaded from disk only
  on first access. Source images are also cached. On RPi with 8 sprite keys
  and ~20 size buckets, memory usage is approximately 2.5MB.
- Sprite alpha blending uses a per-pixel loop with early-out for transparent
  pixels and direct copy for opaque pixels. A 100x100 sprite at 30fps with
  5 detections is ~150K ops/frame — negligible on RPi5.

## Color Table

When `use-custom-colors` is disabled and no `style-config` is set, detection
colors are assigned from a built-in 19-color table indexed by `class_id % 19`:

| Index | Color |
|---|---|
| 0 | Red (255, 0, 0) |
| 1 | Green (0, 255, 0) |
| 2 | Blue (0, 0, 255) |
| 3 | Yellow (255, 255, 0) |
| 4 | Cyan (0, 255, 255) |
| 5 | Magenta (255, 0, 255) |
| 6 | Orange (255, 170, 0) |
| 7 | Pink (255, 0, 170) |
| 8 | Spring Green (0, 255, 170) |
| 9 | Chartreuse (170, 255, 0) |
| 10 | Purple (170, 0, 255) |
| 11 | Sky Blue (0, 170, 255) |
| 12 | Dark Orange (255, 85, 0) |
| 13 | Light Green (85, 255, 0) |
| 14 | Teal (0, 255, 85) |
| 15 | Royal Blue (0, 85, 255) |
| 16 | Indigo (85, 0, 255) |
| 17 | Rose (255, 0, 85) |
| 18 | White (255, 255, 255) |

Detections with `class_id == -1` (NULL_CLASS_ID) use white (255, 255, 255).

## License

LGPL v2.1 - see [LICENSE](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt).
