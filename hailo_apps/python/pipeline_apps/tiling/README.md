# Tiling Application

> ⚠️ **Beta:** This application is currently in beta. Features and APIs may change.

![Tiling Example](../../../../local_resources/tiling.gif)

## Overview

The tiling pipeline demonstrates splitting each frame into several tiles which are processed independently by the `hailonet` element. This method is especially effective for **detecting small objects in high-resolution frames**.

**To demonstrate tiling capabilities, we've selected a drone/aerial use case as the default:**
- **Default Model:** `hailo_yolov8n_4_classes_vga` - optimized for aerial object detection
- **Default Video:** `tiling_visdrone_720p.mp4` - aerial footage with small objects
- **Use Case:** Perfect for demonstrating small object detection in high-resolution frames

For scenes with varied object sizes, you can use the `--multi-scale` flag to enable multi-scale tiling.


## Usage Examples
**Note: It's very easy to reach very high FPS requirements by using too many tiles.**
If you application seems to be slow, try to use less tiles. If you do need to use more tiles, you can reduce the frame rate using the `--frame-rate` flag.

### Basic Examples

**Default (aerial detection):**
```bash
hailo-tiling
```
To close the application, press `Ctrl+C`.

- Uses `hailo_yolov8n_4_classes_vga` + VisDrone video
- Perfect for demonstrating small object detection

**With multi-scale for varied object sizes:**
```bash
hailo-tiling --multi-scale
```
- Uses `hailo_yolov8n_4_classes_vga` + multi-scale
- Optimized for scenes with a mix of small and large objects

**With live camera:**
```bash
hailo-tiling --input rpi --multi-scale
```
- Uses `hailo_yolov8n_4_classes_vga` with a live camera feed and multi-scale.

**Manual tile grid:**
```bash
hailo-tiling --input rpi --tiles-x 3 --tiles-y 2 --hef /path/to/hef_file.hef
```
- Uses Raspberry Pi Camera
- Creates exactly 3×2 = 6 tiles
- Uses custom HEF file (a stronger model for example)

## How Tiling Works

### Single-Scale Tiling (Default)
In single-scale mode, each frame is divided into a grid of tiles. Each tile is:
- Sized to match the model's input resolution
- Processed independently through the detection model
- Results are aggregated and filtered using NMS to remove duplicates

**Benefits:**
- Maintains high detection accuracy for small objects
- No downscaling of the original image
- Efficient processing of high-resolution frames

### Multi-Scale Tiling (Advanced)
Multi-scale mode is useful when your scene contains objects of varying sizes. It **adds predefined tile grids** on top of your custom tile configuration.

**How it works:** Multi-scale mode processes BOTH:
1. **Your custom tile grid** (auto-calculated or manually specified)
2. **Additional predefined grids** based on scale-level

Multi-scale adds these predefined grids:
- **scale-level 1**: Adds full image (1×1) = +1 tile
- **scale-level 2**: Adds 1×1 + 2×2 = +5 tiles
- **scale-level 3**: Adds 1×1 + 2×2 + 3×3 = +14 tiles

**Example:** With 4×3 custom grid and `--multi-scale --scale-levels 2`:
- Custom tiles: 4×3 = 12 tiles
- Additional: 1×1 + 2×2 = 5 tiles
- **Total: 17 tiles per frame**

The pipeline performs: Crop → Inference → Post-process → Aggregate → Remove border objects → Perform NMS

## Understanding Overlap

**Overlap** is the percentage of tile area that overlaps with adjacent tiles. It's automatically calculated to ensure:
1. Full coverage of the input frame
2. Objects on tile boundaries are detected reliably

### Why Overlap Matters
- Objects that fall exactly on tile boundaries might be cut in half
- Each half might be too small to be detected reliably
- **Critical Rule:** Overlap should be ≥ the pixel size of your **smallest object** you need to detect

### Setting Minimum Overlap

Use the `--min-overlap` parameter to ensure sufficient overlap for your objects:

**Formula:** `min-overlap ≥ smallest_object_size / model_input_size`

**Examples for 640×640 model:**
- Detecting 32px objects: `--min-overlap 0.05` (32/640 = 0.05 = 5%)
- Detecting 64px objects: `--min-overlap 0.10` (64/640 = 0.10 = 10%, **default**)

### How It Works

1. **Auto Mode:** The application calculates the number of tiles needed to ensure at least `min-overlap` between adjacent tiles
2. **Manual Mode:** If you specify tile counts that result in less than `min-overlap`, the tiles will be enlarged to meet the minimum overlap. And you'll receive a warning

### Overlap Recommendations


**Important:** Setting `--min-overlap` ensures the application creates enough tiles to maintain the specified overlap. A higher minimum overlap means:
- More tiles will be generated
- Better coverage of objects on boundaries
- Higher computational cost (more tiles to process)

**Note:** The actual overlap may be higher than the minimum if needed for full frame coverage.

## Why Use Native Resolution Tiles (No Upscaling)

To ensure maximum detection performance, particularly for small objects, all tiled inputs must meet or exceed the model's trained resolution.

### The Critical Issue: Data Fidelity

**Loss of Detail:** When you slice the original image into smaller tiles and then upsample them back to the original size, you are not adding information. The interpolation process (like bilinear or bicubic resizing) introduces smoothing and artifacts, effectively blurring the fine details that the model relies on to distinguish small objects from the background.

**Training Mismatch:** The model was trained on high-quality, native-resolution data. Feeding it synthetically upscaled, degraded inputs creates a distribution shift between the training and inference data, leading to an immediate and significant drop in Mean Average Precision (mAP) and overall detection reliability.

**Feature Integrity:** The early layers of the convolutional neural network (CNN) are responsible for extracting low-level features (edges, corners). These features are severely compromised by the upscaling artifacts, hindering the entire detection pipeline.

**Actionable Rule:** Always tile your large image into chunks that are the model's input size or larger, and then crop/resize down to the model's input size if necessary, but never resize a sub-native tile up to match the required input size.

### How This Application Handles It

This tiling application automatically ensures that:
- Each tile is **exactly the model's input size**
- **No upscaling occurs** - tiles are always at or above native resolution
- The original image is **never downscaled** before tiling
- Overlap between tiles compensates for boundary effects without degrading quality

This approach maintains the full fidelity of your input data, ensuring optimal detection accuracy.

## Command-Line Arguments

### Model Options

*   `--hef-path` - Path to a custom HEF model file. If not specified, the default `hailo_yolov8n_4_classes_vga` is used.

**Automatic Configuration:**
The pipeline automatically reads the model's input resolution directly from the HEF file.

**Default Video:**
- Uses `tiling_visdrone_720p.mp4` as the default when no input is specified

### Tiling Options

The application operates in two modes:

**Auto Mode (Default):** Automatically calculates optimal tile grid based on model size
- Tiles are sized to `model-input-size × model-input-size`
- Grid is calculated to cover the entire frame
- Overlap is automatically adjusted for full coverage

**Manual Mode:** Activated when you specify `--tiles-x` or `--tiles-y`
- You control the tile grid dimensions
- Tiles sized to model input resolution (or larger if needed for minimum overlap)
- Tiles always maintain square aspect ratio (width = height)
- Overlap automatically calculated to ensure coverage
- **Note:** If minimum overlap can't be met with model input size, tiles will be enlarged while maintaining square aspect ratio

*   `--tiles-x` - Number of tiles horizontally (triggers manual mode)
*   `--tiles-y` - Number of tiles vertically (triggers manual mode)
*   `--min-overlap` - Minimum overlap ratio between tiles (default: 0.1 = 10%)
    - Should be ≥ (smallest_object_size / model_input_size)
    - Example: For 64px objects with 640px model: `--min-overlap 0.10`
    - Higher values create more tiles for better object coverage

**Note:** You can specify just `--tiles-x` or just `--tiles-y`, and the other dimension will be auto-calculated.

### Multi-Scale Options

**Note:** Multi-scale mode ADDS predefined grids to your custom tiles (does not replace them)

*   `--multi-scale` - Enable multi-scale tiling (adds predefined grids to custom tiles)
*   `--scale-levels` - Which predefined grids to add (default: 1, range: 1-3)
    - `1`: Adds full image (1×1) = +1 tile
    - `2`: Adds 1×1 + 2×2 = +5 tiles
    - `3`: Adds 1×1 + 2×2 + 3×3 = +14 tiles

### Detection Options

*   `--iou-threshold` - NMS IOU threshold for filtering overlapping detections (default: 0.3)
*   `--border-threshold` - Border threshold to remove tile edge detections in multi-scale mode (default: 0.15)

## Configuration Output

When you run the application, it displays a detailed configuration summary:

```
======================================================================
TILING CONFIGURATION
======================================================================
Input Resolution:     1280x720
Model:                hailo_yolov8n_4_classes_vga.hef (YOLO, 640x480)

Tiling Mode:          AUTO
Custom Tile Grid:     3x2 = 6 tiles
Tile Size:            640x480 pixels
Overlap:              X: 50.0% (~320px), Y: 50.0% (~240px)

Multi-Scale:          DISABLED
  Total Tiles:        6

Detection Parameters:
  Batch Size:         6
  IOU Threshold:      0.3
======================================================================
```

This helps you understand:
- What tile configuration is being used
- Whether overlap is sufficient for your use case
- Total number of tiles being processed per frame

## Performance Considerations

**Batch Size:**
- The batch size is automatically set to match the total number of tiles
- Single-scale with 4×3 grid → batch size = 12
- Multi-scale (2×2 custom) + scale-levels 3 → batch size = 4 + 14 = 18
- This ensures optimal processing throughput

**Tile Count Impact:**
- More tiles = better small object detection
- More tiles = higher processing time (scales with batch size)
- Balance based on your hardware and requirements

**Hailo8L Performance:**
- For Hailo8L with the default `ssd_mobilenet_visdrone` model, the frame rate is automatically set to 19 fps to support Hailo8L's lower performance
- This adjustment is applied automatically when using the default MobileNetSSD model with batch size 15 on Hailo8L devices

**Experiment:** Try different tile counts and overlap values to find the best balance for your use case. The "auto mode" is a good starting point but probably an overkill for your use case. Note that you might get better results by using less tiles and a stronger model.


### All pipeline commands support these common arguments:

[Common arguments](../../../../doc/user_guide/running_applications.md#command-line-argument-reference)