# Running Pre-built Applications

This guide explains how to run the ready-to-use AI applications included in this repository. Each application is a command-line tool designed to showcase a specific AI capability on Hailo hardware.

## Setup Environment
**Note:** This should be run on every new terminal session.
This will activate the virtual environment and set the PYTHONPATH.
```bash
source setup_env.sh
```

## Available Applications

### Pipeline Applications

The following applications are available as command-line tools. Each one is a self-contained GStreamer pipeline that can be launched with a simple command.

| CLI Command           | Application                                                                                    | Description                                                                                                                                                       |
| --------------------- | ---------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hailo-detect-simple` | [Simple Object Detection](../../hailo_apps/python/pipeline_apps/detection_simple/README.md)    | A lightweight version focused on demonstrating raw Hailo performance with minimal CPU overhead. It uses a YOLOv6-Nano model and does not include object tracking. |
| `hailo-detect`        | [Full Object Detection](../../hailo_apps/python/pipeline_apps/detection/README.md)             | A comprehensive detection application featuring object tracking and support for multiple video resolutions.                                                       |
| `hailo-pose`          | [Pose Estimation](../../hailo_apps/python/pipeline_apps/pose_estimation/README.md)             | Detects human pose keypoints (e.g., joints and limbs) in real-time.                                                                                               |
| `hailo-seg`           | [Instance Segmentation](../../hailo_apps/python/pipeline_apps/instance_segmentation/README.md) | Provides pixel-level masks for each detected object, distinguishing different instances from one another.                                                         |
| `hailo-depth`         | [Depth Estimation](../../hailo_apps/python/pipeline_apps/depth/README.md)                      | Estimates the depth of a scene from a single 2D camera input.                                                                                                     |
| `hailo-face-recon`    | [Face Recognition](../../hailo_apps/python/pipeline_apps/face_recognition/README.md)           | A face recognition application that identifies and verifies faces in real-time. This application is currently in BETA.                                            |
| `hailo-tiling`        | [Tiling](../../hailo_apps/python/pipeline_apps/tiling/README.md)                               | Single & multi-scale tiling splitting each frame into several tiles, effective for small objects in high-resolution frames.                                       |
| `hailo-multisource`   | [Multisource](../../hailo_apps/python/pipeline_apps/multisource/README.md)                     | Demonstrating parallel processing on multiple streams from a combination of various inputs (USB cameras, files, RTSP, etc.).                                      |
| `hailo-reid`          | [REID Multisource](../../hailo_apps/python/pipeline_apps/reid_multisource/README.md)           | Track people (faces) across multiple cameras (or any other input method) in a pipeline with multiple streams. This application is currently in BETA.              |
| `hailo-clip`          | [CLIP Zero-shot](../../hailo_apps/python/pipeline_apps/clip/README.md)                         | Zero-shot image classification using CLIP-style embeddings for flexible label sets.                                                                               |
| `hailo-ocr`           | [PaddleOCR](../../hailo_apps/python/pipeline_apps/paddle_ocr/README.md)                        | Text detection and recognition using PaddleOCR models. This application is currently in BETA.                                                                     |

### GenAI Standalone Applications

These standalone GenAI applications are located in `hailo_apps/python/gen_ai_apps/` and can be run directly as Python scripts.

| Application                                                                                | Description                                                                                                                                                               |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Agent Tools Example](../../hailo_apps/python/gen_ai_apps/agent_tools_example/README.md)   | **Voice-to-action AI agent** with tool integration for controlling hardware (elevators, servos, RGB LEDs) and accessing external services (weather API, math operations). |
| [Voice Assistant](../../hailo_apps/python/gen_ai_apps/voice_assistant/README.md)           | Complete voice assistant implementation combining speech recognition, LLM, and text-to-speech.                                                                            |
| [VLM Chat](../../hailo_apps/python/gen_ai_apps/vlm_chat/README.md)                         | Vision-Language Model chat application combining vision and language understanding.                                                                                       |
| [Speech Recognition](../../hailo_apps/python/standalone_apps/speech_recognition/README.md) | Hailo Whisper-based speech recognition application with GUI.                                                                                                              |

**Simple Example Applications:** The `gen_ai_apps/` directory contains additional simple examples including LLM chat, VLM chat, and Whisper chat demonstrations. See the [GenAI Apps README](../../hailo_apps/python/gen_ai_apps/README.md) for more details.

### Other Standalone Applications

In addition to the pipeline and GenAI applications above, this repository includes other standalone Python applications for computer vision use cases. These applications are located in `hailo_apps/python/standalone_apps/` and can be run directly as Python scripts.

| Application                                                                                      | Description                                                              |
| ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------ |
| [Instance Segmentation](../../hailo_apps/python/standalone_apps/instance_segmentation/README.md) | Standalone example of instance segmentation with custom post-processing. |
| [Lane Detection](../../hailo_apps/python/standalone_apps/lane_detection/README.md)               | Lane detection application for automotive use cases.                     |
| [Object Detection](../../hailo_apps/python/standalone_apps/object_detection/README.md)           | Standalone object detection example with custom post-processing.         |
| [PaddleOCR](../../hailo_apps/python/standalone_apps/paddle_ocr/README.md)                        | Standalone text detection and recognition using PaddleOCR models.        |
| [Pose Estimation](../../hailo_apps/python/standalone_apps/pose_estimation/README.md)             | Standalone pose estimation example with custom visualization.            |
| [Super Resolution](../../hailo_apps/python/standalone_apps/super_resolution/README.md)           | Image super-resolution for enhancing image quality.                      |

These standalone applications typically require additional dependencies which can be installed using the `requirements.txt` file in each application's directory.

## How to Run Pipeline Applications

Pipeline applications can be run using their CLI command. For example, to start the simple object detection:

```bash
hailo-detect-simple
```
To close any application, press `Ctrl+C` in the terminal.

![Detection Example](../images/detection.gif)

### Selecting an Input Source

By default, applications may use a pre-packaged video file. You can specify a different input source using the `--input` (or `-i`) flag.

**Run with a Raspberry Pi Camera:**
```bash
hailo-detect --input rpi
```

**Run with a USB Camera (Webcam):**
This command will automatically find and use the first available USB camera.
```bash
hailo-detect --input usb
```

**Run with a specific camera device:**
First, find your camera's device path. You can use a command like `ls /dev/video*` or our provided script:
```bash
get-usb-camera
```
Then, use the device path as the input:
```bash
hailo-detect --input /dev/video0
```

**Run with a video file:**
```bash
hailo-detect --input your_video.mp4
```

**Run with an RTSP (Real-Time Streaming Protocol):**
```bash
hailo-detect --input rtsp://username:password@ip_address:port/path
```

## Customizing with Command-Line Arguments

While the applications run out-of-the-box, you can customize their behavior using command-line arguments.

For a quick list of all options for any command, use the `--help` flag:
```bash
hailo-detect --help
```

All applications share a common set of arguments for controlling the input source, hardware, performance, and display settings.

---

## How to Run Standalone Applications

Standalone applications are Python scripts located in `hailo_apps/python/standalone_apps/`. Each application has its own README file with specific instructions.

**General steps to run a standalone application:**

1. Navigate to the application directory:
```bash
cd hailo_apps/python/standalone_apps/<application_name>/
```

2. Install application-specific dependencies (if a `requirements.txt` exists):
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python <application_name>.py [arguments]
```

**Example - Running the Voice Assistant:**
```bash
cd hailo_apps/python/gen_ai_apps/voice_assistant/
python voice_assistant.py
```

Refer to each application's README for detailed usage instructions, configuration options, and specific requirements.

---

## Command-Line Argument Reference (Pipeline Applications)

| Flag(s)                  | Description                                                                                                                                   |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `--input, -i <source>`   | Specifies the input source. Common options include: `rpi`, `usb`, a device path like `/dev/video0`, or a path to a video file.                |
| `--arch <architecture>`  | Manually sets the Hailo device architecture (e.g., `hailo8`, `hailo8l`, `hailo10h`). If not provided, the system will auto-detect the device. |
| `--hef-path <path>`      | Path to a custom compiled HEF model file, allowing you to run your own trained models.                                                        |
| `--show-fps, -f`         | Displays a real-time Frames-Per-Second (FPS) counter on the output video window.                                                              |
| `--frame-rate, -r <fps>` | Sets the target input frame rate for the video source. Defaults to 30.                                                                        |
| `--disable-sync`         | Disables display synchronization to run the pipeline at maximum speed. This is ideal for benchmarking processing throughput.                  |
| `--disable-callback`     | Disables user-defined Python callback functions. Frame counting for watchdog continues. Use for performance benchmarking.                     |
| `--dump-dot`             | Generates a `pipeline.dot` file, which is a graph of the GStreamer pipeline that can be visualized with tools like Graphviz.                  |
| `--labels-json <path>`   | Path to a custom JSON file containing the labels for the classes your model can detect or classify.                                           |
| `--use-frame, -u`        | In applications with a Python callback, this flag indicates that the callback is responsible for providing the frame for display.             |
| `--enable-watchdog`      | Monitors the pipeline for stalls (no frames processed for 5s) and automatically rebuilds it. Works with --disable-callback.                   |
| `--log-level <level>`    | Set logging level: debug, info, warning, error, critical. Default: info. Can also use --debug for debug level.                                |
| `--log-file <path>`      | Optional log file path for persistent logging. Also respects $HAILO_LOG_FILE environment variable.                                            |