# Running Applications (Pipeline and Standalone)

This guide explains how to run the ready-to-use AI applications included in this repository. It covers both:

- **Full-repository installation flow** (pipeline CLIs + all app groups)
- **Standalone app flow** (Python/C++ apps run independently)

If you still need installation steps, see the [Installation Guide](installation.md).

## Setup Environment
If you installed only a standalone app independently, you can skip this step and follow that app's local setup instructions in its README.

```bash
source setup_env.sh
```

## Useful CLI Utilities

| Command | Purpose |
| --- | --- |
| `get-usb-camera` | Detect and print available USB camera devices. |
| `hailo-download-resources` | Download/update model resources (by app group, model, or architecture). |
| `hailo-post-install` | Complete post-install setup (resources + postprocess compilation). |
| `hailo-audio-troubleshoot` | Audio diagnostics utility for voice workflows. |


## Available Applications

### Pipeline Applications

The following applications are available as command-line tools. Each one is a self-contained GStreamer pipeline that can be launched with a simple command.

> **Note:** Pipeline apps rely on resources (models, post-process `.so` files). In full installs this is handled by `install.sh`. For pip/manual installs, run `hailo-post-install` (see [Installation Guide](installation.md)).

| CLI Command           | Application                                                                                    | Description                                                                                                                                                       |
| --------------------- | ---------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hailo-detect-simple` | [Simple Object Detection](../../hailo_apps/python/pipeline_apps/detection_simple/README.md)    | A lightweight version focused on demonstrating raw Hailo performance with minimal CPU overhead. It uses a YOLOv6-Nano model and does not include object tracking. |
| `hailo-detect`        | [Full Object Detection](../../hailo_apps/python/pipeline_apps/detection/README.md)             | A comprehensive detection application featuring object tracking and support for multiple video resolutions.                                                       |
| `hailo-pose`          | [Pose Estimation](../../hailo_apps/python/pipeline_apps/pose_estimation/README.md)             | Detects human pose keypoints (e.g., joints and limbs) in real-time.                                                                                               |
| `hailo-seg`           | [Instance Segmentation](../../hailo_apps/python/pipeline_apps/instance_segmentation/README.md) | Provides pixel-level masks for each detected object, distinguishing different instances from one another.                                                         |
| `hailo-depth`         | [Depth Estimation](../../hailo_apps/python/pipeline_apps/depth/README.md)                      | Estimates the depth of a scene from a single 2D camera input.                                                                                                     |
| `hailo-tiling`        | [Tiling](../../hailo_apps/python/pipeline_apps/tiling/README.md)                               | Single & multi-scale tiling splitting each frame into several tiles, effective for small objects in high-resolution frames.                                       |
| `hailo-clip`          | [CLIP Zero-shot](../../hailo_apps/python/pipeline_apps/clip/README.md)                         | Zero-shot image classification using CLIP-style embeddings for flexible label sets.                                                                               |
| `hailo-multisource`   | [Multisource](../../hailo_apps/python/pipeline_apps/multisource/README.md)                     | Demonstrating parallel processing on multiple streams from a combination of various inputs (USB cameras, files, RTSP, etc.).                                      |
| `hailo-face-recon`    | [Face Recognition](../../hailo_apps/python/pipeline_apps/face_recognition/README.md)           | A face recognition application that identifies and verifies faces in real-time. This application is currently in BETA.                                            |
| `hailo-ocr`           | [PaddleOCR](../../hailo_apps/python/pipeline_apps/paddle_ocr/README.md)                        | Text detection and recognition using PaddleOCR models.  This application is currently in BETA.                                                                    |
| `hailo-reid`          | [REID Multisource](../../hailo_apps/python/pipeline_apps/reid_multisource/README.md)           | Track people (faces) across multiple cameras (or any other input method) in a pipeline with multiple streams. This application is currently in BETA.              |
|                       | [Easter Game](../../hailo_apps/python/pipeline_apps/easter_game/README.md)                     | Interactive Easter Egg catching game using pose estimation. Created autonomously by AI.                                                                            |

### Python GenAI Applications

These standalone GenAI applications are located in `hailo_apps/python/gen_ai_apps/` and can be run directly as Python scripts.

| Application                                                                                | Description                                                                                                                                                               |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
| [Agent Tools Example](../../hailo_apps/python/gen_ai_apps/agent_tools_example/README.md)   | **Voice-to-action AI agent** with tool integration for controlling hardware (elevators, servos, RGB LEDs) and accessing external services (weather API, math operations). |
| [Voice Assistant](../../hailo_apps/python/gen_ai_apps/voice_assistant/README.md)           | Complete voice assistant implementation combining speech recognition, LLM, and text-to-speech.                                                                            |
| [VLM Chat](../../hailo_apps/python/gen_ai_apps/vlm_chat/README.md)                         | Vision-Language Model chat application combining vision and language understanding.                                                                                       |
| [Hailo Ollama](../../hailo_apps/python/gen_ai_apps/hailo_ollama/README.md)                 | Ollama integration utilities for running local LLM workflows with Hailo.                                                                                                 |
| [Simple LLM Chat](../../hailo_apps/python/gen_ai_apps/simple_llm_chat/README.md)           | Minimal text-only LLM chat example.                                                                                                                                        |
| [Simple VLM Chat](../../hailo_apps/python/gen_ai_apps/simple_vlm_chat/README.md)           | Minimal Vision-Language chat example (image + text).                                                                                                                      |
| [Simple Whisper Chat](../../hailo_apps/python/gen_ai_apps/simple_whisper_chat/README.md)   | Minimal Whisper-based speech recognition chat example.                                                                                                                     |
| [Voice-to-Action Demo](../../hailo_apps/python/gen_ai_apps/v2a_demo/README.md)             | End-to-end voice pipeline: wake word → Whisper STT → tool selection → LLM → Piper TTS, for controlling hardware via spoken commands.                                     |

See the [GenAI Apps README](../../hailo_apps/python/gen_ai_apps/README.md) for additional details and usage notes.

### Python Standalone Applications

This repository also includes standalone Python applications for computer vision use cases. These applications are located in `hailo_apps/python/standalone_apps/` and can be run directly as Python scripts.

| Application                                                                                      | Description                                                              |
| ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------ |
| [Instance Segmentation](../../hailo_apps/python/standalone_apps/instance_segmentation/README.md) | Standalone example of instance segmentation with custom post-processing. |
| [Lane Detection](../../hailo_apps/python/standalone_apps/lane_detection/README.md)               | Lane detection application for automotive use cases.                     |
| [Object Detection](../../hailo_apps/python/standalone_apps/object_detection/README.md)           | Standalone object detection example with custom post-processing.         |
| [Oriented Object Detection](../../hailo_apps/python/standalone_apps/oriented_object_detection/README.md) | Standalone oriented object detection (rotated bounding boxes).           |
| [PaddleOCR](../../hailo_apps/python/standalone_apps/paddle_ocr/README.md)                        | Standalone text detection and recognition using PaddleOCR models.        |
| [Pose Estimation](../../hailo_apps/python/standalone_apps/pose_estimation/README.md)             | Standalone pose estimation example with custom visualization.            |
| [Super Resolution](../../hailo_apps/python/standalone_apps/super_resolution/README.md)           | Image super-resolution for enhancing image quality.                      |
| [Speech Recognition](../../hailo_apps/python/standalone_apps/speech_recognition/README.md)        | Speech recognition for Hailo-8/8L/10H with live mic recording.          |
| [YOLO26 Object Detection](../../hailo_apps/python/standalone_apps/yolo26/object_detection/README.MD) | YOLO26 object detection with split HEF + ONNX postprocessing pipeline.  |
| [YOLO26 Pose Estimation](../../hailo_apps/python/standalone_apps/yolo26/pose_estimation/README.md)   | YOLO26 pose estimation with split HEF + ONNX postprocessing pipeline.   |

These standalone applications typically require additional dependencies which can be installed using the `requirements.txt` file in each application's directory.

### C++ Standalone Applications

This repository also includes standalone C++ applications under `hailo_apps/cpp/`.

| Application                                                                                 | Description                                                           |
| ------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| [Classification](../../hailo_apps/cpp/classification/README.md)                            | Image classification with models trained on ImageNet.                |
| [Mono Depth Estimation](../../hailo_apps/cpp/depth_estimation_mono/README.md)             | Monocular depth estimation with SCDepthV3.                           |
| [Stereo Depth Estimation](../../hailo_apps/cpp/depth_estimation_stereo/README.md)         | Stereo depth estimation with StereoNet.                              |
| [Instance Segmentation](../../hailo_apps/cpp/instance_segmentation/README.md)             | Instance segmentation with YOLO segmentation models.                 |
| [Object Detection](../../hailo_apps/cpp/object_detection/README.md)                        | Generic and asynchronous object detection.                           |
| [ONNX Runtime Pipeline](../../hailo_apps/cpp/onnxrt_hailo_pipeline/README.md)             | Inference with Hailo and post-processing via ONNX Runtime.           |
| [Oriented Object Detection](../../hailo_apps/cpp/oriented_object_detection/README.md)      | Detection with rotated bounding boxes.                               |
| [Pose Estimation](../../hailo_apps/cpp/pose_estimation/README.md)                          | Human pose estimation.                                                |
| [Semantic Segmentation](../../hailo_apps/cpp/semantic_segmentation/README.md)              | Pixel-wise semantic segmentation (Cityscapes-style outputs).         |
| [Zero-Shot Classification](../../hailo_apps/cpp/zero_shot_classification/README.md)        | CLIP-based zero-shot image classification.                           |

For C++ build and usage instructions, see [CPP Examples README](../../hailo_apps/cpp/README.md).

## Run Guidance

### Pipeline apps (CLI)

- Use this flow when you installed the full repository.
- Activate environment per terminal:
```bash
source setup_env.sh
```
- Run any pipeline command directly (for example `hailo-detect`, `hailo-pose`, `hailo-seg`).
- See app-specific options with:
```bash
<command> --help
```
#### Common Pipeline Arguments

| Flag(s)                     | Description                                                                                                                                   |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `--input, -i <source>`      | Specifies the input source. Common options include: `rpi`, `usb`, a device path like `/dev/video0`, an RTSP URL, or a path to a video file.    |
| `--arch, -a <architecture>` | Manually sets the Hailo device architecture (`hailo8`, `hailo8l`, `hailo10h`). If not provided, the system will auto-detect the device.        |
| `--hef-path, -n <path>`     | Path or name of a custom compiled HEF model file, allowing you to run your own trained models.                                                 |
| `--list-models`             | Lists all available models for the application (default + extras usable with `--hef-path`) and exits.                                          |
| `--batch-size, -b <n>`      | Number of frames processed in parallel during inference. Default is 1.                                                                          |
| `--show-fps`                | Displays a real-time Frames-Per-Second (FPS) counter on the output video window.                                                                |
| `--frame-rate, -f <fps>`    | Sets the target input frame rate for the video source. Defaults to 30.                                                                          |
| `--width, -W <px>` / `--height, -H <px>` | Custom output resolution; if omitted, uses the input resolution or model default.                                                  |
| `--labels, -l <path>`       | Path to a text file with one class label per line, used to override the default label set.                                                     |
| `--labels-json <path>`      | Path to a custom JSON labels file (used by detection/CLIP/tiling apps); auto-detected from the HEF when not provided.                           |
| `--disable-sync`            | Disables display synchronization to run the pipeline at maximum speed. This is ideal for benchmarking processing throughput.                    |
| `--disable-callback`        | Disables user-defined Python callback functions. Frame counting for watchdog continues. Use for performance benchmarking.                       |
| `--use-frame`               | In applications with a Python callback, this flag indicates that the callback is responsible for providing the frame for display.               |
| `--dump-dot`                | Generates a `pipeline.dot` file, which is a graph of the GStreamer pipeline that can be visualized with tools like Graphviz.                    |
| `--print-pipeline`          | Prints the GStreamer pipeline string to stdout before launching.                                                                                |
| `--horizontal-mirror` / `--vertical-mirror` | Flips the video source horizontally/vertically (useful when the camera is mounted upside down).                                |
| `--run-duration <seconds>`  | Runs the pipeline for the given number of seconds, then shuts down cleanly (EOS). Useful for benchmarking/automated testing.                    |
| `--enable-watchdog`         | Monitors the pipeline for stalls (no frames processed for 5s) and automatically rebuilds it. Works with `--disable-callback`.                   |
| `--log-level <level>`       | Set logging level: debug, info, warning, error, critical. Default: info. Can also use `--debug` for debug level.                                |
| `--log-file <path>`         | Optional log file path for persistent logging. Also respects `$HAILO_LOG_FILE` environment variable.                                            |

### Standalone apps (Python + C++)

- "Standalone" means standalone **installation flow** (no full `install.sh` flow required).
- Many standalone apps still depend on shared modules in this repo, so a repo clone is commonly required.
- Use each app README as source of truth for exact dependencies and run command.

Quick notes:
- Python CV standalone apps: usually install local `requirements.txt` in app directory.
- Python GenAI apps: install extras from repo root:
```bash
pip install -e ".[gen-ai]"
```
- C++ standalone apps: build/run from each app folder (`./build.sh`).

For C++ overview, see [CPP Examples README](../../hailo_apps/cpp/README.md).