# Hailo C++ Applications

> **Beta:** These applications are actively developed. APIs and features may change between releases.

A collection of standalone C++ inference applications for Hailo AI accelerators. Each application runs inference directly using the HailoRT C++ API — no GStreamer required — and supports image, video file, and live camera inputs out of the box.

---

## Table of Contents

- [Supported Platforms](#supported-platforms)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Building the Applications](#building-the-applications)
- [Application Reference](#application-reference)
- [Running an Application](#running-an-application)
- [Input Sources](#input-sources)
- [Visualization Configuration](#visualization-configuration)
- [Using Multiple Models on the Same Device](#using-multiple-models-on-the-same-device)
- [Platform Notes](#platform-notes)

---

## Supported Platforms

| Platform | Architecture | Notes |
|---|---|---|
| Linux x86_64 | x64 | Primary development platform |
| Linux aarch64 | arm64 | NXP i.MX8, Raspberry Pi 5, and similar SBCs |
| Windows 10/11 | x64 | Requires Visual Studio with MSVC build tools |

Supported Hailo devices: **Hailo-8**, **Hailo-8L**, **Hailo-10H**

---

## Prerequisites

### Linux

| Dependency | Minimum Version | Install |
|---|---|---|
| HailoRT | 4.23.0 (Hailo-8/8L) / 5.1.0 (Hailo-10H) | [Hailo Developer Zone](https://hailo.ai/developer-zone/) |
| CMake | 3.20 | `pip install cmake` |
| OpenCV | 4.5.4 | `sudo apt-get install libopencv-dev` |
| GCC / G++ | 9+ | `sudo apt-get install build-essential` |
| Git | any | `sudo apt-get install git` |

Install the yaml-cpp and libcurl development libraries:

```bash
sudo apt-get install libyaml-cpp-dev libcurl4-openssl-dev
```

If these system packages are unavailable, the build scripts build yaml-cpp and libcurl from the bundled submodules.

### Windows

| Dependency | Minimum Version | Install |
|---|---|---|
| HailoRT | 4.23.0 (Hailo-8/8L) / 5.1.0 (Hailo-10H) | [Hailo Developer Zone](https://hailo.ai/developer-zone/) |
| CMake | 3.20 | [cmake.org](https://cmake.org/download/) |
| OpenCV | 4.5.4 | `vcpkg install opencv` |
| Visual Studio | 2019+ | With "Desktop development with C++" workload |
| Git for Windows | any | [git-scm.com](https://git-scm.com/download/win) — required by `build.ps1` for Unix tools (`sed`) |

---

## Getting Started

### 1. Clone the Repository

Clone with `--recurse-submodules` to pull in the bundled third-party libraries (yaml-cpp, curl) that are required for the build:

```bash
git clone --recurse-submodules https://github.com/hailo-ai/hailo-apps.git
cd hailo-apps
```

If you already cloned without submodules, initialize them now:

```bash
git submodule update --init --recursive
```

> **Why submodules?** yaml-cpp and libcurl are included as Git submodules under `hailo_apps/cpp/external/`. The build system uses them automatically if system packages are not available, so the build works identically on all supported platforms without additional setup.

### 2. Navigate to the C++ Applications Directory

```bash
cd hailo_apps/cpp
```

---

## Building the Applications

All applications are built through a single build script: **`build.sh`** on Linux and **`build.ps1`** on Windows. Both scripts share the same interface and behavior:

1. Check whether yaml-cpp and libcurl are available as system packages.
2. If not, build them **once** from the bundled submodules into a shared `deps/` directory.
3. Build whichever applications you specify (or all of them).
4. Print a pass/fail summary at the end.

Shared dependencies are built only once and reused across all applications — subsequent builds are fast.

Use `build.sh` on Linux and `build.ps1` on Windows — the interface is identical.

#### Build All Applications

```bash
./build.sh
```

#### Build Specific Applications

```bash
./build.sh object_detection
./build.sh object_detection instance_segmentation pose_estimation
```

#### Clean Build (Remove Previous Build Artifacts)

Use `--rebuild` to delete each application's `build/` directory before rebuilding. Useful when switching branches or after changing CMake options:

```bash
./build.sh --rebuild
./build.sh --rebuild object_detection
```

#### Help

```bash
./build.sh --help
```

#### Build Output

Each application binary is placed in its own `build/` directory:

```
hailo_apps/cpp/
├── object_detection/build/object_detection
├── instance_segmentation/build/instance_segmentation
├── classification/build/classifier
└── ...
```

A `config/` folder is copied next to each binary at build time, containing the YAML configuration files the application needs at runtime:

```
object_detection/build/
├── object_detection          ← executable
└── config/
    ├── resources_config.yaml
    └── visualization_config.yaml
```

#### Build Summary

At the end of each build, a summary shows which applications succeeded and which failed:

```
==========================================
 Build Summary
==========================================
  ✓  classification
  ✓  object_detection
  ✓  instance_segmentation
  ✗  onnxrt_hailo_pipeline
==========================================
```

---

## Application Reference

| Application | Description |
|---|---|
| [`classification`](classification/README.md) | Image classification |
| [`depth_estimation_mono`](depth_estimation_mono/README.md) | Monocular depth estimation |
| [`depth_estimation_stereo`](depth_estimation_stereo/README.md) | Stereo depth estimation |
| [`instance_segmentation`](instance_segmentation/README.md) | Instance segmentation with masks |
| [`object_detection`](object_detection/README.md) | Generic object detection |
| [`onnxrt_hailo_pipeline`](onnxrt_hailo_pipeline/README.md) | Hailo inference + ONNX Runtime postprocessing |
| [`oriented_object_detection`](oriented_object_detection/README.md) | Object detection with rotation angles |
| [`pose_estimation`](pose_estimation/README.md) | Human pose estimation |
| [`semantic_segmentation`](semantic_segmentation/README.md) | Per-pixel scene segmentation |
| [`zero_shot_classification`](zero_shot_classification/README.md) | Open-vocabulary classification without retraining |

---

## Running an Application

Each application has its own arguments and usage. Refer to the individual README linked in the [Application Reference](#application-reference) table for full details.

From `hailo_apps/cpp`, run a binary with `--help` to see its arguments:

```bash
./object_detection/build/object_detection --help
./classification/build/classifier --help
```

---

## Input Sources

| Input | Argument | Notes |
|---|---|---|
| Image file | `--input bus.jpg` | JPEG, PNG, BMP supported |
| Video file | `--input video.mp4` | Any format OpenCV can decode |
| Directory | `--input images/` | Processes all images in the directory |
| USB camera | `--input usb` | Auto-selects first USB camera |
| Specific camera | `--input /dev/video2` | Linux only |
| Raspberry Pi camera | `--input rpi` | Raspberry Pi only |
| CSI camera | `--input csi` | Yocto-based systems (e.g. Astrial/IMX8). ISP must be initialized first — see [Platform Notes](#platform-notes) |
| Camera index | `--input 1` | Windows camera index |
| Predefined resource | `--input bus` | Auto-downloaded from `resources_config.yaml` |

---

## Visualization Configuration

`object_detection` and `instance_segmentation` support a `config/visualization_config.yaml` file (copied next to the executable at build time) that controls how detections are rendered:

```yaml
visualization_params:
  score_thres: 0.42        # Minimum confidence to display a detection
  max_boxes_to_draw: 30    # Maximum detections drawn per frame
```


---

## Using Multiple Models on the Same Device

To run multiple models on the same Hailo device simultaneously, assign them the same `group_id`:

```cpp
std::string group_id = "my_group";
AsyncModelInfer model1("detector.hef", group_id);
AsyncModelInfer model2("classifier.hef", group_id);
```

Models in the same group share the device's virtual device context, improving resource utilization when running inference pipelines that involve more than one network.

---

## Platform Notes

### Astrial / NXP i.MX8MP

- **Video input format:** Input video files must be H.264-encoded with **YUV 4:2:0** chroma subsampling. The hardware video decoder (`v4l2h264dec`) on the i.MX8MP VPU only accepts this format. Videos encoded with 4:4:4 subsampling or other codecs will not decode correctly with hardware acceleration.

  To re-encode a video to the correct format:
  ```bash
  ffmpeg -i input.mp4 -c:v libx264 -pix_fmt yuv420p -profile:v baseline output.mp4
  ```

- **Video saving (`-s`):** When saving a processed video stream (`-s`/`--save-stream-output`), FPS may drop when decoding from a video file due to VPU contention between the hardware decoder and encoder. This does not affect camera input.

- **CSI camera (`--input csi`):** The Astrial exposes its CSI camera as a standard V4L2 device after the ISP is initialized. Before using `--input csi`, start the ISP on the device:

  ```bash
  cd /opt/imx8-isp/bin && ./run.sh -lm -c dual_imx219_1080p60 &
  ```

  The application will then auto-detect the first non-USB V4L2 capture device. If no device is found, a warning is printed with this command as a reminder. `--input csi` is only accepted on Yocto-based systems — it will error immediately on Windows or standard desktop Linux.

---

## Camera Tips

- On some systems, OpenCV may default to GStreamer for camera capture and print warnings. Force V4L2 instead:
  ```bash
  export OPENCV_VIDEOIO_PRIORITY_GSTREAMER=0
  export OPENCV_VIDEOIO_PRIORITY_V4L2=100
  ```

- If the camera returns a permission error:
  ```bash
  sudo chmod 777 /dev/video0
  ```

- Press **Q** in any display window to exit cleanly.

---

## Disclaimer

This software is provided by Hailo on an "AS IS" basis and "with all faults". No responsibility or liability is accepted regarding accuracy, merchantability, completeness, or suitability. These examples were validated on the specific versions listed above; behavior on other versions or environments is not guaranteed. For issues, please open a ticket in the repository's Issues tab.
