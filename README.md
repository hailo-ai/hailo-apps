![Hailo Apps Banner](doc/images/banner.png)

# Hailo-Apps

High performance AI applications for Hailo accelerators, including GStreamer pipelines, GenAI assistants, and standalone C++/Python apps.

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/hailo-ai/hailo-apps)

## Supported Platforms and Devices
| Platforms | Accelerators |
|---|---|
| ![Raspberry Pi](https://img.shields.io/badge/Raspberry-Pi%205-red?logo=raspberrypi&logoColor=white) ![Ubuntu](https://img.shields.io/badge/Ubuntu-x86__64-E95420?logo=ubuntu&logoColor=white) ![Windows](https://img.shields.io/badge/Windows-blue?logo=windows&logoColor=white) | ![Hailo-8](https://img.shields.io/badge/Hailo-8-00A4EF?logoColor=white) ![Hailo-8L](https://img.shields.io/badge/Hailo-8L-00A4EF?logoColor=white) ![Hailo-10H](https://img.shields.io/badge/Hailo-10H-00A4EF?logoColor=white) |

## AI-Powered Development (Beta)

Use AI coding agents to quickly create Hailo applications. Just describe your idea and the agent builds, validates, and runs it for you.

 Supports VLM, LLM, pipeline, and standalone app types across all Hailo accelerators. **[Get started →](./doc/user_guide/agentic_development.md)**

🎮 Try out our [Easter Eggs game](hailo_apps/python/pipeline_apps/easter_game/), built autonomously by AI.

<img src="doc/images/agentic_ai.gif" width="600"/>


## Applications

30+ ready-to-run applications:

| Type | Best For | Location |
|------|----------|----------|
| **GenAI Apps** | LLM/VLM/speech workflows on Hailo-10H | `hailo_apps/python/gen_ai_apps/` |
| **Pipeline Apps** | Real-time camera/RTSP/video processing | `hailo_apps/python/pipeline_apps/` |
| **Standalone Apps** | HailoRT learning and minimal per-app installs | `hailo_apps/python/standalone_apps/` + `hailo_apps/cpp/` |

[All Applications](./doc/user_guide/running_applications.md)

### New in v26.03.0
Windows support, YOLO26 models, Voice2Action demo, AI-powered agentic development, and more.

[Full changelog →](./changelog.md)

## Requirements

All packages must be installed **before** running `install.sh`. Download from the [Hailo Developer Zone](https://hailo.ai/developer-zone/).

| Package | Type | Required For |
|---|---|---|
| HailoRT PCIe Driver | .deb | All apps |
| HailoRT | .deb | All apps |
| GStreamer Runtime (TAPPAS Core) | .deb | GStreamer pipeline apps |
| HailoRT Python Binding | .whl | All Python apps |
| GStreamer Python Binding (TAPPAS Core) | .whl | GStreamer pipeline apps |

> **Tip:** Standalone and gen-ai apps do **not** require the GStreamer packages above. Use `--skip-gstreamer` with `install.sh` to skip them.

[Full installation Guide](./doc/user_guide/installation.md)

## Quick Start
> **💡 Tip:** Standalone apps can be installed and run independently, they do **not** require GStreamer (`hailo-tappas-core`) or installing the full Hailo-Apps repository.

### Install Hailo-Apps

```bash
git clone https://github.com/hailo-ai/hailo-apps.git
cd hailo-apps
sudo ./install.sh
```

### Upgrading

When upgrading to a new version, use `--force-cleanup` to remove stale build artifacts before reinstalling:

```bash
git pull
sudo ./install.sh --force-cleanup
```

### Quick Examples

```bash
source setup_env.sh           # Activate environment
hailo-detect-simple           # Object detection
hailo-pose                    # Pose estimation
hailo-seg                     # Instance segmentation
hailo-depth                   # Depth estimation
hailo-tiling                  # Tiling for high-res processing
```

| Detection | Pose Estimation | Instance Segmentation | Depth Estimation |
|---|---|---|---|
| <img src="doc/images/detection.gif" width="200"/> | <img src="doc/images/pose_estimation.gif" width="200"/> | <img src="doc/images/instance_segmentation.gif" width="200"/> | <img src="doc/images/depth.gif" width="200"/> |

## Documentation

**[📖 Complete Documentation](./doc/README.md)**

| Guide | What's Inside |
|-------|---------------|
| **[User Guide](./doc/user_guide/README.md)** | Installation, running apps, configuration, repository structure |
| **[Developer Guide](./doc/developer_guide/README.md)** | Build custom apps, write post-processing, model retraining |

## Support

💬 [Hailo Community Forum](https://community.hailo.ai/)

**License:** MIT - see [LICENSE](LICENSE)