![Hailo Applications Infrastructure](doc/images/github_applications_infrastructure.png)

# Hailo Applications
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/hailo-ai/hailo-apps)

Build and deploy high-performance AI applications on edge devices with Hailo hardware accelerators. From real-time computer vision to **GenAI voice-to-action agents** - production-ready applications and reusable infrastructure to accelerate your AI development.

**Highlight:** Voice-to-action AI agents that understand natural language commands and control hardware in real-time - elevators, servos, LEDs, and more.

**Supports:** Hailo-8, Hailo-8L, and Hailo-10H accelerators

**Perfect for:** Raspberry Pi 5, x86_64 Ubuntu systems, and edge AI deployments.

## What You Get

✨ **20+ Ready-to-Run Applications**
- **🎯 GenAI - Voice to Action (Featured):** AI agents that convert natural language into hardware control - talk to your devices and make things happen
- **Computer Vision:** Object detection, pose estimation, instance segmentation, face recognition, depth estimation, OCR
- **More GenAI:** Voice assistants, VLM chat, speech recognition (Whisper)
- **Advanced:** Multi-camera tracking, zero-shot classification (CLIP), tiling for high-res processing

🚀 **Production-Ready Infrastructure**
- GStreamer-based pipelines for efficient video processing
- Modular Python library for rapid prototyping
- Hardware-optimized performance with Hailo accelerators

🛠️ **Developer-Friendly**
- Install in minutes with automated scripts
- Extensive documentation and examples
- Easy integration into your own projects

**Learn more:** [Hailo Official Website](https://hailo.ai/) | [Community Forum](https://community.hailo.ai/)

## Quick Start

### Installation
```bash
git clone https://github.com/hailo-ai/hailo-apps.git
cd hailo-apps
sudo ./install.sh
```

**Need help?** See the [Full Installation Guide](./doc/user_guide/installation.md) for hardware setup and troubleshooting.

### Run Your First Application
```bash
source setup_env.sh           # Activate environment
hailo-detect-simple           # Start object detection
```

![Detection Example](doc/images/detection.gif)

**Try more computer vision:**
```bash
hailo-pose                    # Pose estimation
hailo-seg                     # Instance segmentation
hailo-tiling                  # Tiling for high-res processing
hailo-depth                   # Depth estimation
```

![Pose Estimation](doc/images/pose_estimation.gif)
![Instance Segmentation](doc/images/instance_segmentation.gif)
![Depth Estimation](doc/images/depth.gif)

### GenAI Applications
```bash
# Voice Assistant
cd hailo_apps/python/gen_ai_apps/voice_assistant/
python voice_assistant.py

# Agent Tools - Voice to Action
python -m hailo_apps.python.gen_ai_apps.agent_tools_example.agent

# VLM Chat - Vision + Language
cd hailo_apps/python/gen_ai_apps/vlm_chat/
python vlm_chat.py
```

**📚 See all applications:** [Running Applications Guide](./doc/user_guide/running_applications.md)

## Documentation

**[📖 Complete Documentation](./doc/README.md)**

| Guide | What's Inside |
|-------|---------------|
| **[User Guide](./doc/user_guide/README.md)** | Installation, running apps, configuration, repository structure |
| **[Developer Guide](./doc/developer_guide/README.md)** | Build custom apps, write post-processing, model retraining |

## Choosing the Right App Type

This repository provides three types of applications, each suited for different development scenarios:

| App Type | Best For | Examples |
|----------|----------|----------|
| **Pipeline Apps** | Production-ready video processing with cameras, RTSP streams, and real-time inference | Object detection, pose estimation, instance segmentation, face recognition |
| **Standalone Apps** | Supports video and camera input, but specialized for image processing, model validation, and rapid prototyping | Lane detection, super resolution, OCR |
| **GenAI Apps** | Hailo-10H generative AI applications | Voice assistants, VLM chat, voice-to-action agents, Whisper speech recognition |

### 📹 Pipeline Apps (`hailo_apps/python/pipeline_apps/`)
Use pipeline apps when you need **real-time video processing** with:
- Raspberry Pi Camera, USB cameras, or video files
- RTSP streams for IP cameras
- Multi-camera and multi-source processing
- GStreamer-based efficient pipelines

### 🖼️ Standalone Apps (`hailo_apps/python/standalone_apps/`)
Use standalone apps when you need to:
- Test and validate individual models with images
- Prototype and experiment quickly
- Process single images or batches
- Debug and benchmark model performance

### 🤖 GenAI Apps (`hailo_apps/python/gen_ai_apps/`)
Use GenAI apps for **Hailo-10H** generative AI features:
- Voice assistants with speech recognition
- Vision-Language Models (VLM) chat
- Voice-to-action AI agents
- Whisper-based speech recognition

## Key Features

**🎯 Input Flexibility**
- Raspberry Pi Camera, USB cameras, video files, RTSP streams
- Multi-camera and multi-source processing

**⚡ Optimized Performance**
- Hardware-accelerated inference on Hailo devices
- Efficient GStreamer pipelines for real-time processing
- Minimal CPU overhead

**🧩 Modular & Extensible**
- Reusable Python library (`hailo_apps`)
- Custom model support with HEF files
- Easy integration into existing projects

**[→ Learn about the repository structure](./doc/user_guide/repository_structure.md)**

## Support & Community

💬 **[Hailo Community Forum](https://community.hailo.ai/)** - Get help, share projects, connect with other developers

🐛 **Issues?** Search the forum or open a discussion - the community and Hailo team are here to help!

---

**License:** MIT - see [LICENSE](LICENSE) for details