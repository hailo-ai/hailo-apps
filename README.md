<img src="doc/images/banner.png" alt="Hailo Apps" width="100%">

| **Supported Hosts** | **Accelerators** |
| --- | --- |
| ![Raspberry Pi](https://img.shields.io/badge/Raspberry-Pi%205-red?logo=raspberrypi&logoColor=white) ![Ubuntu](https://img.shields.io/badge/Ubuntu-x86__64-E95420?logo=ubuntu&logoColor=white) ![Windows 10/11](https://img.shields.io/badge/Windows-10%20%2F%2011%2064--bit-0078D4?logo=windows&logoColor=white) | ![Hailo-8](https://img.shields.io/badge/Hailo-8-00A4EF?logoColor=white) ![Hailo-8L](https://img.shields.io/badge/Hailo-8L-00A4EF?logoColor=white) ![Hailo-10H](https://img.shields.io/badge/Hailo-10H-00A4EF?logoColor=white) |

## Getting Started

> **Setup → [Installation Guide](./doc/user_guide/installation.md)**  
> Follow the setup for your platform, then explore the applications below.

### Application Types

Hailo Apps includes 30+ ready-to-run applications across three categories:

| Type | Best For | Location |
| --- | --- | --- |
| **GenAI Apps** | LLM, VLM, and speech applications on Hailo-10H | `hailo_apps/python/gen_ai_apps/` |
| **GStreamer Pipeline Apps** | Real-time camera, RTSP, and video processing | `hailo_apps/python/pipeline_apps/` |
| **Standalone Apps** | Minimal Python and C++ examples using HailoRT directly | `hailo_apps/python/standalone_apps/` · `hailo_apps/cpp/` |

### Try an App

A few example GStreamer pipeline applications:

<table>
<tr>
<td align="center"><img src="doc/images/detection.gif" height="140"/><br><code>hailo-detect-simple</code></td>
<td align="center"><img src="doc/images/pose_estimation.gif" height="140"/><br><code>hailo-pose</code></td>
<td align="center"><img src="doc/images/instance_segmentation.gif" height="140"/><br><code>hailo-seg</code></td>
<td align="center"><img src="doc/images/depth.gif" height="140"/><br><code>hailo-depth</code></td>
</tr>
</table>

See **[Running Applications](./doc/user_guide/running_applications.md)** for the complete application list, supported inputs, and usage examples.

## AI-Powered Development (Beta)

Use AI coding agents to quickly create Hailo applications. Describe your idea and the agent builds, validates, and runs it for you.

Supports VLM, LLM, pipeline, and standalone app types across all Hailo accelerators.

**[Get started →](./doc/user_guide/agentic_development.md)**

## Documentation

**[Complete Documentation](./doc/README.md)** · **[User Guide](./doc/user_guide/README.md)**

| Guide | Use It For |
| --- | --- |
| **[Prerequisites](./doc/user_guide/prerequisites.md)** | Installing HailoRT and required platform components |
| **[Installation Guide](./doc/user_guide/installation.md)** | Choosing and completing the correct installation method |
| **[Running Applications](./doc/user_guide/running_applications.md)** | Running apps, input sources, and CLI options |
| **[Configuration](./doc/user_guide/configuration.md)** | Configuring application defaults |
| **[Developer Guide](./doc/developer_guide/README.md)** | Building custom GStreamer pipeline apps, post-processing, and model retraining |

### Ask DeepWiki

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/hailo-ai/hailo-apps)

Use DeepWiki to explore the repository and ask questions about the codebase.

## Support

💬 [Hailo Community Forum](https://community.hailo.ai/)

## License

This project is licensed under the [MIT License](LICENSE).