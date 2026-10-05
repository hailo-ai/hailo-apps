## 26.10.0

### Fixed
- Corrected super-resolution scaling, English-only Whisper decoding, and LanceDB table clearing
- Fixed V2A tool selection, LLM context, wake-word recording, microphone compatibility, and device sharing
- Corrected Raspberry Pi camera colors and reduced Easter game display latency
- Aligned Model Zoo resources with HailoRT versions; added StereoNet download fallback and explicit download failures
- Fixed C++ directory inputs and explicit resource URLs; improved architecture and missing-ONNX errors

### Improved
- Installation: automatic privilege elevation, Docker/venv reuse, package detection, and compatible TAPPAS resources and bindings
- C++ builds: updated prerequisites, preferred system HailoRT, and skipped overlays with missing dependencies
- V2A: Raspberry Pi/Python 3.12 support and travel queries using home, work, current location, and nearby destinations
- C++ tests: reliable subprocess output, Qt environment handling, and device cleanup

### Documentation
- Updated installation, prerequisites, application, and V2A setup/troubleshooting guides
- Updated GenAI toolsets and added Cursor development rules

## 26.03.1

### Fixed
- Improved Raspberry Pi compatibility by supporting both legacy and new HailoRT package names
- Fixed H10 default Model Zoo version auto-resolution and added compatibility support for HailoRT 5.3.0
- Refactored USB/Raspberry Pi camera detection with earlier validation when required camera components are unavailable
- Strengthened installation prerequisite checks and version compatibility validation across supported architectures
- Removed deprecated resource scripts `get_inputs.sh` and `get_hef.sh`

## 26.03.0

### Added
- YOLO26 object detection and pose estimation, including the AI Gym example
- C++ ONNX Runtime pipeline for post-processing unsupported ONNX operations
- Voice2Action demo with speech recognition, tool selection, speech synthesis, and weather, travel, LED, system-check, and storage tools
- Agentic AI development framework (Beta) for GitHub Copilot, Claude Code, and Cursor
- Agent tools example, standalone speech recognition, and Easter Eggs game
- Instance-segmentation models with built-in NMS and simplified tiling with a 4-class HEF
- Windows support for GenAI, standalone Python, and C++ applications

### Improved
- Resource defaults, downloads, input handling, and pipeline helpers
- Unified mirror/flip handling and fixed missing-framerate handling
- Voice assistant, VLM chat, and shared terminal/audio diagnostics
- C++ resource handling and standalone detection, segmentation, OCR, pose, lane, and super-resolution apps
- Test runners, agent testing, mirror tests, and application/developer documentation
