# Hailo Applications Documentation

Welcome to the complete documentation for the Hailo Applications repository. This page serves as the central hub for all guides and resources.

---

## Documentation Sections

### [📖 User Guide](./user_guide/README.md)
**For end-users.** This section is your go-to resource for getting the applications up and running. It includes:

*   **[Installation Guide](./user_guide/installation.md)**: Comprehensive instructions for installing the Hailo Application Infrastructure on supported platforms, including automated and manual installation steps.
*   **[Running Applications](./user_guide/running_applications.md)**: A detailed guide to using the pre-built AI applications, including examples for different input sources and a full command-line reference.
*   **[Repository Structure](./user_guide/repository_structure.md)**: An explanation of the project's directory structure.
*   **[Configuration](./user_guide/configuration.md)**: An explanation of the `config.yaml` file and how to customize system-wide settings.

### [👩‍💻 Developer Guide](./developer_guide/README.md)
**For developers and contributors.** This section provides the technical details you need to build on top of our framework. It includes:

*   **[Application Development](./developer_guide/app_development.md)**: The primary guide for developers. It covers the core concepts of the Python framework and how to build new applications.
*   **[GStreamer Helper Pipelines Reference](./developer_guide/gstreamer_helper_pipelines.md)**: Comprehensive reference documentation for all helper functions in the `gstreamer_helper_pipelines.py` module.
*   **[Writing a C++ Post-Process](./developer_guide/writing_postprocess.md)**: A step-by-step tutorial for creating custom C++ post-processing functions for new or unsupported neural networks.
*   **[Retraining Models](./developer_guide/retraining_example.md)**: A step-by-step tutorial for retraining models.
*   **[Debugging with GST Shark](./developer_guide/debugging_with_gst_shark.md)**: Debugging tool for GStreamer pipelines.

### ✔️ Applications Guide
**References for applications guides.**

#### Pipeline apps ####

*   **[Clip](../hailo_apps/python/pipeline_apps/clip/README.md)**
*   **[Depth](../hailo_apps/python/pipeline_apps/depth/README.md)**
*   **[Detection](../hailo_apps/python/pipeline_apps/detection/README.md)**
*   **[Detection simple](../hailo_apps/python/pipeline_apps/detection_simple/README.md)**
*   **[Face Recognition](../hailo_apps/python/pipeline_apps/face_recognition/README.md)**
*   **[Instance Segmentation](../hailo_apps/python/pipeline_apps/instance_segmentation/README.md)**
*   **[Multisource](../hailo_apps/python/pipeline_apps/multisource/README.md)**
*   **[Paddle OCR](../hailo_apps/python/pipeline_apps/paddle_ocr/README.md)** (BETA)
*   **[Pose Estimation](../hailo_apps/python/pipeline_apps/pose_estimation/README.md)**
*   **[REID Multisource](../hailo_apps/python/pipeline_apps/reid_multisource/README.md)**
*   **[Tiling](../hailo_apps/python/pipeline_apps/tiling/README.md)**

#### GenAI apps ####
*   **[GenAI Apps Overview](../hailo_apps/python/gen_ai_apps/README.md)**
*   **[Agent Tools Example](../hailo_apps/python/gen_ai_apps/agent_tools_example/README.md)**
*   **[VLM Chat](../hailo_apps/python/gen_ai_apps/vlm_chat/README.md)**
*   **[Voice Assistant](../hailo_apps/python/gen_ai_apps/voice_assistant/README.md)**
*   **[Simple LLM Chat](../hailo_apps/python/gen_ai_apps/simple_llm_chat/README.md)**
*   **[Simple VLM Chat](../hailo_apps/python/gen_ai_apps/simple_vlm_chat/README.md)**
*   **[Simple Whisper Chat](../hailo_apps/python/gen_ai_apps/simple_whisper_chat/README.md)**
*   **[Hailo Ollama](../hailo_apps/python/gen_ai_apps/hailo_ollama/README.md)**

#### Standalone apps ####
*   **[Instance Segmentation](../hailo_apps/python/standalone_apps/instance_segmentation/README.md)**
*   **[Lane Detection](../hailo_apps/python/standalone_apps/lane_detection/README.md)**
*   **[Object Detection](../hailo_apps/python/standalone_apps/object_detection/README.md)**
*   **[Paddle OCR](../hailo_apps/python/standalone_apps/paddle_ocr/README.md)**
*   **[Pose Estimation](../hailo_apps/python/standalone_apps/pose_estimation/README.md)**
*   **[Speech Recognition](../hailo_apps/python/standalone_apps/speech_recognition/README.md)**
*   **[Super Resolution](../hailo_apps/python/standalone_apps/super_resolution/README.md)**