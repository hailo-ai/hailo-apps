# Repository Structure Guide

This document provides an overview of the directory structure for the Hailo Applications repository, explaining the purpose of each key folder and clarifying which directories are tracked by git and which are generated or managed by scripts.

```
hailo-apps/
├── doc/                        # Comprehensive documentation (user & developer guides)
│   ├── user_guide/             # User-facing docs (installation, running apps, config, structure)
│   ├── developer_guide/        # Developer docs (app development, post-process, retraining)
│   └── images/                 # Documentation assets
├── hailo_apps/                 # Main AI application package (Python)
│   ├── config/                 # YAML configs used by installers and apps
│   ├── installation/           # Python-side installers and env helpers
│   ├── postprocess/            # C++ post-processing sources and builds
│   └── python/
│       ├── pipeline_apps/      # GStreamer-based CLI apps (hailo-detect, hailo-pose, etc.)
│       ├── gen_ai_apps/        # GenAI applications (voice assistant, VLM chat, LLM chat, etc.)
│       ├── standalone_apps/    # Other standalone Python applications (lane detection, super resolution, etc.)
│       └── core/               # Shared logic (common utils, gstreamer, trackers, postprocess)
├── scripts/                    # Shell installers/utilities (install, cleanup, set-env)
├── tests/                      # Pytest-based test suite
├── config/                     # Top-level configs referenced by installers
├── local_resources/            # Local demo assets (not tracked by git)
├── resources -> /usr/local/hailo/resources  # Symlink to shared models/videos store
├── venv_hailo_apps/            # Default virtual environment (created by install.sh)
├── install.sh                  # Main installation script
├── setup_env.sh                # Per-shell environment activation helper
├── pyproject.toml              # Python package configuration and console entrypoints
```

## Key Directories

### `doc/`
Contains all project documentation, including user guides, developer guides, and architectural overviews.

### `hailo_apps/`
Main Python package for AI applications. Contains:
- **`python/`**:
  - `pipeline_apps/`: GStreamer-based pipeline applications available as CLI commands (e.g., `hailo-detect`, `hailo-pose`, `hailo-seg`). These are production-ready applications that leverage GStreamer for efficient video processing.
  - `gen_ai_apps/`: Generative AI applications including:
    - Full applications: `voice_assistant/`, `agent_tools_example/`, `vlm_chat/`
    - Simple examples: `simple_llm_chat/`, `simple_vlm_chat/`, `simple_whisper_chat/`
    - Documentation: `hailo_ollama/` (Ollama integration guide)
    - Shared utilities: `gen_ai_utils/` (LLM utilities, voice processing components)
  - `standalone_apps/`: Other standalone Python applications (e.g., lane detection, super resolution, speech recognition). These applications demonstrate various computer vision capabilities and can be run directly with Python.
  - `core/`: Shared logic, utilities, and GStreamer integration for apps.
    - `common/`: Foundational utilities (installation, configuration, helpers, logging).
    - `gstreamer/`: Reusable GStreamer components and pipelines.
    - `cpp_postprocess/`: C++ post-processing modules for AI outputs.
    - `installation/`: Installation and environment setup utilities.

### `resources/`
After running the installation script, you will see a `resources` directory in the root of the project. This is a **symbolic link** (symlink) to a system-wide directory, `/usr/local/hailo/resources`.

- **What it is**: A shortcut to a central location for large files needed by the applications (models, videos, assets).
- **Why a symlink**: Avoids duplication of large files across projects. All Hailo applications can share a single pool of models and videos, saving disk space and simplifying resource management.
- **How it's created**: The post install script creates this symlink if it doesn't exist.

### `venv_hailo_apps/`
Python virtual environment for local development. Not tracked by git.

### `scripts/`
Shell scripts for installation, environment setup, and utilities. The main `install.sh` orchestrates many of these scripts.

---

For more details on each application or component, see the respective README files in their directories.