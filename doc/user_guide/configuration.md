# System Configuration

The Hailo Applications infrastructure uses a YAML configuration file to manage global settings for installation, model management, and hardware detection. While the default settings are suitable for most users, this file provides fine-grained control.

The configuration is typically managed by the installation scripts, but can be manually edited if needed.

## Configuration Options

Here is a breakdown of what each option in the configuration file controls.

```yaml
# HailoRT version configuration
hailort_version: "auto"  # Options: "auto" or a specific version like "5.1.1"

# TAPPAS framework version
tappas_version: "auto"   # Recommended: keep "auto" for detection

# Model zoo version for downloading models
model_zoo_version: "auto"  # Mapped automatically per architecture

# Hardware architecture detection
host_arch: "auto"        # Options: "rpi", "x86", "arm", or "auto"
hailo_arch: "auto"       # Options: "hailo8", "hailo8l", "hailo10h", or "auto"

# File paths and directories
resources_path: "resources"          # Symlink in repo root
virtual_env_name: "venv_hailo_apps"  # Default virtual environment created by install.sh
storage_dir: "hailo_temp_resources"  # Temporary directory for downloads

# TAPPAS configuration
tappas_postproc_path: "auto"  # The path to the post-processing libraries (auto from pkg-config)
```

## Configuration Tips

*   **Use "auto" where possible**: For `hailort_version`, `tappas_version`, `host_arch`, and `hailo_arch`, the `"auto"` setting allows the system to detect the correct configuration for your hardware and software setup. This is the safest and most reliable option.
*   **Modifying `resources_path`**: Only change this if you have a specific need to store the AI models and other resources in a non-default location.
*   **Model Zoo versioning**: Leave `model_zoo_version` as `"auto"`; the installer uses `model_zoo_mapping` to pick the right version for the detected architecture.
*   **Virtual environment name**: The default created by `install.sh` is `venv_hailo_apps`. Set `virtual_env_name` only if you need a different name.