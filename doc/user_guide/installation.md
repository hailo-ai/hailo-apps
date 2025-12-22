# Hailo Software Installation Guide

This guide provides comprehensive instructions for installing the Hailo Application Infrastructure on both x86_64 Ubuntu systems and Raspberry Pi devices.

## Table of Contents

- [Hailo Software Installation Guide](#hailo-software-installation-guide)
  - [Table of Contents](#table-of-contents)
  - [Automated Installation (Recommended)](#automated-installation-recommended)
  - [Download Resources](#download-resources)
    - [Usage](#usage)
    - [Available Options](#available-options)
    - [App Groups](#app-groups)
    - [Examples](#examples)
    - [Features](#features)
  - [Installing Hailo Packages](#installing-hailo-packages)
    - [Overview](#overview)
    - [Usage](#usage-1)
      - [Common Options](#common-options)
    - [Examples](#examples-1)
      - [Install for Hailo-8 on Ubuntu 24.04](#install-for-hailo-8-on-ubuntu-2404)
      - [Download Only (No Installation)](#download-only-no-installation)
  - [Raspberry Pi Installation](#raspberry-pi-installation)
    - [Hardware Setup for RPi](#hardware-setup-for-rpi)
    - [Software Setup for RPi](#software-setup-for-rpi)
    - [Verification for RPi](#verification-for-rpi)
  - [Post-Installation Verification](#post-installation-verification)
  - [Uninstallation](#uninstallation)

---

## Automated Installation (Recommended)

This is the easiest and recommended way to get started on any supported platform. The script automatically detects your environment and installs the appropriate packages.
This script supports both x86_64 Ubuntu and Raspberry Pi.
On the Raspberry Pi, make sure you first install the HW and SW as described in the [Raspberry Pi Installation](#raspberry-pi-installation) section.


```bash
# 1. Clone the repository
git clone https://github.com/hailo-ai/hailo-apps.git
cd hailo-apps

# 2. Run the automated installation script
sudo ./install.sh
```

The installation script will:
1. Create a Python virtual environment (`venv_hailo_apps` by default).
2. Install all required system and Python dependencies.
3. Download necessary AI model files.
4. Configure the environment.

For more options, such as using a custom virtual environment name:
```bash
# Download additional recommended models (not just the default ones)
sudo ./install.sh --all
```


<details>
<summary><b>Manual Installation (Advanced)</b></summary>

If you need full control over the process use the following instructions.

The `hailo_installer.sh` script handles the installation of the HailoRT and Tappas Core libraries. The main `install.sh` script in the root directory will run this for you, but you can also run it manually for custom installations.

1. **HailoRT and TAPPAS-CORE Installation:**
```bash
sudo ./scripts/hailo_installer.sh
```
This installs the default versions of HailoRT and TAPPAS-CORE.
On the Raspberry Pi, use their apt server.
For additional versions, please visit the [Hailo Developer Zone](https://hailo.ai/developer-zone/).

2.  **Create & activate a virtual environment**
    ```bash
    python3 -m venv your_venv_name --system-site-packages
    source your_venv_name/bin/activate
    ```
We use system-site-packages to inherit python packages from the system.
On the Raspberry Pi, the hailoRT and TAPPAS-CORE python bindings are installed on the system. As part of hailo-all installation.
On the x86_64 Ubuntu, the hailoRT and TAPPAS-CORE python bindings can be installed inside the virtual environment.
Note that also on the x86_64 Ubuntu, the gi library is installed on the system (apt install python3-gi python3-gi-cairo gir1.2-gtk-4.0). You can try installing using pip but it is not recommended.

3.  **Install Hailo Python packages**
    This script will install the HailoRT and TAPPAS-CORE python bindings.
    ```bash
    ./scripts/hailo_python_installation.sh
    ```
4.  **Install repository**
    ```bash
    pip install --upgrade pip
    pip install -e .
    ```
5.  **Run post-install setup**
    This downloads models and configures the environment.
    ```bash
    hailo-post-install
    ```

</details>

---
## Download Resources

The Hailo Apps Infrastructure includes a robust resource downloader utility that automatically fetches AI models, configuration files, and test videos optimized for your Hailo hardware. The tool supports parallel downloads, automatic retry on failures, and file integrity validation.

### Usage

```bash
hailo-download-resources [OPTIONS]
```

### Available Options

| Option | Description |
|--------|-------------|
| `--all` | Download all models (default + extra) for all apps |
| `--group <APP>` | Download resources for a specific app (e.g., `detection`, `vlm_chat`, `face_recognition`) |
| `--model <NAME>` | Download a specific model by name |
| `--arch <ARCH>` | Force a specific Hailo architecture: `hailo8`, `hailo8l`, or `hailo10h`. Auto-detected if not specified |
| `--config <PATH>` | Use a custom resources configuration file |
| `--list-models` | List all available models for the detected/selected architecture |
| `--dry-run` | Preview what would be downloaded without actually downloading |
| `--force` | Force re-download even if files already exist |
| `--no-parallel` | Disable parallel downloads (download sequentially) |
| `--include-gen-ai` | Include gen-ai apps (VLM, LLM, Whisper) in bulk downloads |

### App Groups

Resources are organized by application. Each app has models optimized for different architectures:

| App | Description | Architectures |
|-----|-------------|---------------|
| `detection` | Object detection (YOLOv8, YOLOv11) | hailo8, hailo8l, hailo10h |
| `pose_estimation` | Human pose estimation | hailo8, hailo8l, hailo10h |
| `instance_segmentation` | Instance segmentation | hailo8, hailo8l, hailo10h |
| `face_recognition` | Face detection and recognition | hailo8, hailo8l, hailo10h |
| `depth` | Monocular depth estimation | hailo8, hailo8l, hailo10h |
| `clip` | Zero-shot image classification | hailo8, hailo8l, hailo10h |
| `tiling` | High-resolution tiled detection | hailo8, hailo8l, hailo10h |
| `vlm_chat` | Vision-Language Model (Qwen2-VL) | hailo10h only |
| `llm_chat` | Large Language Model (Qwen2.5) | hailo10h only |
| `whisper_chat` | Speech-to-text (Whisper) | hailo10h only |

> **Note:** Gen-AI apps (`vlm_chat`, `llm_chat`, `whisper_chat`) are only available on Hailo-10H hardware.

By default, we download models optimized for real-time frame rates on your device. Larger models with higher accuracy can be downloaded using `--all`. Additional models are available from the [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo).

### Examples

```bash
# Download default resources for your detected hardware
hailo-download-resources

# Download all models (default + extra) for all apps
hailo-download-resources --all

# Download resources for a specific app
hailo-download-resources --group detection

# Download a specific model
hailo-download-resources --model yolov8m

# Download for a specific architecture
hailo-download-resources --arch hailo10h

# Preview what would be downloaded (dry run)
hailo-download-resources --dry-run

# Force re-download existing files
hailo-download-resources --force

# Download gen-ai app (auto-includes gen-ai models)
hailo-download-resources --group vlm_chat --arch hailo10h

# Include gen-ai apps in bulk download
hailo-download-resources --all --include-gen-ai --arch hailo10h

# List all available models for your architecture
hailo-download-resources --list-models
```

### Features

- **Parallel Downloads**: Downloads multiple files simultaneously for faster completion
- **Automatic Retry**: Retries failed downloads with exponential backoff (3 attempts by default)
- **File Validation**: Compares file sizes to detect corrupted or partial downloads
- **Atomic Operations**: Uses temporary files to prevent incomplete downloads from being saved
- **Architecture Awareness**: Automatically downloads models appropriate for your Hailo hardware

The downloader organizes resources into `/usr/local/hailo/resources/`, with models separated by architecture (`models/hailo8/`, `models/hailo10h/`, etc.) and videos/configs in dedicated subdirectories.

---


## Installing Hailo Packages

This section explains how to install all necessary **Hailo runtime components** using the automated installer script `scripts/hailo_installer.sh`.
The script supports both **Hailo-8** and **Hailo-10** architectures and can either **download** or **install** all required `.deb` and `.whl` packages from the official Hailo Debian server.

* If running **Raspberry Pi OS**, follow the **official installation guide** on
  [raspberrypi.com/documentation/computers/ai.html](https://www.raspberrypi.com/documentation/computers/ai.html).
* If running **Ubuntu on RPi**, use this installer script.

### Overview

The installer performs the following tasks automatically:

* Detects your **system architecture** (`x86_64`, `aarch64`, or Raspberry Pi).
* Validates your **Python version** (supported: `3.10 – 3.13`).
* Checks **kernel compatibility** and warns if not officially supported.
* Downloads and installs:

  * `hailort` PCIe driver and runtime `.deb` packages
  * `hailo-tappas-core` `.deb` package
  * `hailort` Python wheel
  * `hailo_tappas_core_python_binding` wheel

The packages are fetched from:

```
http://dev-public.hailo.ai/<date>/<Hailo8|Hailo10>/
```

### Usage

```bash
chmod +x scripts/hailo_installer.sh
sudo ./scripts/hailo_installer.sh [options]
```

#### Common Options

| Option                      | Description                                                                       |                                     |
| --------------------------- | --------------------------------------------------------------------------------- | ----------------------------------- |
| `--hw-arch=`           | hailo10h,hailo8                                                                         | Target hardware platform. Required. |
| `--venv-name=NAME`          | Name of the Python virtual environment (default: `hailo_venv`).                   |                                     |
| `--download-only`           | Only download the packages without installing them.                               |                                     |
| `--output-dir=DIR`          | Change where packages are saved (default: `/usr/local/hailo/resources/packages`). |                                     |
| `--py-tag=TAG`              | Manually specify Python wheel tag (e.g., `cp311-cp311`).                          |                                     |
| `-h                         | --help`                                                                           | Show help menu.                     |

### Examples

#### Install for Hailo-8 on Ubuntu 24.04

```bash
sudo ./scripts/hailo_installer.sh --hw-arch=hailo8
```

#### Download Only (No Installation)

```bash
./scripts/hailo_installer.sh --hw-arch=hailo10h --download-only
```

Packages will be saved under:

```
/usr/local/hailo/resources/packages/<hailo8|hailo10h>/
```

---

## Raspberry Pi Installation

These instructions are for setting up a Raspberry Pi 5 with a Hailo AI accelerator.

### Hardware Setup for RPi

1.  **Required Hardware**:
    *   Raspberry Pi 5 (8GB recommended) with Active Cooler.
    *   A Hailo accelerator:
        *   **Raspberry Pi AI Kit**: M.2 HAT + Hailo-8L/Hailo-8 Module.
        *   **Raspberry Pi AI HAT+**: A board with an integrated Hailo accelerator (13 or 26 TOPs).
    *   A 27W USB-C Power Supply.

2.  **Assembly**:
    *   **For AI Kit**: Follow the [Raspberry Pi's official AI Kit Guide](https://www.raspberrypi.com/documentation/accessories/ai-kit.html#ai-kit).
        *   Ensure a thermal pad is placed between the M.2 module and the HAT.
        *   Ensure the GPIO header is connected for stable operation.
    *   **For AI HAT+**: Follow the [Raspberry Pi's official AI HAT+ Guide](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html#ai-hat-plus).
        *   Ensure the GPIO header is connected for stable operation.

### Software Setup for RPi

1.  **Install Raspberry Pi OS**:
    *   Use the Raspberry Pi Imager to install the latest version of Raspberry Pi OS from [here](https://www.raspberrypi.com/software/).

2.  **Install Hailo Software**:
    *   The official Raspberry Pi AI stack includes the Hailo firmware and runtime. Follow the [Raspberry Pi's official AI Software Guide](https://www.raspberrypi.com/documentation/computers/ai.html#getting-started).

3.  **Enable PCIe Gen3 for Optimal Performance**:
    *   This is required for the M.2 HAT to achieve full performance. The AI HAT+ should configure this automatically if the GPIO is connected.
    *   Open the configuration tool: `sudo raspi-config`
    *   Go to `6 Advanced Options` -> `A8 PCIe Speed`.
    *   Choose `Yes` to enable PCIe Gen 3 mode.
    *   Reboot the Raspberry Pi: `sudo reboot`.

### Verification for RPi

1.  **Check if the Hailo chip is recognized**:
    ```bash
    hailortcli fw-control identify
    ```
    This should show your board details (e.g., Board Name: Hailo-8). If not, see the troubleshooting section.

2.  **Check GStreamer plugins**:
    *   Verify `hailotools`: `gst-inspect-1.0 hailotools`
    *   Verify `hailo` (inference element): `gst-inspect-1.0 hailo`
    *   If a plugin is not found, you may need to clear the GStreamer cache: `rm ~/.cache/gstreamer-1.0/registry.aarch64.bin` and reboot.

---

## Post-Installation Verification

After running any of the installation methods, you can verify that everything is working correctly.

1.  **Activate your environment**
    ```bash
    source venv_hailo_apps/bin/activate
    # or simply run the helper each session
    source setup_env.sh
    ```
2.  **Check installed Hailo packages**
    ```bash
    pip list | grep hailo
    # You should see packages like hailort, hailo-tappas-core, and hailo-apps.

    apt list | grep hailo
    # This shows all installed Hailo-related system packages.
    ```
3.  **Verify the Hailo device connection**
    ```bash
    hailortcli fw-control identify
    ```
4.  **Run a demo application**
    ```bash
    hailo-detect-simple
    ```
    A video window with live detections should appear.

<details>
<summary><b>Troubleshooting Tips</b></summary>

*   **PCIe Issues (RPi)**: If `lspci | grep Hailo` shows no device, check your M.2 HAT or AI HAT+ connections, power supply, and ensure PCIe is enabled in `raspi-config`.
*   **Driver Issues (RPi)**: If you see driver errors, ensure your kernel is up to date (`sudo apt update && sudo apt full-upgrade`).
*   **`DEVICE_IN_USE()` Error**: This means the Hailo device is being used by another process. Run the cleanup script: `./scripts/kill_first_hailo.sh`.
*   **GStreamer `cannot allocate memory in static TLS block` (RPi)**: This is a known issue. Add `export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1` to your `~/.bashrc` file and reboot.
*   **Emoji Display Issues (RPi)**: If emoji icons (❌, ✅, etc.) are not displaying correctly in terminal output, install the Noto Color Emoji font:
    ```bash
    sudo apt-get update
    sudo apt-get install fonts-noto-color-emoji
    fc-cache -f -v
    ```
    After installation, restart your terminal or log out and back in. If emojis still don't display, ensure your locale supports UTF-8:
    ```bash
    export LANG=en_US.UTF-8
    export LC_ALL=en_US.UTF-8
    ```

</details>

---

## Uninstallation

To remove the environment and downloaded resources:

```bash
# Deactivate the virtual environment if active
deactivate

# Delete project files and logs
sudo rm -rf venv_hailo_apps/ resources/ hailort.log hailo_apps.egg-info
```
To uninstall system packages, use `apt remove`.