# Hailo Software Installation Guide

This guide provides instructions for installing the Hailo Application Infrastructure on both x86_64 Ubuntu systems and Raspberry Pi devices.

> **Prerequisites:** Before installing hailo-apps, you must have the Hailo runtime packages installed on your system. If you haven't installed them yet, see the [Installing Hailo Packages](#installing-hailo-packages-prerequisites) section first.

## Table of Contents

**Installing hailo-apps**
- [Automated Installation (Recommended)](#automated-installation-recommended)
  - [Download Resources](#download-resources)
- [Installing via pip](#installing-via-pip-for-integration-into-other-projects)
- [Manual Installation (Advanced)](#manual-installation-advanced)
- [Hailo Suite Docker Installation](#hailo-suite-docker-installation)
- [Post-Installation Verification](#post-installation-verification)
- [Uninstallation](#uninstallation)

**Installing Hailo Packages (Prerequisites)**
- [Installing Hailo Packages (Prerequisites)](#installing-hailo-packages-prerequisites)
  - [Raspberry Pi Installation](#raspberry-pi-installation)
  - [x86_64 Ubuntu Installation](#x86_64-ubuntu-installation)

---

# Installing hailo-apps

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

For more all options:
```bash
sudo ./install.sh --help
```

After installation completes, see [Post-Installation Verification](#post-installation-verification) to verify everything is working.

### Download Resources

The `install.sh` script automatically downloads AI models for your hardware. You can also use the `hailo-download-resources` command to download additional models or update existing ones.

```bash
hailo-download-resources [OPTIONS]
```

#### Available Options

| Option | Description |
|--------|-------------|
| `--all` | Download all models (default + extra) for all apps |
| `--group <APP>` | Download resources for a specific app (e.g., `detection`, `vlm_chat`, `face_recognition`) |
| `--model <NAME>` | Download a specific model by name |
| `--arch <ARCH>` | Force a specific Hailo architecture: `hailo8`, `hailo8l`, or `hailo10h`. Auto-detected if not specified |
| `--list-models` | List all available models for the detected/selected architecture |
| `--dry-run` | Preview what would be downloaded without actually downloading |
| `--force` | Force re-download even if files already exist |
| `--include-gen-ai` | Include gen-ai apps (VLM, LLM, Whisper) in bulk downloads |

#### App Groups

Resources are organized by application:

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

#### Examples

```bash
# Download default resources for your detected hardware
hailo-download-resources

# Download all models (default + extra) for all apps
hailo-download-resources --all

# Download resources for a specific app
hailo-download-resources --group detection

# Download for a specific architecture
hailo-download-resources --arch hailo10h

# List all available models for your architecture
hailo-download-resources --list-models
```

Resources are organized into `/usr/local/hailo/resources/`, with models separated by architecture (`models/hailo8/`, `models/hailo10h/`, etc.).

---

## Installing via pip (For Integration into Other Projects)

If you want to integrate hailo-apps into an existing Python project, you can install it directly via pip.

> **⚠️ Important: PyGObject & GStreamer**
> 
> hailo-apps requires PyGObject (Python bindings for GObject) to manage GStreamer pipelines.
> 
> **Do NOT install PyGObject via pip** - it requires system-level dependencies to build correctly.
> 
> Standard pip environments won't see system-installed PyGObject. You **must** create your virtual environment with access to system site-packages.

### Prerequisites

1. **Install System Dependencies:**
   ```bash
   # PyGObject and GStreamer bindings (required)
   sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-4.0
   
   # HailoRT and TAPPAS Core system packages
   # Download from Hailo Developer Zone: https://hailo.ai/developer-zone/
   ```

2. **Create a Virtual Environment with System Site-Packages:**
   ```bash
   python3 -m venv --system-site-packages my_hailo_env
   source my_hailo_env/bin/activate
   ```

### Installation Options

**Install from GitHub (latest):**
```bash
pip install git+https://github.com/hailo-ai/hailo-apps.git

# Refresh shell's command cache so new scripts are found
hash -r
```

**Install in editable mode (for development):**
```bash
git clone https://github.com/hailo-ai/hailo-apps.git
cd hailo-apps
pip install -e .

# Refresh shell's command cache
hash -r
```

### Setup Hailo Resources Directory

After pip install, create the Hailo resources directory where models and compiled libraries will be stored.

Create the directory with proper permissions (one-time setup):
```bash
sudo mkdir -p /usr/local/hailo/resources/packages
sudo chown -R $USER:$USER /usr/local/hailo
```

### Post-Installation Setup

After pip install, you must run the post-install command to complete the setup:

```bash
hailo-post-install
```

This command performs three essential steps:
1. **Downloads models and resources** to `/usr/local/hailo/resources/`
2. **Compiles the C++ postprocess shared libraries** (.so files required for GStreamer pipelines)
3. **Sets up environment configuration** (.env file)

> **Note:** By default, gen-ai models (VLM, LLM, Whisper) are **NOT** downloaded since they are very large. Use `--group vlm_chat` or `--all --include-gen-ai` to download them explicitly.

> **⚠️ Important:** If you skip this step, applications like `hailo-detect-simple` will fail with errors like:
> ```
> Could not load lib /usr/local/hailo/resources/so/libyolo_hailortpp_postprocess.so
> ```

**Options:**

| Command | What it does |
|---------|--------------|
| `hailo-post-install` | Downloads default models + compiles .so files (recommended) |
| `hailo-post-install --group detection` | Downloads only detection resources + compiles .so files |
| `hailo-post-install --skip-download` | Compiles .so files only (no downloads) |
| `hailo-post-install --skip-compile` | Downloads resources only (no compilation) |

**Standalone commands:**

| Command | What it does |
|---------|--------------|
| `hailo-download-resources --group detection` | Downloads resources only (does NOT compile .so files) |
| `hailo-compile-postprocess` | Compiles .so files only (does NOT download resources) |

After installation completes, see [Post-Installation Verification](#post-installation-verification) to verify everything is working.

---

## Manual Installation (Advanced)

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
    ./scripts/hailo_installer_python.sh hailo8
    ```
    Or for Hailo10:
    ```bash
    ./scripts/hailo_installer_python.sh hailo10h
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

After installation completes, see [Post-Installation Verification](#post-installation-verification) to verify everything is working.

---

## Hailo Suite Docker Installation

If you're running inside the **Hailo Software Suite Docker** container (available from the [Hailo Developer Zone](https://hailo.ai/developer-zone/)), HailoRT and TAPPAS Core are already pre-installed.

### Prerequisites for Docker

Run the following commands to install required dependencies:

```bash
# Update package lists
sudo apt-get update

# Install Python virtual environment support
sudo apt install -y python3-venv

# Install required utilities
sudo apt-get install -y software-properties-common gnupg

# Upgrade libstdc++6 (required for newer C++ features)
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt install -y --only-upgrade libstdc++6
```

### Installation in Docker

After installing the prerequisites, proceed with the standard installation:

```bash
# Clone the repository (if not already done)
git clone https://github.com/hailo-ai/hailo-apps.git
cd hailo-apps

# Run the automated installation script
sudo ./install.sh
```

> **Note:** The Hailo "Suite Docker" already has HailoRT and TAPPAS Core pre-installed. The `install.sh` script will detect this and skip those components.

After installation completes, see [Post-Installation Verification](#post-installation-verification) to verify everything is working.

---

# Installing Hailo Packages (Prerequisites)

Before running hailo-apps, you need to install the Hailo runtime packages. The installation method differs depending on your platform.

## Required Packages (5 files)

| Package | Type | Description |
|---------|------|-------------|
| `hailort-pcie-driver` | .deb | PCIe driver for Hailo devices |
| `hailort` | .deb | HailoRT runtime library |
| `hailo-tappas-core` | .deb | TAPPAS Core GStreamer plugins |
| `hailort` | .whl | HailoRT Python bindings |
| `hailo_tappas_core_python_binding` | .whl | TAPPAS Core Python bindings |

**Note: Hailo Model Zoo GenAI** Is required only for Hailo-10H & GenAI use cases, like Hailo-Ollama, more details [Hailo Model Zoo GenAI](/hailo_apps/python/gen_ai_apps/hailo_ollama/README.md)

> **Supported versions:**
> - **Hailo-8 / Hailo-8L:** HailoRT 4.23, TAPPAS Core 5.1.0
> - **Hailo-10H:** HailoRT 5.1.1 & 5.2.0, TAPPAS Core 5.1.0 & 5.2.0

---

## Raspberry Pi Installation

For Raspberry Pi 5 with a Hailo AI accelerator, use the official Raspberry Pi AI guide:

- **For AI Kit**: Follow the [Raspberry Pi AI Kit Guide](https://www.raspberrypi.com/documentation/accessories/ai-kit.html#ai-kit)
- **For AI HAT+ / HAT+ 2**: Follow the [Raspberry Pi AI HAT+ / HAT+ 2 Guide](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html#ai-hat-plus)
- Make sure to visit this page: [Raspberry Pi AI Software Guide](https://www.raspberrypi.com/documentation/computers/ai.html#getting-started)
- Install the latest Raspberry Pi OS: [Raspberry Pi Imager](https://www.raspberrypi.com/software/) 

---

## x86_64 Ubuntu Installation

For x86_64 Ubuntu systems, you have two options to install the Hailo packages:

### Option A: Automated Installer Script (Recommended)

The `install.sh` script handles everything automatically, including downloading and installing Hailo packages:

```bash
git clone https://github.com/hailo-ai/hailo-apps.git
cd hailo-apps
sudo ./install.sh
```

Or use the lower-level installer script directly:

```bash
# For Hailo-8/8L
sudo ./scripts/hailo_installer.sh hailo8

# For Hailo-10H
sudo ./scripts/hailo_installer.sh hailo10h
```

The script downloads and installs all 5 packages:
- `hailort-pcie-driver` (.deb)
- `hailort` (.deb)
- `hailo-tappas-core` (.deb)
- `hailort` Python wheel
- `hailo_tappas_core_python_binding` Python wheel

**Installer Options:**

| Option | Description |
|--------|-------------|
| `--hailort-version VER` | Override HailoRT version |
| `--tappas-core-version VER` | Override TAPPAS Core version |
| `--download-only` | Download packages without installing |
| `--output-dir DIR` | Change download location (default: `/usr/local/hailo/resources/packages`) |

### Option B: Manual Download from Hailo Developer Zone

If you need specific versions or offline installation:

1. **Download packages** from the [Hailo Developer Zone](https://hailo.ai/developer-zone/)

2. **Install system packages (.deb):**
   ```bash
   sudo dpkg -i hailort-pcie-driver_<version>_amd64.deb
   sudo dpkg -i hailort_<version>_amd64.deb
   sudo dpkg -i hailo-tappas-core_<version>_amd64.deb
   ```

3. **Install Python wheels (.whl):**
   ```bash
   pip install hailort-<version>-cp<pyver>-linux_x86_64.whl
   pip install hailo_tappas_core_python_binding-<version>-cp<pyver>-linux_x86_64.whl
   ```

### Verification

```bash
# Check if Hailo device is recognized
hailortcli fw-control identify

# Check installed packages
apt list --installed | grep hailo
pip list | grep hailo
```

---

## Post-Installation Verification

After running any of the installation methods, you can verify that everything is working correctly.

1.  **Activate your environment**

    Note: If installed via pip - there is no need for this step.
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

### Quick Uninstall (hailo-apps only)

To remove just the hailo-apps environment and downloaded resources:

```bash
# Deactivate the virtual environment if active
deactivate

# Delete project files and logs
sudo rm -rf venv_hailo_apps/ resources/ hailort.log hailo_apps.egg-info
```

### Complete Manual Uninstall

To completely remove all Hailo components from your system:

**1. Remove system packages:**
```bash
# List installed Hailo packages
sudo dpkg -l | grep hailo

# Remove them (replace with actual package names from above)
sudo apt purge hailort hailort-pcie-driver hailo-tappas-core
```

**2. Remove Python packages:**
```bash
# List installed Hailo Python packages
pip list | grep hailo

# Remove them (add --break-system-packages if required)
pip uninstall hailort hailo-tappas-core hailo-apps
```

**3. Remove hailo-apps resources directory:**
```bash
sudo rm -rf /usr/local/hailo
```

**4. Remove hailo-apps repository:**
```bash
sudo rm -rf /path/to/hailo-apps
```

**5. Delete all Hailo kernel modules:**
```bash
# Find and delete hailo*.ko and hailo*.ko.xz files
sudo find /lib/modules -type f \( -name 'hailo*.ko' -o -name 'hailo*.ko.xz' \) -print -delete
sudo rm -rf <list from above>
```

**6. Remove any empty hailo directories left behind:**
```bash
sudo find /lib/modules -type d -name 'hailo' -print -exec rm -rf {} +
sudo rm -rf <list from above>
```

**7. Recompute module dependency database:**
```bash
sudo depmod -a
```

**8. Update initramfs:**
```bash
sudo update-initramfs -u
```

**9. Remove any leftover configuration files:**
```bash
# Check for remaining files
sudo find /etc/ | grep hailo
sudo rm -rf <list from above>
```

**10. Reboot:**
```bash
sudo reboot now
```