# Hailo-Apps Installation Guide

## Before you begin

Complete the [Prerequisites](./prerequisites.md) for your platform. This sets up HailoRT, the device driver, and any platform specific components required by Hailo Apps.

## Clone Hailo-Apps

The Hailo Apps repository is required for all installation methods:

```bash
git clone https://github.com/hailo-ai/hailo-apps.git
cd hailo-apps
```

## Choose an installation type

Choose the installation method based on whether you want a shared Hailo Apps installation, a dedicated setup for a single Python app, or a C++ build:

| Installation type                                                 | Use for                             | Platforms                     |
| ----------------------------------------------------------------- | ----------------------------------- | ----------------------------- |
| [Shared Hailo Apps installation](#shared-hailo-apps-installation) | GStreamer, standalone Python, GenAI | Ubuntu x86_64, Raspberry Pi 5 |
| [Per-app Python installation](#per-app-python-installation)       | Standalone Python, GenAI            | Linux, Windows                |
| [C++ app installation](#c-app-installation)                       | C++ apps                            | Linux, Windows                |

> GStreamer pipeline apps are only supported by the shared Hailo Apps installation.

---

## Shared Hailo Apps installation

Use this option to install a single Hailo Apps environment used by multiple apps.

It supports:

* GStreamer pipeline apps
* Python standalone apps
* GenAI apps

For **Hailo AI Software Suite Docker** and **Raspberry Pi 5**:

```bash
sudo ./install.sh
```

For a **clean Python environment on Ubuntu x86_64**, provide the PyHailoRT wheel downloaded during the prerequisite setup:

```bash
sudo ./install.sh --pyhailort /path/to/hailort-*.whl
```

Add `--skip-gstreamer` if you only need Python standalone or GenAI apps.

### Activate the environment

Activate it in **every new terminal session** before running apps:

```bash
source setup_env.sh
```

---

## Per-app Python installation

Use this option to run a single **Python standalone** or **GenAI** app without installing the shared Hailo Apps environment.

The example below uses the `object_detection` standalone app.

### Ubuntu x86_64

For a **clean Python environment on Ubuntu x86_64**, create a virtual environment:

```bash
cd hailo_apps/python/standalone_apps/object_detection

python3 -m venv .venv
source .venv/bin/activate
```

Install the PyHailoRT wheel downloaded during the prerequisite setup, followed by the app dependencies:

```bash
pip install /path/to/hailort-*.whl
pip install -r requirements.txt
```

When using the **Hailo AI Software Suite Docker**, PyHailoRT is already available in the container, so a separate virtual environment is not required. Install only the app specific dependencies:

```bash
cd hailo_apps/python/standalone_apps/object_detection

pip install -r requirements.txt
```

Run the app:

```bash
./object_detection.py -n yolov8n -i usb
```

### Windows

Create a virtual environment:

```powershell
cd hailo_apps\python\standalone_apps\object_detection

python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the PyHailoRT wheel provided by the HailoRT Windows installation and the app dependencies:

```powershell
pip install "C:\Program Files\HailoRT\python\hailort-*.whl"
pip install -r requirements.txt
```

Run the app:

```powershell
python .\object_detection.py -n yolov8n -i 0
```

### Raspberry Pi 5

Create a virtual environment with access to the Hailo packages installed during the prerequisite setup:

```bash
cd hailo_apps/python/standalone_apps/object_detection

python3 -m venv --system-site-packages .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Run the app:

```bash
./object_detection.py -n yolov8n -i rpi
```

Replace `object_detection` with the directory of the app you want to run.

Models are downloaded automatically on first use.

### GenAI apps

GenAI apps use the same per-app setup. Use the corresponding app directory under:

```text
hailo_apps/python/gen_ai_apps/<app>/
```

GenAI apps require **Hailo-10H**. See the app's README for any additional dependencies or run instructions.

---

## C++ app installation

C++ apps are located under:

```text
hailo_apps/cpp/<app>/
```

They do not require a Python environment.

The build uses system-installed `yaml-cpp` and `libcurl` if available:

```bash
sudo apt-get install libyaml-cpp-dev libcurl4-openssl-dev
```

Otherwise, initialize the submodules to build them from source instead:

```bash
git submodule update --init --recursive
```

### Linux

```bash
cd hailo_apps/cpp/object_detection
./build.sh
```

### Windows

```powershell
cd hailo_apps\cpp\object_detection
.\build.ps1
```

See each app's README for its dependencies and run instructions.

---

## Troubleshooting

**`DEVICE_IN_USE()`** — Another process is using the device. Close it or run:

```bash
./scripts/release_hailo.sh
```

**Device not detected on Raspberry Pi** — Check:

```bash
lspci | grep Hailo
```

If nothing is returned, check the HAT connection, power supply, and PCIe configuration.

**`cannot allocate memory in static TLS block` on Raspberry Pi** — Add the following to `~/.bashrc` and reboot:

```bash
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
```

**Hailo version mismatch** — Make sure the platform setup matches the supported Hailo component versions. See the [Prerequisites guide](./prerequisites.md).

---

## Upgrade

From the repository root:

```bash
git pull
sudo ./install.sh --force-cleanup
```

---

## Uninstall

For a shared Hailo Apps installation:

```bash
deactivate
sudo rm -rf venv_hailo_apps/ /usr/local/hailo
```

For a per-app installation, delete the app's `.venv` directory.
