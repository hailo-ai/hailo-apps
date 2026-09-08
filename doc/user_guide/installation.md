# Hailo-Apps Installation Guide

Hailo-Apps offers two installation types:

| Installation type | Use it for | Installs | Platforms |
| --- | --- | --- | --- |
| **Full repo installation** (`install.sh`) | GStreamer pipeline apps (required), Python standalone apps (optional, sharing one environment) | Shared virtual environment, plus (optionally) TAPPAS Core, pipeline app dependencies, and resources | Ubuntu x86_64, Raspberry Pi 5 (including inside the Hailo AI Software Suite Docker container) |
| **Standalone installation** (per app folder) | One standalone app at a time, self-contained | That app's own virtual environment (Python) or CMake build (C++) | Linux and Windows |

- [Step 1: Install HailoRT](#step-1-install-hailort)
- [Step 2: Install the apps](#step-2-install-the-apps)
  - [2A. Full repo installation](#2a-full-repo-installation-linux-only)
  - [2B. Standalone installation](#2b-standalone-installation-per-app)
- [Step 3: Verify](#step-3-verify)
- [Uninstall](#uninstall)

## Step 1: Install HailoRT

<!-- tabs -->

<a id="step1-ubuntu"></a>

**Clean Ubuntu x86_64**

Download the packages from the [Hailo Developer Zone](https://hailo.ai/developer-zone/) and install the system packages:

```bash
sudo dpkg -i hailort-pcie-driver_<version>_all.deb
sudo dpkg -i hailort_<version>_amd64.deb
sudo dpkg -i hailo-tappas-core_<version>_amd64.deb   # pipeline apps only
```

Keep the two Python wheels for the next step:

- `hailort-<version>-cp<py>-cp<py>-linux_x86_64.whl`: required for all Python apps
- `hailo_tappas_core_python_binding-<version>-py3-none-any.whl`: pipeline apps only

<a id="step1-suite-docker"></a>

**Hailo AI Software Suite Docker**

Before installing the Suite, install the HailoRT PCIe driver on the host, as described in the [Hailo AI Software Suite documentation](https://hailo.ai/developer-zone/). TAPPAS Core is already included in the Suite container. (`install.sh` adds the few system packages the container is missing.)

<a id="step1-windows"></a>

**Windows**

Windows supports standalone apps only (Python and C++; no GStreamer pipeline apps).

1. Download and run the **HailoRT Windows MSI** from the [Hailo Developer Zone](https://hailo.ai/developer-zone/).
2. In *Custom Setup*, make sure **PyHailoRT** is selected.
3. After installation the Python wheel is at `C:\Program Files\HailoRT\python\hailort-*.whl`; you will install it in Step 2.

<a id="step1-rpi"></a>

**Raspberry Pi 5**

Set up the AI Kit / AI HAT+ hardware as described in the [Raspberry Pi AI documentation](https://www.raspberrypi.com/documentation/computers/ai.html#getting-started), then install everything from the Raspberry Pi apt server:

```bash
sudo apt update && sudo apt full-upgrade
sudo apt install hailo-all
sudo reboot
```

`hailo-all` installs the driver, HailoRT, TAPPAS Core and both Python bindings system-wide.

<!-- /tabs -->

Confirm the device is visible (Linux and Windows):

```bash
hailortcli fw-control identify
```

---

## Step 2: Install the apps

Use [2A](#2a-full-repo-installation-linux-only) for GStreamer pipeline apps (Linux only). Use [2B](#2b-standalone-installation-per-app) to install a single standalone app on its own (Linux or Windows).

### 2A. Full repo installation (Linux only)

One script sets up a virtual environment (`venv_hailo_apps`), installs the `hailo_apps` package and its dependencies, downloads the default models for your device, and (unless skipped) compiles the TAPPAS post-processing libraries.

<!-- tabs -->

<a id="2a-ubuntu"></a>

**Clean Ubuntu x86_64**

```bash
git clone https://github.com/hailo-ai/hailo-apps.git
cd hailo-apps
cp /path/to/hailort-*.whl /path/to/hailo_tappas_core_python_binding-*.whl .
sudo ./install.sh
```

`install.sh` installs any Hailo `.whl` files found in the repository root into the virtual environment. Only need standalone apps, not TAPPAS or pipeline apps? Skip the TAPPAS wheel and add `--no-tappas-required`:

```bash
sudo ./install.sh --no-tappas-required
```

<a id="2a-suite-docker"></a>

**Hailo AI Software Suite Docker**

```bash
git clone https://github.com/hailo-ai/hailo-apps.git
cd hailo-apps
sudo ./install.sh
```

<a id="2a-rpi"></a>

**Raspberry Pi 5**

```bash
git clone https://github.com/hailo-ai/hailo-apps.git
cd hailo-apps
sudo ./install.sh
```

<!-- /tabs -->

Then, in every new terminal:

```bash
source setup_env.sh
```

Run a pipeline app:

```bash
hailo-detect-simple      # a video window with live detections should appear
```

Run a Python standalone app from the same environment:

```bash
cd hailo_apps/python/standalone_apps/object_detection
./object_detection.py -n yolov8n -i usb
```

If an app's `requirements.txt` lists packages beyond the shared environment, install them into `venv_hailo_apps` first:

```bash
pip install -r requirements.txt
```

Models are stored under `/usr/local/hailo/resources/`. `install.sh` downloads the default model of every pipeline app; for more, use `hailo-download-resources`:

| Option | Does |
| --- | --- |
| `--group <app>` | Download resources for one app (e.g. `detection`, `vlm_chat`) |
| `--all` | Download every model for every app |
| `--include-gen-ai` | Include GenAI models (VLM/LLM/Whisper) in `--all`; not downloaded by default |
| `--list-models` / `--dry-run` | List or preview without downloading |

Installed via `pip install -e .` instead of `install.sh`? Run `hailo-post-install` once to download resources and compile the postprocess libraries (or `hailo-compile-postprocess` to just compile).

### 2B. Standalone installation (per app)

Already did a [full repo installation](#2a-ubuntu)? Your Python standalone apps are ready to run, no need for this section.

Use this to run a single app without setting up the rest of the repo. Only HailoRT and the PyHailoRT wheel are needed; TAPPAS is never required.

**Python apps** (`hailo_apps/python/standalone_apps/<app>/`)

<!-- tabs -->

<a id="2b-ubuntu"></a>

**Ubuntu x86_64 / Suite Docker**

```bash
git clone https://github.com/hailo-ai/hailo-apps.git
cd hailo-apps/hailo_apps/python/standalone_apps/object_detection
python3 -m venv .venv && source .venv/bin/activate
pip install /path/to/hailort-*.whl        # skip in the Suite Docker (already installed)
pip install -r requirements.txt
./object_detection.py -n yolov8n -i usb
```

<a id="2b-windows"></a>

**Windows (PowerShell)**

```powershell
git clone https://github.com/hailo-ai/hailo-apps.git
cd hailo-apps\hailo_apps\python\standalone_apps\object_detection
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install "C:\Program Files\HailoRT\python\hailort-*.whl"
pip install -r requirements.txt
python .\object_detection.py -n yolov8n -i 0
```

<a id="2b-rpi"></a>

**Raspberry Pi 5**

```bash
git clone https://github.com/hailo-ai/hailo-apps.git
cd hailo-apps/hailo_apps/python/standalone_apps/object_detection
python3 -m venv --system-site-packages .venv && source .venv/bin/activate   # reuses the PyHailoRT installed by hailo-all
pip install -r requirements.txt
./object_detection.py -n yolov8n -i rpi
```

<!-- /tabs -->

Replace `object_detection` with any app folder. Model files (`-n <model-name>`) are downloaded automatically on first use.

**GenAI apps** (`hailo_apps/python/gen_ai_apps/<app>/`, Hailo-10H only): follow the same steps as above; the extra dependencies are installed from the repository root:

```bash
pip install -e ".[gen-ai]"
```

**C++ apps** (`hailo_apps/cpp/<app>/`, Linux and Windows): no Python environment is needed, but the repository must be cloned with `--recurse-submodules` (pulls in bundled yaml-cpp and libcurl). Build and run from the app folder:

```bash
git clone --recurse-submodules https://github.com/hailo-ai/hailo-apps.git
cd hailo-apps/hailo_apps/cpp/object_detection
./build.sh          # Linux
```

```powershell
.\build.ps1         # Windows (PowerShell)
```

Each app's README lists its exact HailoRT version requirement, dependencies, and run command.

---

## Step 3: Verify

| Check | Command | Expected |
| --- | --- | --- |
| Device is detected | `hailortcli fw-control identify` | Board name, firmware and serial number are printed |
| Pipeline apps installed | `source setup_env.sh && hailo-detect-simple` | Live detection window |
| Standalone app runs | `./<app>.py -n <model> -i <input>` from the app folder | Annotated output in a window or in `output/` |

[Back to top](#hailo-apps-installation-guide)

**Common issues**

- **`DEVICE_IN_USE()`**: another process holds the device. Run `./scripts/release_hailo.sh` (pipeline install) or close the other application.
- **No device (Raspberry Pi)**: `lspci | grep Hailo` shows nothing: check the HAT connection and power supply, and make sure PCIe is enabled in `raspi-config`.
- **`cannot allocate memory in static TLS block` (Raspberry Pi)**: add `export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1` to `~/.bashrc` and reboot.
- **Hailo version mismatch**: `install.sh` stops if the installed HailoRT / TAPPAS versions aren't a valid combination (see [Step 1](#step-1-install-hailort)). Install matching versions from the Developer Zone.

---

## Uninstall

```bash
# hailo-apps only (pipeline install)
deactivate
sudo rm -rf venv_hailo_apps/ /usr/local/hailo

# HailoRT (Ubuntu)
sudo apt purge hailort hailort-pcie-driver hailo-tappas-core

# HailoRT (Raspberry Pi)
sudo apt purge hailo-all
```

Standalone apps: delete the app's `.venv` folder. Windows: uninstall HailoRT from *Apps & features*.

**Upgrading:** `git pull && sudo ./install.sh --force-cleanup` clears stale build caches and resources before reinstalling.
