# Hailo-Apps Installation Guide

Hailo-Apps supports four app types:

| App type                                            | Location                                   | Installation               | Platforms                       |
| --------------------------------------------------- | ------------------------------------------ | -------------------------- | ------------------------------- |
| [GStreamer pipeline apps](#gstreamer-pipeline-apps) | `hailo_apps/python/pipeline_apps/`         | Full repository            | Ubuntu x86_64, Raspberry Pi 5   |
| [Python standalone apps](#python-standalone-apps)   | `hailo_apps/python/standalone_apps/<app>/` | Full repository or per app | Linux, Windows                  |
| [GenAI apps](#genai-apps)                           | `hailo_apps/python/gen_ai_apps/<app>/`     | Full repository or per app | Linux, Windows (Hailo-10H only) |
| [C++ apps](#c-apps)                                 | `hailo_apps/cpp/<app>/`                    | Per app                    | Linux, Windows                  |

Before continuing, complete the setup for your platform in the [Prerequisites guide](./prerequisites.md).

* [GStreamer pipeline apps](#gstreamer-pipeline-apps)
* [Python standalone apps](#python-standalone-apps)
* [GenAI apps](#genai-apps)
* [C++ apps](#c-apps)
* [Troubleshooting](#troubleshooting)
* [Uninstall](#uninstall)

---

## GStreamer pipeline apps

GStreamer pipeline apps use the full Hailo-Apps installation and are supported on **Ubuntu x86_64** (including the Hailo AI Software Suite Docker) and **Raspberry Pi 5**.

`install.sh` sets up the Python environment, installs the required GStreamer components, compiles the post-processing libraries, and downloads the default models.

### Ubuntu x86_64 / Hailo AI Software Suite Docker

Clone the repository:

```bash
git clone https://github.com/hailo-ai/hailo-apps.git
cd hailo-apps
```

On a **clean Ubuntu installation**, provide the PyHailoRT wheel downloaded during the prerequisite setup:

```bash
sudo ./install.sh --pyhailort /path/to/hailort-*.whl
```

When using the **Hailo AI Software Suite Docker**, PyHailoRT is already available:

```bash
sudo ./install.sh
```

### Raspberry Pi 5

After completing the [Raspberry Pi prerequisites](./prerequisites.md#step1-rpi):

```bash
git clone https://github.com/hailo-ai/hailo-apps.git
cd hailo-apps

sudo ./install.sh
```

### Run

In every new terminal:

```bash
source setup_env.sh
```

For example:

```bash
hailo-detect-simple
```

### Models and resources

`install.sh` downloads the default resources automatically. **No additional download is required.**

Models are stored under `/usr/local/hailo/resources/`.

Optionally, use `hailo-download-resources` to manage resources:

| Option             | Description                       |
| ------------------ | --------------------------------- |
| `--group <app>`    | Download resources for one app    |
| `--all`            | Download all models               |
| `--include-gen-ai` | Include GenAI models with `--all` |
| `--list-models`    | List available models             |
| `--dry-run`        | Preview downloads                 |


---

## Python standalone apps

Standalone apps under `hailo_apps/python/standalone_apps/<app>/` do not require GStreamer.

### Shared Hailo-Apps environment

Clone the repository if you have not already:

```bash
git clone https://github.com/hailo-ai/hailo-apps.git
cd hailo-apps
```

To install Hailo-Apps for standalone apps only:

```bash
sudo ./install.sh --skip-gstreamer
```

If you already installed Hailo-Apps using `install.sh`, you can use the existing environment instead.

Activate the environment and run an app:

```bash
source setup_env.sh

cd hailo_apps/python/standalone_apps/object_detection
./object_detection.py -n yolov8n -i usb
```

Install any additional app-specific dependencies with:

```bash
pip install -r requirements.txt
```

### Per-app environment

If you only need one app, you can create a dedicated environment instead of installing the shared Hailo-Apps environment.


#### Ubuntu x86_64 / Hailo AI Software Suite Docker

```bash
git clone https://github.com/hailo-ai/hailo-apps.git
cd hailo-apps/hailo_apps/python/standalone_apps/object_detection

python3 -m venv .venv
source .venv/bin/activate
```

On a **clean Ubuntu installation**, install the PyHailoRT wheel downloaded during the prerequisite setup:

```bash
pip install /path/to/hailort-*.whl
```

Skip this step when using the **Hailo AI Software Suite Docker**.

Then:

```bash
pip install -r requirements.txt
./object_detection.py -n yolov8n -i usb
```

#### Windows

```powershell
git clone https://github.com/hailo-ai/hailo-apps.git
cd hailo-apps\hailo_apps\python\standalone_apps\object_detection

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install "C:\Program Files\HailoRT\python\hailort-*.whl"
pip install -r requirements.txt

python .\object_detection.py -n yolov8n -i 0
```

#### Raspberry Pi 5

After completing the [Raspberry Pi prerequisites](./prerequisites.md#step1-rpi):

```bash
git clone https://github.com/hailo-ai/hailo-apps.git
cd hailo-apps/hailo_apps/python/standalone_apps/object_detection

python3 -m venv --system-site-packages .venv
source .venv/bin/activate

pip install -r requirements.txt
./object_detection.py -n yolov8n -i rpi
```

Replace `object_detection` with the desired app directory.

Models are downloaded automatically on first use.

---

## GenAI apps

GenAI apps are located under `hailo_apps/python/gen_ai_apps/<app>/` and require **Hailo-10H**.

You can use either the shared Hailo-Apps environment or a dedicated environment, as described for [Python standalone apps](#python-standalone-apps).

From the repository root, install the additional GenAI dependencies:

```bash
pip install -e ".[gen-ai]"
```


---

## C++ apps

C++ apps are located under `hailo_apps/cpp/<app>/` and do not require a Python environment.

Clone the repository with its submodules:

```bash
git clone --recurse-submodules https://github.com/hailo-ai/hailo-apps.git
```

On Linux:

```bash
cd hailo-apps/hailo_apps/cpp/object_detection
./build.sh
```

On Windows:

```powershell
cd hailo-apps\hailo_apps\cpp\object_detection
.\build.ps1
```

See each app's README for its dependencies and run command.

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

## Uninstall

For a full Hailo-Apps installation:

```bash
deactivate
sudo rm -rf venv_hailo_apps/ /usr/local/hailo
```

For a standalone app, delete its `.venv` directory.

To remove HailoRT or other platform components, refer to the corresponding platform documentation.

## Upgrade

```bash
git pull
sudo ./install.sh --force-cleanup
```
