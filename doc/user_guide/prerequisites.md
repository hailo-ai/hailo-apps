# Prerequisites

Before installing Hailo-Apps, set up the Hailo software for your platform.

<!-- tabs -->

<a id="step1-ubuntu"></a>

**Clean Ubuntu x86_64**

Download the following packages from the [Hailo Developer Zone](https://hailo.ai/developer-zone/):

* HailoRT PCIe driver (`.deb`)
* HailoRT (`.deb`)
* PyHailoRT (`.whl`)

Install the driver and HailoRT:

```bash id="6ggvdx"
sudo dpkg -i hailort-pcie-driver_<version>_all.deb
sudo dpkg -i hailort_<version>_amd64.deb
```

Keep the PyHailoRT wheel (`hailort-<version>-cp<py>-cp<py>-linux_x86_64.whl`) for the Hailo-Apps installation step.

> **GStreamer pipeline apps:** TAPPAS Core and its Python binding are installed automatically by `install.sh`. No manual installation is required. To use a custom PyTAPPAS wheel, pass `--pytappas /path/to/wheel` to `install.sh`.

<a id="step1-suite-docker"></a>

**Hailo AI Software Suite Docker**

Install the HailoRT PCIe driver on the host as described in the [Hailo AI Software Suite documentation](https://hailo.ai/developer-zone/).

HailoRT and the GStreamer components are already available inside the Suite container. `install.sh` installs any additional system packages required by Hailo-Apps.

<a id="step1-windows"></a>

**Windows**

> Windows supports Python standalone and C++ apps. GStreamer pipeline apps are not supported.

1. Download and run the **HailoRT Windows MSI** from the [Hailo Developer Zone](https://hailo.ai/developer-zone/).
2. In **Custom Setup**, make sure **PyHailoRT** is selected.
3. After installation, the Python wheel is available under:

   ```text
   C:\Program Files\HailoRT\python\hailort-*.whl
   ```

   Keep this wheel for the Python app installation.

<a id="step1-rpi"></a>

**Raspberry Pi 5**

Set up the AI Kit / AI HAT+ as described in the [Raspberry Pi AI documentation](https://www.raspberrypi.com/documentation/computers/ai.html#getting-started).

Then install the Hailo software stack:

```bash id="oh5d8e"
sudo apt update && sudo apt full-upgrade
sudo apt install hailo-all
sudo reboot
```

`hailo-all` installs the HailoRT driver, runtime, GStreamer components, and Python bindings. No additional Hailo packages need to be installed manually.

<!-- /tabs -->

## Verify the installation

Confirm that the Hailo device is detected:

```bash id="q2ytsj"
hailortcli fw-control identify
```

Once the device is detected, continue to the [Installation Guide](./installation.md#step-2-install-by-app-type).
