# Prerequisites

> The most recent HailoRT version tested with Hailo Apps is v5.4.0.

Set up the Hailo software for your platform before installing Hailo Apps.

* [Ubuntu x86_64](#ubuntu-x86_64)
* [Windows](#windows)
* [Raspberry Pi 5](#raspberry-pi-5)

## Ubuntu x86_64

Choose the setup based on where you want Hailo Apps to run.

[**Hailo AI Software Suite Docker**](#hailo-ai-software-suite-docker) or the [**HailoRT Docker container**](#hailort-docker-container) are the simpler options if you want an isolated environment with HailoRT and PyHailoRT already available inside the container. The Suite Docker also includes the required GStreamer components.

[**Install HailoRT directly on Ubuntu**](#install-hailort-directly-on-ubuntu) if you want Hailo Apps to run directly on the host system.

### Install HailoRT directly on Ubuntu

Follow the **HailoRT installation instructions** in the [Hailo documentation](https://hailo.ai/developer-zone/documentation/?product=accelerators&device=hailo_8&category=sw) to install:

* HailoRT PCIe driver
* HailoRT
* PyHailoRT

Keep the downloaded PyHailoRT wheel, as it is required later when setting up a clean Python environment during the [Hailo Apps installation](./installation.md).

The wheel has a name similar to:

```text id="14etvz"
hailort-<version>-cp<py>-cp<py>-linux_x86_64.whl
```

### Hailo AI Software Suite Docker

Follow the **Hailo AI Software Suite installation instructions** in the [Hailo documentation](https://hailo.ai/developer-zone/documentation/?product=accelerators&device=hailo_8&category=sw).

The HailoRT PCIe driver must be installed on the host. HailoRT, PyHailoRT, and the required GStreamer components are already available inside the Suite container.

### HailoRT Docker container

Follow the **HailoRT Docker installation instructions** in the [Hailo documentation](https://hailo.ai/developer-zone/documentation/?product=accelerators&device=hailo_8&category=sw).

The HailoRT PCIe driver must be installed on the host. HailoRT and PyHailoRT are already available inside the container.

> Minimal Docker images (Suite Docker or HailoRT Docker container) may not include `git`, `curl`, or `wget`, which are required to clone the repository and download resources. Install them before proceeding:
>
> ```bash
> apt-get update && apt-get install -y git curl wget
> ```

## Windows

> Windows supports Python standalone, GenAI, and C++ apps. GStreamer pipeline apps are not supported.

Follow the **HailoRT Windows installation instructions** in the [Hailo documentation](https://hailo.ai/developer-zone/documentation/?product=accelerators&device=hailo_8&category=sw) and install the **HailoRT Windows MSI**.

During **Custom Setup**, make sure **PyHailoRT** is selected.

The PyHailoRT wheel can be found under:

```text id="pmmfpi"
C:\Program Files\HailoRT\python\hailort-*.whl
```

This wheel is used during the [per-app Python installation](./installation.md#per-app-python-installation).

## Raspberry Pi 5

Set up the AI Kit or AI HAT+ by following the [Raspberry Pi AI documentation](https://www.raspberrypi.com/documentation/computers/ai.html#getting-started).

The recommended setup uses the `hailo-all` package, which installs the HailoRT driver and runtime, GStreamer components, and Python bindings.

## Verify the installation

Confirm that the Hailo device is detected:

```bash id="k7z0g1"
hailortcli fw-control identify
```

Once the device is detected, continue to the [Hailo Apps Installation Guide](./installation.md).
