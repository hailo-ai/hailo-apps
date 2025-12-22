Lane Detection
===============================

> ⚠️ **Beta:** This application is currently in beta. Features and APIs may change.

This example demonstrates lane detection using a **Hailo8** or **Hailo10H** device.  
It receives an input video and annotates it with the lane detection coordinates.

![output GIF example](lane_det_output.gif)

Requirements
------------
- hailo_platform:
    - 4.23.0 (for Hailo-8 devices)
    - 5.1.1 (for Hailo-10H devices)
- loguru
- tqdm
- opencv-python



Supported Models
----------------
- ufld_v2_tu

## Installation and Usage

Run this app in one of two ways:
1. Standalone installation in a clean virtual environment (no TAPPAS required) — see [Option 1](#option-1-standalone-installation)
2. From an installed `hailo-apps` repository — see [Option 2](#option-2-inside-an-installed-hailo-apps-repository)

## Option 1: Standalone Installation

To avoid compatibility issues, it's recommended to use a clean virtual environment.

0. Install PyHailoRT
    - Download the HailoRT whl from the Hailo website - make sure to select the correct Python version. 
    - Install whl:
        ```shell script
        pip install hailort-X.X.X-cpXX-cpXX-linux_x86_64.whl
        ```

1. Clone the repository:
    ```shell script
    git clone https://github.com/hailo-ai/hailo-apps.git
    cd hailo-apps/python/standalone_apps/lane_detection
    ```

2. Install dependencies:
    ```shell script
    pip install -r requirements.txt
    ```

## Option 2: Inside an Installed hailo-apps Repository
If you installed the full repository:
```bash
git clone https://github.com/hailo-ai/hailo-apps.git
cd hailo-apps
sudo ./install.sh
source setup_env.sh
```
Then the app is already ready for usage:
```bash
cd hailo-apps/python/standalone_apps/lane_detection
```

## Run
After completing either installation option, run from the application folder:
```shell script
./lane_detection.py -n <model_path> -i <input_video_path> -o <output_path>
```

Arguments
---------

- `-n, --net`: 
    - A **model name** (e.g., `ufld_v2_tu`) → the script will automatically download and resolve the correct HEF for your device.
    - A **file path** to a local HEF → the script will use the specified network directly.
- `-i, --input`:
  - An **input source** such as an image (`bus.jpg`), a video (`video.mp4`), a directory of images, or `camera` to use the system camera.
  - A **predefined input name** from `inputs.json` (e.g., `bus`, `street`).
    - If you choose a predefined name, the input will be **automatically downloaded** if it doesn't already exist.
  - Use `--list-inputs` to display all available predefined inputs.
- `-o, --output`: Path to save the output video with annotated lanes.
- `--list-nets` Print all supported networks for this application (from `networks.json`) and exit.
- `--list-inputs`: Print the available predefined input resources (videos) defined in `inputs.json` for this application, then exit.

For more information:
```shell script
./lane_detection.py -h
```
Example 
-------

**List supported networks**
```shell script
./lane_detection.py --list-nets
```

**List available input resources**
```shell script
./lane_detection.py --list-inputs
```

**Inference**
```shell script
./lane_detection.py -n ./ufld_v2_tu.hef -i input_video.mp4
```

Additional Notes
----------------

- The example was only tested with:
    - 4.23.0 (for Hailo-8 devices)
    - 5.1.1 (for Hailo-10H devices) 
- The postprocessed video will be saved as **output_video.mp4**.  
- The list of supported detection models is defined in `networks.json`.
- For any issues, open a post on the [Hailo Community](https://community.hailo.ai)

Disclaimer
----------
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.

This example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.