Paddle OCR
================

This example performs end-to-end text detection and recognition using the PaddleOCR pipeline accelerated by a **Hailo-8**, **Hailo-8L**, or **Hailo-10H** device.
It processes images, videos, folders, or camera input, detects text regions, and decodes the text using an OCR model.

Optionally, FPS performance can be shown, and output can be saved for each input.

![output example](./output.png)

Requirements
------------

- hailo_platform:
    - 4.23.0 (for Hailo-8 devices)
    - 5.1.1 (for Hailo-10H devices)
- loguru
- opencv-python
- paddlepaddle
- shapely
- pyclipper
- symspellpy

## Installation and Usage

Run this app in one of two ways:
1. Standalone installation in a clean virtual environment (no TAPPAS required) — see [Option 1](#option-1-standalone-installation)
2. From an installed `hailo-apps` repository — see [Option 2](#option-2-inside-an-installed-hailo-apps-repository)

## Option 1: Standalone Installation

To avoid compatibility issues, it's recommended to use a clean virtual environment.

0. Install PCIe driver and PyHailoRT
    - Download and install the PCIe driver and PyHailoRT from the Hailo website
    - To install the PyHailoRT whl:
    ```shell script
    pip install hailort-X.X.X-cpXX-cpXX-linux_x86_64.whl
    ```

1. Clone the repository:
    ```shell script
    git clone https://github.com/hailo-ai/hailo-apps.git
    cd hailo-apps/python/standalone_apps/paddle_ocr
    ```

2. Install dependencies:
    ```shell script
    pip install -r requirements.txt
    ```

## Option 2: Inside an Installed hailo-apps Repository
If you installed the full repository:
```shell script
git clone https://github.com/hailo-ai/hailo-apps.git
cd hailo-apps
sudo ./install.sh
source setup_env.sh
```
Then the app is already ready for usage:
```shell script
cd hailo-apps/python/standalone_apps/paddle_ocr
```

## Run
After completing either installation option, run from the application folder:
```shell script
./paddle_ocr.py -n <det_hef> <ocr_hef> -i <input_path>
```

The output results will be saved under a folder named `output`, or in the directory specified by `--output-dir`.

Arguments
---------

- `-n, --net`: 
    - A **model name** (e.g., `yolov8n`) → the script will automatically download and resolve the correct HEF for your device.
    - A **file path** to a local HEF → the script will use the specified network directly.
- `-i, --input`:
  - An **input source** such as an image (`bus.jpg`), a video (`video.mp4`), a directory of images, or `camera` to use the system camera.
  - A **predefined input name** from `inputs.json` (e.g., `bus`, `street`).
    - If you choose a predefined name, the input will be **automatically downloaded** if it doesn't already exist.
  - Use `--list-inputs` to display all available predefined inputs.
- `-b, --batch-size`: [optional] Number of images in one batch. Defaults to 1.
- `-s, --save_stream_output`: [optional] Save the output of the inference from a stream.
- `-o, --output-dir`: [optional] Directory where output images/videos will be saved.
- `--show-fps`: [optional] Display FPS performance metrics for video/camera input.
- `--use-corrector`: [optional] Enable text correction after OCR (e.g., spelling or formatting fixes).
- `--camera-resolution`: [optional][Camera only] Input resolution: `sd` (640x480), `hd` (1280x720), or `fhd` (1920x1080).
- `--output-resolution`: [optional] Set output size using `sd|hd|fhd`, or pass custom width/height (e.g., `--output-resolution 1920 1080`).
- `-f, --framerate`: [optional][Camera only] Override the camera input framerate.
- `--list-nets` [optional] Print all supported networks for this application (from `networks.json`) and exit.
- `--list-inputs`: [optional] Print the available predefined input resources (images/videos) defined in `inputs.json` for this application, then exit.


### Environment Variables
- `CAMERA_INDEX`: [Camera input only] Select which camera index to use when -i camera is specified. Defaults to 0 if not set.
    - Example: `CAMERA_INDEX=1 ./paddle_ocr.py -n model.hef -i camera`


For more information:
```shell script
./paddle_ocr.py -h
```
Example 
-------

**List supported networks**
```shell script
./paddle_ocr.py --list-nets
```

**List available input resources**
```shell script
./paddle_ocr.py --list-inputs
```

**Inference on single image**
```shell script
./paddle_ocr.py -n ocr_det.hef ocr_model.hef -i ocr_img1.jpg
```

**Inference on a camera stream**
```shell script
./paddle_ocr.py -n ocr_det.hef ocr_model.hef -i camera
```

**Inference on a camera stream with custom frame rate**
```shell script
./paddle_ocr.py -n ocr_det.hef ocr_model.hef -i camera -f 20
```


Additional Notes
----------------
- Images are only supported in the following formats: .jpg, .jpeg, .png or .bmp
- Number of input images should be divisible by batch_size
- For any issues, open a post on the [Hailo Community](https://community.hailo.ai)

Disclaimer
----------
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.

This example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.
