Pose Estimation
================

This example demonstrates pose estimation using a Hailo-8, Hailo-8L, or Hailo-10H device.<br>
The example takes an input, performs inference using the input HEF file and draws the detection boxes, class type, confidence, keypoints and joints connection on the resized image.  

Supported input formats include:
- Images: .jpg, .jpeg, .png, .bmp
- Video: .mp4
- Live camera feed


![output example](./output.gif)

Requirements
------------

- hailo_platform:
    - 4.23.0 (for Hailo-8 devices)
    - 5.1.1 (for Hailo-10H devices)
- loguru
- opencv-python

Supported Models
----------------

This example only supports pose estimation networks that allow HailoRT-Postprocess:
- yolov8m_pose
- yolov8s_pose

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
    cd hailo-apps/python/standalone_apps/pose_estimation
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
cd hailo-apps/python/standalone_apps/pose_estimation
```

## Run
After completing either installation option, run from the application folder:
```shell script
./pose_estimation.py -n <model_path> -i <input_path>
```

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
- `-b, --batch-size`: Number of images in one batch.
- `-cn, --class_num`: The number of classes the model is trained on. Defaults to 1.
- `-s, --save_stream_output`: [optional] Save the output of the inference from a stream.
- `-o, --output-dir`: [optional] Directory where output images/videos will be saved.
- `--show-fps`: [optional] Display FPS performance metrics for video/camera input.
- `--camera-resolution`: [optional][Camera only] Input resolution: `sd` (640x480), `hd` (1280x720), or `fhd` (1920x1080).
- `--output-resolution`: [optional] Set output size using `sd|hd|fhd`, or pass custom width/height (e.g., `--output-resolution 1920 1080`).
- `-f, --framerate`: [optional][Camera only] Override the camera input framerate.
- `--list-nets` [optional] Print all supported networks for this application (from `networks.json`) and exit.
- `--list-inputs`: [optional] Print the available predefined input resources (images/videos) defined in `inputs.json` for this application, then exit.


### Environment Variables
- `CAMERA_INDEX`: [Camera input only] Select which camera index to use when -i camera is specified. Defaults to 0 if not set.
    - Example: `CAMERA_INDEX=1 ./pose_estimation.py -n model.hef -i camera`


For more information:
```shell script
./pose_estimation.py -h
```

Example 
-------
**List supported networks**
```shell script
./pose_estimation.py --list-nets
```

**List available input resources**
```shell script
./pose_estimation.py --list-inputs
```

**Inference on single image**
```shell script
./pose_estimation.py -n yolov8s_pose.hef -i zidane.jpg -b 1
```

**Inference on a camera stream**
```shell script
./pose_estimation.py -n yolov8s_pose.hef -i camera
```

**Inference on a camera stream with custom frame rate**
```shell script
./pose_estimation.py -n yolov8s_pose.hef -i camera -f 20
```


Additional Notes
----------------

- The example was tested with:
    - HailoRT v4.23.0 (for Hailo-8)
    - HailoRT v5.1.1 (for Hailo-10H)
- The example expects a HEF which contains the HailoRT Postprocess
- The script assumes that the image is in one of the following formats: .jpg, .jpeg, .png or .bmp
- The annotated files will be saved in the `output` folder. 
- The number of input images should be divisible by the batch_size  
- The list of supported detection models is defined in `networks.json`.
- For any issues, open a post on the [Hailo Community](https://community.hailo.ai)

Disclaimer
----------
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.

This example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.