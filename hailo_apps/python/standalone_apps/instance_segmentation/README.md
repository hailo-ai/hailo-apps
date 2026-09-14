Instance Segmentation
=====================
This example demonstrates instance segmentation using a Hailo-8, Hailo-8L, or Hailo-10H device.<br>
It processes input images, videos, or a camera stream, performs inference using the input HEF file, and overlays the segmentation masks, bounding boxes, class labels, and confidence scores on the resized output image.<br>  
Optionally, object tracking across frames can be enabled for video and camera streams.

![output example](instance_segmentation_example.gif)

Requirements
------------
- hailo_platform:
    - 4.23.0 (for Hailo-8 devices)
    - 5.3.0 (for Hailo-10H devices)
- opencv-python
- scipy
- lap
- cython_bbox


Supported Models
----------------
- yolov5*_seg
- yolov8*_seg.
- yolov5m_seg_with_nms
- fast_sam_s


## Linux Installation

Run this app in one of two ways:
1. Standalone installation in a clean virtual environment (no GStreamer required) — see [Option 1](#option-1-standalone-installation)
2. From an installed `hailo-apps` repository — see [Option 2](#option-2-inside-an-installed-hailo-apps-repository)


### Option 1: Standalone Installation

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
    cd hailo-apps/python/standalone_apps/instance_segmentation
    ```

2. Install dependencies:
    ```shell script
    pip install -r requirements.txt
    ```

### Option 2: Inside an Installed hailo-apps Repository
If you installed the full repository:
```shell script
git clone https://github.com/hailo-ai/hailo-apps.git
cd hailo-apps
sudo ./install.sh
source setup_env.sh
```
⚠️ **Note (Python 3.10 / 3.11)**: If you encounter `ModuleNotFoundError: distutils.msvccompiler`,
run `export SETUPTOOLS_USE_DISTUTILS=stdlib` **before** executing the application.


Then the app is already ready for usage:
```shell script
cd hailo-apps/python/standalone_apps/instance_segmentation
```



## Windows Installation

To avoid compatibility issues, it's recommended to use a clean virtual environment.

0. Install HailoRT (MSI) + PyHailoRT
    1. Download and install the **HailoRT Windows MSI** from the Hailo website.
    2. During the installation, make sure **PyHailoRT** is selected (in the MSI “Custom Setup” tree).
    3. After installation, the PyHailoRT wheel is located under:
       `C:\Program Files\HailoRT\python`

    4. Create and activate a virtual environment:
    ```powershell
    python -m venv wind_venv
    .\wind_venv\Scripts\Activate.ps1
    ```

    5. Install the PyHailoRT wheel from the MSI installation folder:
    ```powershell
    pip install "C:\Program Files\HailoRT\python\hailort-*.whl"
    ```

1. Clone the repository:
    ```powershell
    git clone https://github.com/hailo-ai/hailo-apps.git
    cd hailo-apps\hailo_apps\python\standalone_apps\instance_segmentation
    ```

2. Install dependencies:
    ```powershell
    pip install -r requirements.txt


## Run
After completing either installation option, run from the application folder:
```shell script
python .\instance_segmentation.py -n <model_path> -i <input_path> -m <model-type>
```

Arguments
---------

- `--hef-path, -n`: 
    - A **model name** (e.g., `yolov8m_seg`) → the script will automatically download and resolve the correct HEF for your device.
    - A **file path** to a local HEF → the script will use the specified network directly.
- `-i, --input`:
  - An **input source** such as an image (`bus.jpg`), a video (`video.mp4`), a directory of images, or `usb` to auto-select the first available USB camera.
    - On Linux, you can also use /dev/vidoeX (e.g., `/dev/video0`) to select a specific camera.
    - On Windows, you can also use a camera index (`0`, `1`, `2`, ...) to select a specific camera.
    - On Raspberry Pi, you can also use `rpi` to enable the Raspberry Pi camera.
  - A **predefined input name** from `resources_config.yaml` (e.g., `bus`, `street`).
    - If you choose a predefined name, the input will be **automatically downloaded** if it doesn't already exist.
    - Use `--list-inputs` to display all available predefined inputs.
- `-m, --model-type`: Specify the model family used by your HEF: v5 (YOLOv5), v8 (YOLOv8), fast (fast-seg).
- `-b, --batch-size`: [optional] Number of images in one batch. Defaults to 1.
- `-l, --labels`: [optional] Path to a text file containing class labels. If not provided, default COCO labels are used.
- `-s, --save_stream_output`: [optional] Save the output of the inference from a stream.
- `-o, --output-dir`: [optional] Directory where output images/videos will be saved.
- `cr, --camera-resolution`: [optional][Camera only] Input resolution: `sd` (640x480), `hd` (1280x720), or `fhd` (1920x1080).
- `or, --output-resolution`: [optional] Set output size using `sd|hd|fhd`, or pass custom width/height (e.g., `--output-resolution 1920 1080`).
- `--track`: [optional] Enable object tracking across frames using BYTETracker.
- `--show-fps`: [optional] Display FPS performance metrics for video/camera input.
- `--no-display`: [optional] Run without opening a display window. Useful for headless or performance testing.
- `--video-unpaced`: [optional] Process video input as fast as possible without respecting the original video FPS (no pacing).
- `-t`, `--time-to-run`: [optional] Maximum runtime in seconds. Stops the application after the specified duration.
- `-f, --frame-rate`: [optional][Camera only] Override the camera input framerate.
- `--list-models`: [optional] Print all supported models for this application (from `resources_config.yaml`) and exit.
- `--list-inputs`: [optional] Print the available predefined input resources (images/videos) defined in `resources_config.yaml` for this application, then exit.


For more information:
```shell script
./instance_segmentation.py -h
```

Example 
-------
**List supported networks**
```shell script
./instance_segmentation.py --list-nets
```

**List available input resources**
```shell script
./instance_segmentation.py --list-inputs
```

**Regular object detection on a usb camera stream**
```shell script
./instance_segmentation.py -n yolov5m_seg_with_nms.hef -i usb -m v5
```

**Object detection with tracking on a usb camera stream**
```shell script
./instance_segmentation.py -n yolov5m_seg_with_nms.hef -i usb -m v5 --track
```

**Inference on an image**
```shell script
./instance_segmentation.py -n yolov5m_seg_with_nms.hef -i dog_bicycle.jpg -m v5
```

**Inference on a folder of images**
```shell script
./instance_segmentation.py -n yolov5m_seg_with_nms.hef -i input_folder -m v5
```

**Output**
Results are saved to output/ by default
Override with:
```shell script
--output-dir <path>
```

🔧 Visualization and Tracking Configuration
-------------------------------------------
The application supports flexible configuration for how detections and tracking results are visualized. These settings can be modified in the configuration file to adjust the appearance of detection outputs and the behavior of the object tracker.

#### Example Configuration:
```json
"visualization_params": {
    "score_thres": 0.42,
    "mask_thresh": 0.2,
    "mask_alpha": 0.7,
    "max_boxes_to_draw": 30,
    "tracker": {
        "track_thresh": 0.01,
        "track_buffer": 30,
        "match_thresh": 0.9,
        "aspect_ratio_thresh": 2.0,
        "min_box_area": 500,
        "mot20": false
    }
}
```

#### Parameter Descriptions:

**Visualization Parameters:**

- `score_thres`: Minimum confidence score required to display a detected object.
- `mask_thresh`: Threshold for displaying segmentation masks (e.g., only show masks with values above this).
- `mask_alpha`: Transparency level of the segmentation mask overlay (0 = fully transparent, 1 = fully opaque).
- `max_boxes_to_draw`: Maximum number of detected objects to display per frame.

**Tracker Parameters:**

- `track_thresh`: Minimum score for a detection to be considered for tracking.
- `track_buffer`: Number of frames to retain lost tracks before deleting them.
- `match_thresh`: IoU threshold used to associate detections with existing tracks.
- `aspect_ratio_thresh`: Maximum allowed aspect ratio of detected objects (used to filter invalid boxes).
- `min_box_area`: Minimum area (in pixels) of a detection to be considered valid for tracking.
- `mot20`: Whether to use MOT20-style tracking behavior (set to `false` for standard tracking).

📊 Performance Notes
--------------------
This example supports two types of models:

- Models with built-in HailoRT postprocessing (including NMS e.g Yolov5m-seg):  
These models include optimized NMS and postprocessing inside the HEF, allowing full offload to the Hailo device.

      Camera input: ~30 FPS
      Video input: ~42 FPS


- Models that require host-side postprocessing (e.g yolov8s_seg):
   These models rely on the host CPU for NMS and mask refinement, which significantly affects real-time performance.

      Camera input: ~5 FPS
      Video input: ~3 FPS

Additional Notes
----------------

- The example was tested with:
    - HailoRT v4.23.0 (for Hailo-8)
    - HailoRT v5.3.0 (for Hailo-10H)
- Images are only supported in the following formats: .jpg, .jpeg, .png or .bmp
- Number of input images should be divisible by `batch_size`
- Using the yolov-seg model for inference, this example performs instance segmentation, draws detection boxes and adds a label to each class. When using the FastSAM model, it only performs the instance segmentation.
- As this example is designed to work with COCO-trained yolo-seg models, when using a custom trained yolo-seg model, please note that some values may need to be changed in the relevant functions AND that the classes under CLASS_NAMES_COCO in hailo_model_zoo/core/datasets/datasets_info.py file in the Hailo Model Zoo are to be changed according to the relevant classes of the custom model.
- The list of supported detection models is defined in `networks.json`.
- For any issues, open a post on the [Hailo Community](https://community.hailo.ai)

Disclaimer
----------
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.<br />
Please note that this example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.
