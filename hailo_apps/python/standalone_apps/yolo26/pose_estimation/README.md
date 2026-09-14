Pose Estimation - *yolo26* with lightweight ONNX postprocessing
================================================================

This example demonstrates pose estimation using a Hailo-10H device, based on the 2026 Ultralytics YOLO26 release of top-performing NMS-free networks and a split HEF + ONNX postprocessing pipeline.

The example takes an input, performs inference using the input HEF file, and draws detection boxes, class type, confidence, keypoints, and joint connections on the resized image.

Supported input formats include:
- Images: .jpg, .jpeg, .png, .bmp
- Video: .mp4
- Live camera feed

This application includes additional demo variations that highlight the practical potential of high-quality, high-speed pose estimation:
1. **Skeleton tracklet ("following shadow")** — showcases dense temporal pose recognition enabled by high FPS.
2. **Ultralytics AI Gym integration** — counts fitness exercise repetitions and demonstrates broader action-recognition capabilities.
   In practice, this runs smoothly only with Hailo acceleration, as the 2–3 FPS typically achievable on Raspberry Pi CPU with the smallest network is not sufficient for capturing reasonably fast movement.


    <p align="center">
        <img src="output.gif" width="320" alt="Reflection-loop trail demo" />
        <img src="output_aigym.gif" width="220" alt="AIGym trail demo" />
    </p>



Requirements
------------

- hailo_platform:
    - 4.23.0 (for Hailo-8 devices)
    - 5.3.0 (for Hailo-10H devices)
- onnxruntime
- opencv-python
- pillow
- python-dotenv
- PyYAML

Supported Models
----------------
- yolov26n_pose
- yolov26s_pose
- yolov26m_pose


ONNX postprocessing
-------------------
Similarly to hailo_apps/cpp/onnxrt_hailo_pipeline, this example uses onnxruntime for the postprocessing part.  This makes integration of new networks especially convenient, by following these steps:
1. Split the ONNX into the "neural processing" and the "postprocessing" parts using extract_postprocessing.py script
2. Process the first part into a HEF using the DFC
3. In runtime, apply the second part on the HEF outputs with onnx-runtime engine to complete an accelerated equivalent of the original ONNX. This runtime part is implemented and exemplified in this app.
4. The desired "debug reference ONNX = HEF + postproc-onnx" equivalence can be tested by passing --neural-onnx-ref <path>, which bypasses HEF inference and feeds postprocessing from a user-provided reference ONNX model. This is useful for quick dry tests and for isolating pipeline-vs-compilation differences.


## Linux Installation

Run this app in one of two ways:
1. Standalone installation in a clean virtual environment (no GStreamer required) — see [Option 1](#option-1-standalone-installation)
2. From an installed `hailo-apps` repository — see [Option 2](#option-2-inside-an-installed-hailo-apps-repository)

### Option 1: Standalone Installation

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
    cd hailo-apps/python/standalone_apps/pose_estimation_onnx_postproc
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
Then the app is already ready for usage:
```shell script
cd hailo-apps/python/standalone_apps/pose_estimation_onnx_postproc
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
    cd hailo-apps\hailo_apps\python\standalone_apps\pose_estimation_onnx_postproc
    ```

2. Install dependencies:
    ```powershell
    pip install -r requirements.txt


## Run
After completing either installation option, run from the application folder:
```shell script
python .\pose_estimation.py -n <model_path> -i <input_path>
```

Arguments
---------

- `--hef-path, -n`: 
    - A **model name** (e.g., `yolov8m_pose`) → the script will automatically download and resolve the correct HEF for your device.
    - A **file path** to a local HEF → the script will use the specified network directly.
- `-i, --input`:
  - An **input source** such as an image (`bus.jpg`), a video (`video.mp4`), a directory of images, or `usb` to auto-select the first available USB camera.
    - On Linux, you can also use /dev/vidoeX (e.g., `/dev/video0`) to select a specific camera.
    - On Windows, you can also use a camera index (`0`, `1`, `2`, ...) to select a specific camera.
    - On Raspberry Pi, you can also use `rpi` to enable the Raspberry Pi camera.
  - A **predefined input name** from `resources_config.yaml` (e.g., `bus`, `street`).
    - If you choose a predefined name, the input will be **automatically downloaded** if it doesn't already exist.
    - Use `--list-inputs` to display all available predefined inputs.
- `-b, --batch-size`: Number of images in one batch.
- `-cn, --class_num`: The number of classes the model is trained on. Defaults to 1.
- `-s, --save-output`: [optional] Save the output of the inference from a stream.
- `-o, --output-dir`: [optional] Directory where output images/videos will be saved.
- `--show-fps`: [optional] Display FPS performance metrics for video/camera input.
- `--no-display`: [optional] Run without opening a display window. Useful for headless or performance testing.
- `--video-unpaced`: [optional] Process video input as fast as possible without respecting the original video FPS (no pacing).
- `-t`, `--time-to-run`: [optional] Maximum runtime in seconds. Stops the application after the specified duration.
- `cr, --camera-resolution`: [optional][Camera only] Input resolution: `sd` (640x480), `hd` (1280x720), or `fhd` (1920x1080).
- `or, --output-resolution`: [optional] Set output size using `sd|hd|fhd`, or pass custom width/height (e.g., `--output-resolution 1920 1080`).
- `-f, --frame-rate`: [optional][Camera only] Override the camera input framerate.
- `--list-models`: [optional] Print all supported models for this application (from `resources_config.yaml`) and exit.
- `--list-inputs`: [optional] Print the available predefined input resources (images/videos) defined in `resources_config.yaml` for this application, then exit.
- `--onnx ONNX_PP_FILE`: [optional] Override path to ONNX postprocessing model file (2nd part of split). If omitted, use existing resource lazy-downloaded from preconfigured cloud path (alongside the HEF)
- `--onnx-config ONNX_CONFIG_FILE`: [optional] Path to the ONNX postprocessing configuration file. If omitted, a default configuration is used if available.
- `--aigym EXERCISE`: [optional] Enable exercise rep-counting mode. Adds ByteTrack multi-person tracking and angle-based hysteresis counting. Choices of EXERCISE: squats, pushups, pullups.
- `--pose-trail N`: [optional]Number of previous frames whose pose skeletons are kept and drawn as a fading trail behind the current detection. 0 (default) disables the trail. Typical value: 10.
- `--mute-background ALPHA`: [optional] Dim the background image to emphasize pose skeletons.
- `--neural-onnx-ref ONNX_HEF_EQ_FILE`: [optional] For debug or quality/speed benchmarking - use a 'neural ONNX' file (1st part of splitting - corresponding to the HEF) to bypass hardware and run reference hef-equivalent model on the host CPU via the onnx-runtime engine.



For more information:
```shell script
./pose_estimation_onnx_postproc.py -h
```

Example 
-------
**List supported networks**
```shell script
./pose_estimation_onnx_postproc.py --list-nets
```

**List available input resources**
```shell script
./pose_estimation_onnx_postproc.py --list-inputs
```

**Inference on single image**
```shell script
./pose_estimation_onnx_postproc.py -i dog_bicycle.jpg -b 1
```

**Inference on a usb camera stream**
```shell script
./pose_estimation_onnx_postproc.py -i usb
```

**Inference on a usb camera stream with custom frame rate**
```shell script
./pose_estimation_onnx_postproc.py -i usb -f 20
```

Draw a trail of 10 past frames (~0.3sec) and deemphasize background:
```
python pose_estimation_onnx_postproc.py --i example.mp4 --hef yolo26m_pose --mute-background 0.5 --pose-trail 10
```
![Reflection-loop trail demo](output.gif)

Count squats for a whole class at once ("aigym"):
```
python pose_estimation_onnx_postproc.py --i grok-squats.mp4 --hef yolo26m_pose --aigym squats
```
try pushups, pullups as well :)

![aigym trail demo](output_aigym.gif)


Additional Notes
----------------

- The example was tested with:
    - HailoRT v4.23.0 (for Hailo-8)
    - HailoRT v5.3.0 (for Hailo-10H)
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