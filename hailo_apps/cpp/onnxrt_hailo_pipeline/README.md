ONNXRT Hailo Pipeline
========================================
This example demonstrates how to integrate ONNX Runtime (ORT) as a postprocess stage inside a Hailo pipeline when dealing with unsupported ONNX operations. The instance segmentation task (YOLOv8-Seg) is only an example — the method applies to any model where you crop at the last supported nodes and run the remainder in ORT. These integrations are mainly shown in the onnxrt_hailo_pipeline.cpp file and onnx_decode hpp/cpp files.
This example aims to demonstrate these capabilites and provide visual results rather than fit one specific model.
When working with another model architecture or another task, expect to modify certain functions to match your model’s inputs and outputs.
That might include modifying the output parsing in onnx_decode.cpp (e.g. parse_onnx_output()), the visualization/postprocess in instance_seg_postprocess.cpp (e.g. masks, boxes, keypoints).

This example performs an instance segmentation task using a **Hailo8**, **Hailo8l** or **Hailo10** device combined with **ONNX Runtime** .
It receives a HEF (without postprocessing), an ONNX model, and images/video/camera as input, and returns the image\video with annotations of detected objects, bounding boxes, and instance masks.

📘 How to crop & choose I/O (step-by-step guide):
[community guide](https://community.hailo.ai/t/handling-unsupported-operations-with-onnxruntime/17956)


![Instance Segmentation Example](instance_seg.gif)

Requirements
------------

- HailoRT  
  - For Hailo-8: `HailoRT==4.23.0`  
  - For Hailo-10: `HailoRT==5.3.0`

- ONNX Runtime >= 1.18.0
    ```shell script
    # Download and install ONNX Runtime from: https://github.com/microsoft/onnxruntime/releases
    # Or use package manager if available
    ```

- **Linux Dependencies**
    - yaml-cpp and libcurl development libraries
        ```shell
        sudo apt-get install libyaml-cpp-dev libcurl4-openssl-dev
        ```
    - OpenCV >= 4.5.4
        ```shell script
        sudo apt-get install -y libopencv-dev python3-opencv
        ```
    - Boost
        ```shell script
        sudo apt-get install libboost-all-dev
        ```
    - CMake >= 3.16
    - Gtk
    - g++-9
        ```shell script
        sudo apt-get install gcc-9 g++-9
        ```

- **Windows Dependencies**

    - OpenCV >= 4.5.4
        ```shell script
        vcpkg install opencv
        ```
    - Boost
        ```shell script
        vcpkg install boost
        ```
    - CMake >= 3.16
    - Visual Studio with MSVC C++ build tools


Supported Models
----------------
This example requires HEFs **without** HailoRT-Postprocess, as postprocessing is handled by ONNX Runtime.

The pipeline works as follows:
1. **Hailo inference**: Raw tensor outputs from the HEF
2. **ONNX Runtime postprocessing**: Decode detections and generate masks
3. **Final processing**: NMS, mask composition, and visualization

Supported model combinations:
- **YOLOv8m-Seg**: HEF (without postprocess) + ONNX postprocessing model
- **YOLOv5m-Seg**: HEF (without postprocess) + ONNX postprocessing model  
- **Other instance segmentation models**: Any HEF + compatible ONNX postprocessing model

The ONNX model should be a "decode-only" model that takes raw Hailo outputs and produces decoded detections and mask coefficients.

Usage
-----
0. Make sure you have installed all of the requirements.

1. Clone the repository:
    ```shell script
    git clone --recurse-submodules https://github.com/hailo-ai/hailo-apps.git
    cd hailo-apps/hailo_apps/cpp/onnrt_hailo_pipeline
    ``` 

2. Compile the project on the development machine

    - **Linux**
        ```shell script
        ./build.sh
        ```
    - **Windows**
        Replace `<path-to-onnxruntime>` with the path to the extracted ONNX Runtime package.
        ```powershell
        cmake -S . -B build -DCMAKE_FIND_PACKAGE_RESOLVE_SYMLINKS=True -DONNXRUNTIME_DIR=<path-to-onnxruntime>
        cmake --build build --config Release
        copy "<path-to-onnxruntime>\lib\*.dll" .\build\Release\
        ```

    This builds `onnxrt_hailo_pipeline` at `build/onnxrt_hailo_pipeline` on Linux and `build/Release/onnxrt_hailo_pipeline.exe` on Windows.

3. Run the example:

    - **Linux**
        ```shell script
        ./build/onnxrt_hailo_pipeline --net <hef_path> --onnx <onnx_path> --input <image_or_video_or_camera_path>
        ```
    - **Windows**
        ```shell script
        .\build\Release\onnxrt_hailo_pipeline.exe --net <hef_path> --onnx <onnx_path> --input <image_or_video_or_camera_path>
        ```

Arguments
---------
- `-n, --net`: 
    - A **model name** (e.g., `yolov8n`) → the script will automatically download and resolve the correct HEF for your device.
    - A **file path** to a local HEF → the script will use the specified network directly.
- `-i, --input`:
  - An **input source** such as an image (`bus.jpg`), a video (`video.mp4`), a directory of images, or `usb` to auto-select the first available USB camera.
    - On Linux, you can also use /dev/vidoeX (e.g., `/dev/video0`) to select a specific camera.
    - On Windows, you can also use a camera index (`0`, `1`, `2`, ...) to select a specific camera.
    - On Raspberry Pi, you can also use `rpi` to enable the Raspberry Pi camera.
    - On Astrial/IMX8 (Yocto-based systems), use `csi` to auto-detect and open the CSI camera. The ISP must be initialized first: `cd /opt/imx8-isp/bin && ./run.sh -lm -c dual_imx219_1080p60 &`
  - A **predefined input name** from `resources_config.yaml` (e.g., `bus`, `street`).
    - If you choose a predefined name, the input will be **automatically downloaded** if it doesn't already exist.
    - Use `--list-inputs` to display all available predefined inputs.
- `-o, --onnx`: Path to the ONNX model file.
- `-b, --batch-size`: [optional] Number of images in one batch. Defaults to 1.
- `-s, --save_stream_output`: [optional] Save the output of the inference from a stream.
- `--no-display`: [optional] Run without opening a display window.
- `-o, --output-dir`: [optional] Directory where output images/videos will be saved.
- `--camera-resolution`: [optional][Camera only] Input resolution: `sd` (640x480), `hd` (1280x720), or `fhd` (1920x1080).
- `--output-resolution`: [optional] Set output size using `sd|hd|fhd`, or pass custom width/height (e.g., `--output-resolution 1920 1080`).
- `-f, --framerate`: [optional][Camera only] Override the camera input framerate.
- `--list-models`: [optional] Print all supported networks for this application from the shared `resources_config.yaml`, then exit.
- `--list-inputs`: [optional] Print the available predefined input resources (images/videos) from the shared `resources_config.yaml`, then exit.


Example
-------------------
- List supported networks:
    ```shell script
    ./build/onnxrt_hailo_pipeline --list-nets
    ```
- List available input resources:
    ```shell script
    ./build/onnxrt_hailo_pipeline --list-inputs
    ```
- For a video:
    ```shell script
	./build/onnxrt_hailo_pipeline --net yolov8m_seg.hef --onnx yolov8m-seg_post.onnx --input full_mov_slow.mp4 --batch-size 16
    ```
    Output video is saved as processed_video.mp4

- For a single image:
    ```shell script
    ./build/onnxrt_hailo_pipeline -n yolov8m_seg.hef -o yolov8m-seg_post.onnx -i image.jpg
    ```
    Output image is saved as processed_image_0.jpg

- For a directory of images:
    ```shell script
    ./build/onnxrt_hailo_pipeline -n yolov8m_seg.hef -o yolov8m-seg_post.onnx -i images -b 4
    ````
    Each image is saved as processed_image_i.jpg
    
- For camera, enabling saving the output:
    ```shell script
    ./build/onnxrt_hailo_pipeline --net yolov8m_seg.hef --onnx yolov8m-seg_post.onnx --input /dev/video0 --batch-size 2 -s
    ```
    Output video is saved as processed_video.mp4

Notes
----------------
- Last ONNX Runtime version checked - ``ONNX Runtime v1.18.0``
- The script assumes that the image is in one of the following formats: .jpg, .jpeg, .png or .bmp 
- There should be no spaces between "=" given in the command line arguments and the file name itself
- **Important**: This example requires HEFs **without** HailoRT-Postprocess, as postprocessing is handled by ONNX Runtime
- The ONNX model must be compatible with the HEF outputs (same tensor shapes and data types)
- When using camera as input:
    - To exit gracefully from openCV window, press 'q'.
    - Camera path is usually found under /dev/video0.
    - Ensure you have the permissions for the camera. You may need to run, for example:
        ```shell script
        sudo chmod 777 /dev/video0
        ```
    - In case OpenCV is defaulting to GStreamer for video capture, warnings might occur.
      To solve, force OpenCV to use V4L2 instead of GStreamer by setting these environment variables:
      ```
        export OPENCV_VIDEOIO_PRIORITY_GSTREAMER=0
        export OPENCV_VIDEOIO_PRIORITY_V4L2=100
      ```
- Using multiple models on same device:
    - If you need to run multiple models on the same virtual device (vdevice), use the AsyncModelInfer constructor that accepts two arguments. Initialize each model using the same group_id. 
    - Example:
      ```
         std::string group_id = "<group_id>";
         AsyncModelInfer model1("<hef1_path>", group_id);
         AsyncModelInfer model2("<hef2_path>", group_id);
      ```
    - By assigning the same group_id to models from different HEF files, you enable the runtime to treat them as part of the same group, allowing them to share resources and run more efficiently on the same hardware.

Disclaimer
----------
This code example is provided by Hailo solely on an "AS IS" basis and "with all faults". No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.

This example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.

