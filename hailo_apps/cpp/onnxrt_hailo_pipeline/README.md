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
  - For Hailo-10: `HailoRT==5.1.1`

- ONNX Runtime >= 1.18.0
    ```shell script
    # Download and install ONNX Runtime from: https://github.com/microsoft/onnxruntime/releases
    # Or use package manager if available
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
    git clone <https://github.com/hailo-ai/Hailo-Application-Code-Examples.git>
        
    cd Hailo-Application-Code-Examples/runtime/hailo-8/cpp/onnxrt_hailo_pipeline
    ``` 

2. Compile the project on the development machine  
	```shell script
    ./build.sh
    ```
	This creates the directory hierarchy build/ and compile an executable file called onnxrt_hailo_pipeline

3. Run the example:

	```shell script
	./build/x86_64/onnxrt_hailo_pipeline --net <hef_path> --onnx <onnx_path> --input <image_or_video_or_camera_path>
    ```

Arguments
---------
- `-n, --net`: 
    - A **model name** (e.g., `yolov8m_seg`) → the script will automatically download and resolve the correct HEF for your device.
    - A **file path** to a local HEF → the script will use the specified network directly.
- `-i, --input`:
  - An **input source** such as an image (`bus.jpg`), a video (`video.mp4`), a directory of images, or `camera` to use the system camera.
  - A **predefined input name** from `inputs.json` (e.g., `bus`, `street`).
    - If you choose a predefined name, the input will be **automatically downloaded** if it doesn't already exist.
- `-o, --onnx`: Path to the ONNX model file.
- `-b, --batch-size`: [optional] Number of images in one batch. Defaults to 1.
- `-s, --save_stream_output`: [optional] Save the output of the inference from a stream.
- `-o, --output-dir`: [optional] Directory where output images/videos will be saved.
- `--camera-resolution`: [optional][Camera only] Input resolution: `sd` (640x480), `hd` (1280x720), or `fhd` (1920x1080).
- `--output-resolution`: [optional] Set output size using `sd|hd|fhd`, or pass custom width/height (e.g., `--output-resolution 1920 1080`).
- `-f, --framerate`: [optional][Camera only] Override the camera input framerate.
- `--list-nets` [optional] Print all supported networks for this application (from `networks.json`) and exit.
- `--list-inputs`: [optional] Print the available predefined input resources (images/videos) defined in `inputs.json` for this application, then exit.


Example
-------------------
- List supported networks:
    ```shell script
    ./build/x86_64/onnxrt_hailo_pipeline --list-nets
    ```
- List available input resources:
    ```shell script
    ./build/x86_64/onnxrt_hailo_pipeline --list-inputs
    ```
- For a video:
    ```shell script
	./build/x86_64/onnxrt_hailo_pipeline --net yolov8m_seg.hef --onnx yolov8m-seg_post.onnx --input full_mov_slow.mp4 --batch-size 16
    ```
    Output video is saved as processed_video.mp4

- For a single image:
    ```shell script
    ./build/x86_64/onnxrt_hailo_pipeline -n yolov8m_seg.hef -o yolov8m-seg_post.onnx -i image.jpg
    ```
    Output image is saved as processed_image_0.jpg

- For a directory of images:
    ```shell script
    ./build/x86_64/onnxrt_hailo_pipeline -n yolov8m_seg.hef -o yolov8m-seg_post.onnx -i images -b 4
    ````
    Each image is saved as processed_image_i.jpg
    
- For camera, enabling saving the output:
    ```shell script
    ./build/x86_64/onnxrt_hailo_pipeline --net yolov8m_seg.hef --onnx yolov8m-seg_post.onnx --input /dev/video0 --batch-size 2 -s
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

