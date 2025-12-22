Semantic Segmentation
====================
This example performs semantic segmentation (e.g., ResNet18-FCN) on Cityscapes data.
It runs on Hailo-8, Hailo-8l and Hailo-10h devices, accepts input from an image, video, or camera, and produces per-pixel class masks overlaid on the original frames.

![output example](./semantic_seg.gif)

Requirements
------------

- HailoRT  
  - For Hailo-8: `HailoRT==4.23.0`  
  - For Hailo-10: `HailoRT==5.1.1`

- Make sure OpenCV and g++-9 are installed:
   ```shell
   sudo apt-get install -y libopencv-dev gcc-9 g++-9
   ```

Supported Models
----------------
- fcn16_resnet_v1_18

Usage
-----
0. Make sure you have installed all of the requirements.

1. Clone the repository:
    ```shell script
    git clone <https://github.com/hailo-ai/Hailo-Application-Code-Examples.git>
        
    cd Hailo-Application-Code-Examples/runtime/hailo-8/cpp/semantic_segmentation
    ``` 

2. Compile the project on the development machine  
    ```shell script
    ./build.sh
    ```
    This creates the directory hierarchy build/ and compile an executable file called semantic_segmentation

3. Run the example:

    ```shell script
    ./build/semantic_segmentation --net <hef_path> --input <image_or_video_or_camera_path>
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
    ./build/x86_64/semantic_segmentation --list-nets
    ```
- List available input resources:
    ```shell script
    ./build/x86_64/semantic_segmentation --list-inputs
    ```
- For a video:
    ```shell script
    ./build/x86_64/semantic_segmentation -n fcn8_resnet_v1_18.hef -i full_mov_slow.mp4 --batch-size 16
    ```
    Output video is saved as processed_video.mp4

- For a single image:
    ```shell script
    ./build/x86_64/semantic_segmentation -n fcn8_resnet_v1_18.hef -i zidane.jpg
    ```
    Output image is saved as processed_image_0.jpg

- For a directory of images:
    ```shell script
    ./build/x86_64/semantic_segmentation -n fcn8_resnet_v1_18.hef -i images -b 4
    ````
    Each image is saved as processed_image_i.jpg
    
- For camera, enabling saving the output:
    ```shell script
    ./build/x86_64/semantic_segmentation -n fcn8_resnet_v1_18.hef --input /dev/video0 --batch-size 2 -s
    ```
    Output video is saved as processed_video.mp4

Notes
----------------
- The script assumes that the image is in one of the following formats: .jpg, .jpeg, .png or .bmp 
- There should be no spaces between "=" given in the command line arguments and the file name itself
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
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.

This example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.
