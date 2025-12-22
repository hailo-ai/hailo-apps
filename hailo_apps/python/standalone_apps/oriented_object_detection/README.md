Oriented Object Detection
=========================================

> ⚠️ **Beta:** This application is currently in beta. Features and APIs may change.

This example demonstrates a YoloV11 OBB based oriented object detection model using a Hailo-8, Hailo-8L, or Hailo-10H device.
It receives a HEF and images/video/camera as input, and returns the image\video with annotations of detected objects and bounding boxes.
Oriented object detection extends traditional bounding box detection by adding rotation angle, making it ideal for:
- Aerial/satellite imagery
- Document analysis
- Rotated text detection
- Any scenario where objects may appear at arbitrary angles

![output example](./example_output.png)


Requirements
------------

- hailo_platform==4.23.0
- loguru
- opencv-python
- scipy
- lap
- cython_bbox
- jq
  ```shell script
  sudo apt-get install jq
  ```

Supported Models
----------------
This example currently supports only YoloV11-OBB model.

Direct usage within the repository:
-----
```bash
cd ~/hailo_apps/python/standalone_apps/oriented_object_detection
python oriented_object_detection.py
```

To see all possible arguments:
```bash
cd ~/hailo_apps/python/standalone_apps/oriented_object_detection
python oriented_object_detection.py --help
```
 
Standalone Usage
-----

Usage
-----
To avoid compatibility issues, it's recommended to have a separate venv from the DFC.

0. Install PCIe driver and PyHailoRT
    - Download and install the PCIe driver and PyHailoRT from the Hailo website
    - To install the PyHailoRT whl:
        ```shell script
        pip install hailort-X.X.X-cpXX-cpXX-linux_x86_64.whl
        ```

1. Clone the repository:
    ```shell script
    git clone <https://github.com/hailo-ai/Hailo-Application-Code-Examples.git>
        
    cd Hailo-Application-Code-Examples/runtime/hailo-8/python/oriented_object_detection
    ```

2. Install dependencies:
    ```shell script
    pip install -r requirements.txt
    ```

3. Run the script:
    ```shell script
    ./oriented_object_detection -n <model_path> -i <input_image_path> -l <label_file_path> -b <batch_size>
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
- `-b, --batch_size`: [optional] Number of images in one batch. Defaults to 1.
- `-l, --labels`: [optional] Path to a text file containing class labels. If not provided, default COCO labels are used.
- `-s, --save_stream_output`: [optional] Save the output of the inference from a stream.
- `-o, --output-dir`: [optional] Directory where output images/videos will be saved.
- `-r, --resolution`: [Camera input only] Choose output resolution: `sd` (640x480), `hd` (1280x720), or `fhd` (1920x1080). If not specified, native camera resolution is used.
- `--show-fps`: [optional] Display FPS performance metrics for video/camera input.

For more information:
```shell script
./oriented_object_detection.py -h
```
Example 
-------
**Inference on a camera stream**
```shell script
./oriented_object_detection.py -n ./yolo11s_obb_.hef -i camera
```

**Inference with tracking on a camera stream**
```shell script
./oriented_object_detection.py -n ./yolo11s_obb.hef -i camera
```

**Inference with tracking and motion trail visualization**
```shell script
DRAW_TRAIL=1 ./oriented_object_detection.py -n ./yolo11s_obb.hef -i camera
```

**Inference on a camera stream with custom frame rate**
```shell script
FPS=30 ./oriented_object_detection.py -n ./yolo11s_obb.hef -i camera
```

**Inference on a video**
```shell script
./oriented_object_detection.py -n ./yolo11s_obb.hef -i full_mov_slow.mp4
```
**Inference on an image**
```shell script
./oriented_object_detection.py -n ./yolo11s_obb.hef -i bus.jpg
```
**Inference on a folder of images**
```shell script
./oriented_object_detection.py -n ./yolo11s_obb.hef -i input_folder
```

Visualization Configuration
-------------------------------------------
The application supports flexible configuration for how detections results are visualized. These settings can be modified in the configuration file to adjust the appearance of detection outputs.

#### Example Configuration:
```json
{
  "visualization_params": {
    "score_th": 0.35,
    "max_boxes_to_draw": 500
  },
  "oriented_postprocess": {
    "obb_model_input_map": {
      "yolo11s_obb_640x640_simp/conv53": "/model.23/cv2.0/cv2.0.2/Conv_output_0", 
      "yolo11s_obb_640x640_simp/conv54": "/model.23/cv4.0/cv4.0.2/Conv_output_0", 
      "yolo11s_obb_640x640_simp/conv57": "/model.23/cv3.0/cv3.0.2/Conv_output_0", 
      "yolo11s_obb_640x640_simp/conv67": "/model.23/cv2.1/cv2.1.2/Conv_output_0", 
      "yolo11s_obb_640x640_simp/conv68": "/model.23/cv4.1/cv4.1.2/Conv_output_0", 
      "yolo11s_obb_640x640_simp/conv71": "/model.23/cv3.1/cv3.1.2/Conv_output_0", 
      "yolo11s_obb_640x640_simp/conv85": "/model.23/cv2.2/cv2.2.2/Conv_output_0", 
      "yolo11s_obb_640x640_simp/conv86": "/model.23/cv4.2/cv4.2.2/Conv_output_0", 
      "yolo11s_obb_640x640_simp/conv89": "/model.23/cv3.2/cv3.2.2/Conv_output_0" 
    },
    "img_size": 640,
    "cls_num": 15,
    "scores_th": 0.375,
    "nms_iou_th": 0.25
  }
}
```

#### Parameter Descriptions:

**Visualization Parameters:**

- `score_thres`: Minimum confidence score required to display a detected object.
- `max_boxes_to_draw`: Maximum number of detected objects to display per frame.

**Oriented Postprocess Parameters:**

- `obb_model_input_map`: Mapping between model original names for identification of classes/boxes/angels heads.
- `img_size`: Input image size (width and height) expected by the model.
- `cls_num`: Number of object classes the model can detect.
- `scores_th`: Confidence threshold for filtering detections during post-processing.
- `nms_iou_th`: Intersection over Union (IoU) threshold for Non-Maximum Suppression to eliminate duplicate detections.


Additional Notes
----------------

- The example was only tested with ``HailoRT v4.23.0``
- The example expects a HEF which maps to the same outputs in the configuration json.
- Images are only supported in the following formats: .jpg, .jpeg, .png or .bmp
- Number of input images should be divisible by batch_size
- For any issues, open a post on the [Hailo Community](https://community.hailo.ai)


Disclaimer
----------
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.

This example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.
