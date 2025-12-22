# Pose Estimation Application

![Pose Estimation](../../../../doc/images/pose_estimation.gif)

#### Run the pose estimation example:
```bash
hailo-pose
```
To close the application, press `Ctrl+C`.

This example demonstrates human pose estimation using the yolov8s_pose model for Hailo-8l (13 TOPS) and the yolov8m_pose model for Hailo-8 (26 TOPS).

#### Running with Raspberry Pi Camera input:
```bash
hailo-pose --input rpi
```

#### Running with USB camera input (webcam):
There are 2 ways:

Specify the argument `--input` to `usb`:
```bash
hailo-pose --input usb
```

This will automatically detect the available USB camera (if multiple are connected, it will use the first detected).

Second way:

Detect the available camera using this script:
```bash
get-usb-camera
```
Run example using USB camera input - Use the device found by the previous script:
```bash
hailo-pose --input /dev/video<X>
```

For additional options, execute:
```bash
hailo-pose --help
```

#### Running as Python script

For examples: 
```bash
python pose_estimation.py --input usb
```

#### App logic

Each person is represented as a `HAILO_DETECTION` with 17 keypoints (`HAILO_LANDMARKS` objects).

The `get_keypoints` function provides a dictionary mapping keypoint names to their corresponding indices. This dictionary includes keypoints for the nose, eyes, ears, shoulders, elbows, wrists, hips, knees, and ankles.

#### Working in Python with the results

The basic idea is to utilize the pipeline's callback function. In simple terms, it can be thought of as a Python function that is invoked at the end of the pipeline when frame processing is complete.

This is the recommended location to implement your logic.

The callback function retrieves pose estimation metadata from the network output. The function parses the landmarks to extract the left and right eye coordinates, printing them to the terminal. If the --use-frame flag is set, the eyes are drawn on the user frame. Obtain the keypoints dictionary using the get_keypoints function.

#### **Detailed Steps**

1. **Retrieve the Buffer**:
   - Extracts the `GstBuffer` from the probe info (`info.get_buffer()`).
   - If the buffer is invalid (`None`), the function exits early.

2. **Frame Counting**:
   - Increments the frame count using `user_data.increment()`.
   - Prints the current frame count for debugging purposes.

3. **Extract Video Frame Properties**:
   - Retrieves the video format, width, and height from the pad using `get_caps_from_pad(pad)`.

4. **Extract Video Frame**:
   - If `user_data.use_frame` is `True`, the function extracts the video frame as a NumPy array using `get_numpy_from_buffer(buffer, format, width, height)`.

5. **Object Detection**:
   - Retrieves the Region of Interest (ROI) from the buffer using `hailo.get_roi_from_buffer(buffer)`.
   - Extracts detections from the ROI using `roi.get_objects_typed(hailo.HAILO_DETECTION)`.

6. **Parse Detections**:
   - For each detection:
     - Extracts the label, bounding box, and confidence score.
     - If the label is `"person"`, retrieves the track ID for the detected object.
     - Appends detection information (ID, label, confidence) to the debug string.

7. **Pose Estimation Landmarks**:
   - If landmarks are available for the detection:
     - Retrieves the pose estimation landmarks (`hailo.HAILO_LANDMARKS`).
     - Extracts specific keypoints (e.g., `left_eye`, `right_eye`) using `get_keypoints()`.
     - Calculates the coordinates of the keypoints based on the bounding box and frame dimensions.
     - Appends keypoint information (e.g., eye positions) to the debug string.
     - If `user_data.use_frame` is `True`, overlays visual markers (e.g., circles) on the video frame using OpenCV (`cv2.circle()`).

### All pipeline commands support these common arguments:

[Common arguments](../../../../doc/user_guide/running_applications.md#command-line-argument-reference)