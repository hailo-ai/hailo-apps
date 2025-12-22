# Instance Segmentation Application

![Instance Segmentation Example](../../../../doc/images/instance_segmentation.gif)

#### Run the instance segmentation example:
```bash
hailo-seg
```
To close the application, press `Ctrl+C`.

Instance segmentation refers to the process of identifying and segmenting individual objects within a video frame. It combines object detection (locating objects and drawing bounding boxes) with semantic segmentation (assigning pixel-level labels to regions of the image). The goal is to not only detect objects but also to generate a mask for each detected object, distinguishing it from other objects and the background.

### How Instance Segmentation Works in the App:

#### **Object Detection**:
- The app uses a neural network to detect objects in the video frame.
- Each detected object is represented as a `HAILO_DETECTION` object, which includes metadata such as:
  - **Label**: The class of the object (e.g., "person").
  - **Bounding Box**: The coordinates of the object's location in the frame.
  - **Confidence**: The probability score of the detection.

#### **Instance Segmentation Masks**:
- For each detected object, the app retrieves a mask (`HAILO_CONF_CLASS_MASK` object).
- The mask is a pixel-level representation of the object's shape within the bounding box.
- Masks are resized and reshaped to match the frame's resolution and the object's bounding box dimensions.

#### **Overlaying Masks**:
- The masks are overlaid on the video frame using colors to visually distinguish different objects.
- The overlay is blended with the original frame to highlight the segmented objects.

#### **Tracking Objects**:
- The app assigns a unique track ID to each detected object, allowing it to track the object across frames.

#### Running with Raspberry Pi Camera input:
```bash
hailo-seg --input rpi
```

#### Running with USB camera input (webcam):
There are 2 ways:

Specify the argument `--input` to `usb`:
```bash
hailo-seg --input usb
```

This will automatically detect the available USB camera (if multiple are connected, it will use the first detected).

Second way:

Detect the available camera using this script:
```bash
get-usb-camera
```
Run example using USB camera input - Use the device found by the previous script:
```bash
hailo-seg --input /dev/video<X>
```

For additional options, execute:
```bash
hailo-seg --help
```

#### Running as Python script

For examples:
```bash
python instance_segmentation.py --input usb
```

#### Working in Python with the results

The basic idea is to utilize the pipeline's callback function. In simple terms, it can be thought of as a Python function that is invoked at the end of the pipeline when frame processing is complete.

This is the recommended location to implement your logic.

The callback function processes instance segmentation metadata from the network output. Each instance is represented as a HAILO_DETECTION with a mask (HAILO_CONF_CLASS_MASK object). The function parses, resizes, and reshapes the masks according to the frame coordinates, and overlays the masks on the frame if the `--use-frame` flag is set. The function also prints the detection details, including the track ID, label, and confidence, to the terminal.

Detailed Steps of the `callback` function:

1. **Retrieve the Buffer**:
   - The function starts by extracting the GstBuffer from the probe info (`info.get_buffer()`).
   - If the buffer is invalid (`None`), it returns immediately without further processing.

2. **Frame Counting**:
   - The `user_data.increment()` method is called to increment the frame count.
   - The current frame count is printed for debugging purposes.

3. **Frame Skipping**:
   - To reduce computational load, the function skips processing for frames based on the `frame_skip` value in `user_data`.
   - Only frames that satisfy the condition (`user_data.get_count() % user_data.frame_skip == 0`) are processed.

4. **Extract Video Frame Properties**:
   - The function retrieves the video format, width, and height from the pad using `get_caps_from_pad(pad)`.

5. **Reduce Resolution**:
   - The resolution of the video frame is reduced by a factor of 4 to optimize processing.

6. **Extract Video Frame**:
   - If `user_data.use_frame` is `True`, the function extracts the video frame as a NumPy array using `get_numpy_from_buffer(buffer, format, width, height)`.
   - The extracted frame is resized to the reduced resolution.

7. **Object Detection**:
   - The function retrieves the Region of Interest (ROI) from the buffer using `hailo.get_roi_from_buffer(buffer)`.
   - It extracts detections from the ROI using `roi.get_objects_typed(hailo.HAILO_DETECTION)`.

8. **Parse Detections**:
   - For each detection:
     - The label, bounding box, and confidence score are extracted.
     - If the label is `"person"`, the function retrieves the track ID for the detected object.
     - Detection information (ID, label, confidence) is appended to the debug string.

9. **Instance Segmentation Mask**:
   - If `user_data.use_frame` is `True` and masks are available:
     - The mask is reshaped to its original dimensions.
     - The mask is resized to match the bounding box dimensions.
     - The mask is overlaid on the reduced video frame using a color corresponding to the track ID.

10. **Overlay Mask on Frame**:
    - The mask overlay is blended with the reduced video frame using `cv2.addWeighted()`.

11. **Print Debug Information**:

    - he detection information (frame count, detection details) is printed to the console.

12. **Convert Frame to BGR**:

    - If user_data.use_frame is True, the reduced frame is converted from RGB to BGR format using OpenCV (cv2.cvtColor()).
    - The processed frame is stored in user_data using user_data.set_frame().

13. **Return Pad Probe Status**:

    - The function returns Gst.PadProbeReturn.OK to indicate successful processing.

### Key Features

#### Frame Skipping: Processes every 2nd frame to reduce computational load.
#### Color Coding: Uses predefined colors to differentiate between tracked instances.
#### Mask Overlay: Resizes and overlays the segmentation masks on the frame.
#### Boundary Handling: Ensures the ROI dimensions are within the frame boundaries and handles negative values.

## Command Line Arguments

### All pipeline commands support these common arguments:

[Common arguments](../../../../doc/user_guide/running_applications.md#command-line-argument-reference)