# Detection Application

![Detection Example](../../../../doc/images/detection.gif)

#### Run the detection example:
```bash
hailo-detect
```
To close the application, press `Ctrl+C`.

This is the full detection example, including object tracker and multiple video resolution support.

#### Running with Raspberry Pi Camera input:
```bash
hailo-detect --input rpi
```

#### Running with USB camera input (webcam):
There are 2 ways:

Specify the argument `--input` to `usb`:
```bash
hailo-detect --input usb
```

This will automatically detect the available USB camera (if multiple are connected, it will use the first detected).

Second way:

Detect the available camera using this script:
```bash
get-usb-camera
```
Run example using USB camera input - Use the device found by the previous script:
```bash
hailo-detect --input /dev/video<X>
```

For additional options, execute:
```bash
hailo-detect --help
```

#### Running as Python script

For examples:
```bash
python detection.py --input usb
```

#### App logic

The application will detect and classify various objects based on the well known ["COCO list"](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml). The "bussines logic" happen in Python's `app_callback` function, and currently the example implementation there is to filter for only detected "Persons":
```bash
if label == "person":
    ...
```
The `user_app_callback_class` has a dummy implementation, but can be used to communicate and share variables with the pipeline class `GStreamerDetectionApp`.

#### Working in Python with the results

The basic idea is to utilize the pipeline's callback function. In simple terms, it can be thought of as a Python function that is invoked at the end of the pipeline when frame processing is complete.

This is the recommended location to implement your logic.

```bash
def app_callback(pad, info, user_data):
    frame_number = user_data.get_count()  # Using the user_data class count frames
    buffer = info.get_buffer()  # Get the GstBuffer from the probe info
    for detection in hailo.get_roi_from_buffer(buffer).get_objects_typed(hailo.HAILO_DETECTION):
        label = detection.get_label()
        confidence = detection.get_confidence()
        track_id = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)[0].get_id()
    return Gst.PadProbeReturn.OK
```

#### C++ Post Processing

Basic post-processing during the pipeline occurs in the C++ code and can be found in the relevant `.so` file, e.g. `/hailo_cpp_postprocess/cpp/yolo_hailortpp.cpp`

The relevant function is the one whose name is specified in `self.post_function_name` or `self.options_menu.pp_function`.

This is the location in the C++ code for custom logic modifications.

For more details: [Writing postprocess](../../../../doc/developer_guide/writing_postprocess.md)

#### Resolution

Based on the specific model and camera capabilities, the resolution can be set using the following `GStreamerDetectionApp` class members:

```bash
self.video_width
self.video_height
```

#### Running with different models

See our [hailo_model_zoo](https://github.com/hailo-ai/hailo_model_zoo) for additional supported models.

By default, the package contains a single model depending on the device architecture.
You can download additional models by running `hailo-download-resources --all`.
The models are downloaded to the `resources/models/` directory.
This application supports all models that are compiled with HailoRT NMS post process.

#### Retrained Networks Support
This application includes support for using retrained detection models. For more information, see [Using Retrained Models](../../../../doc/developer_guide/retraining_example.md).

## Command Line Arguments

### Application specific arguments:
```bash
--labels-json <path>    # Path to custom labels JSON file
--so-path <path>        # Path to the shared object file for post-processing
--pp-function <name>    # Name of the post-processing function in the shared object file
```

### All pipeline commands support these common arguments:

[Common arguments](../../../../doc/user_guide/running_applications.md#command-line-argument-reference)