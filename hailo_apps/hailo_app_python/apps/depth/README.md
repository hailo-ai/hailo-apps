# Depth Application

![Depth Example](../../../../doc/images/depth.gif)

#### Run the depth example:
```bash
hailo-depth
```
To close the application, press `Ctrl+C`.

This example demonstrates depth estimation using the scdepthv3 model.

The result of depth estimation is essentially assigning each pixel in the image frame with an additional property - the distance from the camera.

For example:

Each pixel is represented by its position in the frame (x, y).

The value of the pixel might be represented by a trio of (Red, Green, Blue) values.

Depth estimation adds a fourth dimension to the pixel - the distance from the camera: (Red, Green, Blue, Distance).

However, it's important to familiarize yourself with the meaning of depth values, such as the fact that the distances might be relative, normalized, and unitless. Specifically, the results might not represent real-world distances from the camera to objects in the image. Please refer to the original scdepthv3 paper for more details see the C++ post-processing section below.

#### Running with Raspberry Pi Camera input:
```bash
hailo-depth --input rpi
```

#### Running with USB camera input (webcam):
There are 2 ways:

Specify the argument `--input` to `usb`:
```bash
hailo-depth --input usb
```

This will automatically detect the available USB camera (if multiple are connected, it will use the first detected).

Second way:

Detect the available camera using this script:
```bash
get-usb-camera
```
Run example using USB camera input - Use the device found by the previous script:
```bash
hailo-depth --input /dev/video<X>
```

For additional options, execute:
```bash
hailo-depth --help
```

#### Running as Python script

For examples: 
```bash
python depth.py --input usb
```

#### App logic

This function demonstrates parsing the HAILO_DEPTH_MASK depth matrix. Each GStreamer buffer contains a HAILO_ROI object, serving as the root for all Hailo metadata attached to the buffer. The function extracts the depth matrix for each frame buffer. The depth values are part of a separate matrix representing the frame with only depth values for each pixel (without the RGB values). For each depth matrix, using the User Application Callback Class, a logical calculation is performed and the result is printed to the terminal (CLI).

Note about frame sizing and rescaling: the scdepthv3 output frame size (depth matrix) is 320x256 pixels, which is typically smaller than the camera's frame size (resolution). The Hailo INFERENCE_PIPELINE_WRAPPER GStreamer pipeline element, which is part of the depth GStreamer pipeline, rescales the depth matrix to the original frame size.

The `user_application_callback_class` includes various methods for manipulating the depth results. In this example, we filter out the highest 5% of the values (treating them as outliers) and then calculate the average depth value across the frame.

#### Working in Python with the results

The basic idea is to utilize the pipeline's callback function. In simple terms, it can be thought of as a Python function that is invoked at the end of the pipeline when frame processing is complete.

This is the recommended location to implement your logic.

```bash
def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    roi = hailo.get_roi_from_buffer(buffer)
    depth_mat = roi.get_objects_typed(hailo.HAILO_DEPTH_MASK)
    some_calculation_result = foo_gets_np_array(np.array(depth_mat[0].get_data()).flatten())
    return Gst.PadProbeReturn.OK
```

#### C++ Post Processing

Basic post-processing during the pipeline occurs in the C++ code and can be found in the relevant `.so` file, e.g. `/hailo_cpp_postprocess/cpp/yolo_hailortpp.cpp`

The relevant function is the one whose name is specified in `self.post_function_name` or `self.options_menu.pp_function`.

This is the location in the C++ code for custom logic modifications.

For more details: [Writing postprocess](../../../../doc/developer_guide/writing_postprocess.md)

### All pipeline commands support these common arguments:

[Common arguments](../../../../doc/user_guide/running_applications.md#command-line-argument-reference)