# Hailo Application Development Guide

This guide provides a comprehensive overview of how to develop AI applications using the Hailo GStreamer framework. It covers the underlying architecture and two primary development paths: basic callback customization and advanced pipeline modification.

## The Technology Stack: A Layered Architecture

Our framework is built on a stack of technologies, from the open-source GStreamer foundation to our custom Python tools. The diagram below illustrates how these layers build upon one another, with each layer providing a higher level of abstraction.

![Hailo Architecture](../images/architecture.svg)

This diagram visually represents our technology stack. Understanding this stack is key to effective development.

### Level 1: The GStreamer Framework
At the base of our stack is **GStreamer**, a powerful open-source framework for creating streaming media applications. It provides a plugin-based architecture where elements are linked together to form a pipeline that defines a data flow. For more information, you can visit the [GStreamer official documentation](https://gstreamer.freedesktop.org/documentation/).

### Level 2: Hailo's Tappas C/C++ GStreamer Plugins
On top of GStreamer, we use **Hailo's Tappas** plugins. This is a library of C/C++ GStreamer elements specifically designed to interface with the Hailo AI accelerator. These high-performance elements are the bridge between the GStreamer framework and the Hailo hardware.

#### Core Tappas GStreamer Elements
The framework provides many specialized GStreamer elements.

<details>
<summary><strong>Click to see the list of Core Tappas Elements</strong></summary>

*   **AI/ML Processing Elements**
    *   **[HailoNet](https://github.com/hailo-ai/tappas/blob/master/docs/elements/hailo_net.rst)**: Runs neural network inference on input video frames using a Hailo device.(This element is released as part of HailoRT)
    *   **[HailoFilter](https://github.com/hailo-ai/tappas/blob/master/docs/elements/hailo_filter.rst)**: Applies
    C++ post-processing functions to the raw output from `HailoNet`, converting it into structured data like
    detection objects.
    *   **[HailoOverlay](https://github.com/hailo-ai/tappas/blob/master/docs/elements/hailo_overlay.rst)**: Draws
    the structured Hailo AI metadata (bounding boxes, masks, labels) onto the video frames for visualization.

*   **Multi-Device & Routing Elements**
    *   **[HailoMuxer](https://github.com/hailo-ai/tappas/blob/master/docs/elements/hailo_muxer.rst)**: Combines multiple metadata streams into a single output stream.
    *   **[HailoRoundRobin](https://github.com/hailo-ai/tappas/blob/master/docs/elements/hailo_roundrobin.rst)**: An element that provides muxing functionality. It receives input from one or more sink pads and forwards them into a single src pad in round-robin method.
    *   **[HailoStreamRouter](https://github.com/hailo-ai/tappas/blob/master/docs/elements/hailo_stream_router.rst)**: Routes video or metadata streams dynamically based on user-defined rules.

*   **Cascading Networks Elements**
    *   **[HailoCropper](https://github.com/hailo-ai/tappas/blob/master/docs/elements/hailo_cropper.rst)**: Crops regions of interest from video frames for further processing based on the output of the previous network.
    *   **[HailoAggregator](https://github.com/hailo-ai/tappas/blob/master/docs/elements/hailo_aggregator.rst)**: Merges multiple input streams or metadata into a single output stream.

*   **Tiling Elements**
    *   **[HailoTileCropper](https://github.com/hailo-ai/tappas/blob/master/docs/elements/hailo_tile_cropper.rst)**: Crops tiled regions from input frames, to allow processing each tiles independently.
    *   **[HailoTileAggregator](https://github.com/hailo-ai/tappas/blob/master/docs/elements/hailo_tile_aggregator.rst)**: Combines multiple cropped tiles into a single output frame.

*   **Tracking & Monitoring Elements**
    *   **[HailoTracker](https://github.com/hailo-ai/tappas/blob/master/docs/elements/hailo_tracker.rst)**: Tracks detected objects across video frames, assigning unique IDs to each object.
    *   **[HailoDeviceStats](https://github.com/hailo-ai/tappas/blob/master/docs/elements/hailo_device_stats.rst)**: Reports real-time statistics and health information from the Hailo device.

</details>

### Debugging Tools
For optimizing and debugging your GStreamer pipelines, we recommend using **GstShark**. It allows you to visualize queue levels, latency, and framerate to identify bottlenecks.
See our [GstShark Debugging Guide](debugging_with_gst_shark.md) for installation and usage instructions.

### Level 3: Hailo Apps Python Layer
This layer is developed in this repository to simplify the process of building and running applications on top of the GStreamer and TAPPAS foundation. It consists of three main components:
*   **The Application Runner (`gstreamer_app.py`)**: This component features the `GStreamerApp` class, which serves as the core engine of the application. It is responsible for managing the pipeline's lifecycle, handling bus messages (such as errors or End-Of-Stream), and integrating your Python callback functions.
*   **The Pipeline Factory (`gstreamer_helper_pipelines.py`)**: This module provides a set of Python functions that facilitate the creation of GStreamer pipeline strings in a modular and easily understandable manner. For a comprehensive reference of all available helper functions, see the [GStreamer Helper Pipelines Reference](./gstreamer_helper_pipelines.md).
*   **Hailo Pipelines**: These are pre-configured, ready-to-use AI pipelines that leverage the helper functions from the factory to form complete, executable applications for common scenarios like object detection or pose estimation. You can connect to their outputs with a simple callback, allowing you to easily integrate custom logic or processing steps.
For example, `hailo_apps/python/pipeline_apps/detection/detection_pipeline.py`.

## Development Path 1: Basic (Callback-based)

This is the quickest way to build an application. The core concept is to begin with one of our pre-built pipeline examples and incorporate your custom logic by writing a simple Python callback function. Each pipeline is designed to be executed with a straightforward callback.

We suggest using one of our example callback applications, such as `hailo_apps/python/pipeline_apps/detection/detection.py`, as your starting point. Each pipeline in this repository includes an example callback file (e.g., `detection.py`, `pose_estimation.py`, etc.). These files demonstrate the relevant callback code for that specific pipeline and can serve as a reference or starting point for your own application.

### The Callback Mechanism Explained
The GStreamer pipeline handles all complex tasks, including video decoding, inference, and rendering. Your Python **callback** function is invoked for each frame processed by the pipeline, receiving both the video frame and the AI metadata.

1.  **Data Production**: The pipeline processes a video frame, and the C++ post-processing library generates structured metadata (containing detections, etc.).
2.  **Callback Invocation**: An `identity` element in the pipeline intercepts the GStreamer buffer and triggers your Python function via the `handoff` signal, passing the element, buffer, and user data.
3.  **Your Custom Logic**: Within your callback, you parse the metadata and perform any necessary actions.

> **IMPORTANT**: The callback function must be non-blocking. Long-running tasks should be dispatched to a separate thread or process to prevent stalling the video pipeline.

### The User Application Workflow

A user application script typically has three parts: an optional custom data class, a callback function, and a main execution block.

#### 1. (Optional) Create a Custom Data Class
This class lets you keep track of information between frames, such as the total number of people detected and the number of frames processed. By inheriting from `app_callback_class`, you get built-in features like automatic frame counting (via `get_count()`). You can add any attributes you want to store custom statistics or state.

> **Note**: Frame counting is handled automatically by the framework. You do NOT need to call `user_data.increment()` in your callback - the framework wraps your callback function and handles this for you. Simply use `user_data.get_count()` to access the current frame number.

```python
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class

class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.total_people = 0
        self.total_frames = 0
```

#### 2. Implement the Callback Function
This function is called for every frame processed by the pipeline. This is where you put your custom logic for handling detections or other AI results.
In this toy example, the callback function receives the GStreamer buffer, extracts detection results, counts how many people are detected in the frame, updates the running totals, and prints out statistics.

```python
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import hailo

def app_callback(element, buffer, user_data):
    user_data.increment()  # Count the number of frames
    # buffer is passed directly
    if buffer is None:
        return
    detections = hailo.get_roi_from_buffer(buffer).get_objects_typed(hailo.HAILO_DETECTION)
    people_count = 0
    for det in detections:
        if det.get_label() == "person":
            people_count = people_count + 1
    user_data.total_people = user_data.total_people + people_count
    user_data.total_frames = user_data.total_frames + 1
    if user_data.total_frames > 0:
        running_average = user_data.total_people / user_data.total_frames
    else:
        running_average = 0.0
    string_to_print = (
        "Frame count: " + str(user_data.get_count()) + "\n"
        + "People detected in frame: " + str(people_count) + "\n"
        + "Running average people per frame: " + str(round(running_average, 2)) + "\n"
    )
    for detection in detections:
        string_to_print = string_to_print + (
            "Detection: " + detection.get_label() + " Confidence: " + str(round(detection.get_confidence(), 2)) + "\n"
        )
    print(string_to_print)
    return
```

#### 3. Write the Main Execution Block
This part ties everything together. It creates an instance of your callback class, sets up the detection pipeline, and starts the application. This is the entry point of your script.

```python
from hailo_apps.python.pipeline_apps.detection_simple.detection_simple_pipeline import GStreamerDetectionSimpleApp

if __name__ == "__main__":
    user_data = user_app_callback_class()
    app = GStreamerDetectionSimpleApp(app_callback, user_data)
    app.run()
```

### Putting It All Together
Below is a complete, minimal example you can copy and run. It combines the custom class, callback function, and main block into a single script. This is a good starting point for your own application—just modify the callback logic as needed for your use case.

```python
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import hailo
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.python.pipeline_apps.detection_simple.detection_simple_pipeline import GStreamerDetectionSimpleApp

class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.total_people = 0
        self.total_frames = 0

def app_callback(pad, info, user_data):
    # Note: Frame counting is handled automatically by the framework wrapper
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
    detections = hailo.get_roi_from_buffer(buffer).get_objects_typed(hailo.HAILO_DETECTION)
    people_count = 0
    for det in detections:
        if det.get_label() == "person":
            people_count = people_count + 1
    user_data.total_people = user_data.total_people + people_count
    user_data.total_frames = user_data.total_frames + 1
    if user_data.total_frames > 0:
        running_average = user_data.total_people / user_data.total_frames
    else:
        running_average = 0.0
    string_to_print = (
        "Frame count: " + str(user_data.get_count()) + "\n"
        + "People detected in frame: " + str(people_count) + "\n"
        + "Running average people per frame: " + str(round(running_average, 2)) + "\n"
    )
    for detection in detections:
        string_to_print = string_to_print + (
            "Detection: " + detection.get_label() + " Confidence: " + str(round(detection.get_confidence(), 2)) + "\n"
        )
    print(string_to_print)
    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    user_data = user_app_callback_class()
    app = GStreamerDetectionSimpleApp(app_callback, user_data)
    app.run()
```

This script demonstrates how to:
- Define a callback class to keep state.
- Implement a simple callback function to process detections and print statistics.
- Set up and run the detection pipeline.


### How to Run Your Application
To run your application, execute your Python script from the terminal. The application will automatically handle command-line arguments for inputs, HEF files, and other parameters.
For example, to run the example code above paste it in 'example.py' and run the application with the USB camera as input:
```bash
python example.py --input usb
```

---

## Development Path 2: Advanced (Pipeline Modification)

You can build your own pipelines from scratch or modify our examples to suit different data flow requirements.

### How to Build a Custom Pipeline
Building a custom pipeline involves creating a Python class that inherits from our `GStreamerApp` base class. This approach allows you to leverage the core application logic (like window management and bus handling) while defining your own unique pipeline structure.

The process is straightforward:

1.  **Create a Class Inheriting from `GStreamerApp`**: Your application will be a new class that extends `GStreamerApp`. In the constructor (`__init__`), you'll call the parent constructor and set up any application-specific parameters, such as HEF paths, video sources, or model thresholds.

2.  **Override the `get_pipeline_string` Method**: This is where you define your GStreamer pipeline. You must implement this method in your class. Inside, you'll use the helper functions from `gstreamer_helper_pipelines.py` (like `SOURCE_PIPELINE`, `INFERENCE_PIPELINE`, etc.) to build the pipeline string piece by piece.

3.  **Run Your Application**: In your main execution block, you instantiate your new class and call its `run()` method, which handles the pipeline setup and execution.

### Example: Building a Simple Detection Pipeline

Here is a simplified example based on `detection_simple_pipeline.py` that illustrates the concept:

```python
# Import necessary classes and pipeline helpers
from hailo_apps.python.core.gstreamer.gstreamer_app import GStreamerApp
from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import (
    SOURCE_PIPELINE, INFERENCE_PIPELINE, DISPLAY_PIPELINE
)

# 1. Create a class that inherits from GStreamerApp
class MyCustomPipelineApp(GStreamerApp):
    def __init__(self, args, user_data):
        # Call the parent constructor
        super().__init__(args, user_data)

        # Set up application-specific properties
        self.hef_path = "path/to/your/model.hef"
        self.post_process_so = "path/to/your/postprocess.so"
        # ... other parameters

    # 2. Override this method to define your pipeline
    def get_pipeline_string(self):
        # Use helper functions to build the pipeline components
        source = SOURCE_PIPELINE(video_source=self.video_source, ...)
        infer = INFERENCE_PIPELINE(hef_path=self.hef_path, ...)
        display = DISPLAY_PIPELINE(...)

        # Link the components together with '!'
        pipeline_string = f"{source} ! {infer} ! {display}"
        print(pipeline_string)
        return pipeline_string

# 3. Run the application
if __name__ == "__main__":
    # app_callback and user_data are for the basic path; can be simple for this case
    app = MyCustomPipelineApp(app_callback=dummy_callback, user_data=app_callback_class())
    app.run()
```
You have full control to reorder, remove, or add new GStreamer elements in the string returned by `get_pipeline_string` to create your desired data flow.

> **Note**: For detailed information about all available helper functions and their parameters, see the [GStreamer Helper Pipelines Reference](./gstreamer_helper_pipelines.md).

### Common Architectural Patterns
The following patterns are examples for commonly used pipeline architectures. Most common patterns are already implemented in the GStreamerHelperPipelines.
It is highly recommended to use the helper functions to build your pipeline, but you can also build your own pipeline string from scratch.

#### 1. Single Network Pipeline
**Use case:** Run a single AI model (e.g., object detection) on a video stream.

```mermaid
flowchart LR
    A[Source] --> B[Video conversion] --> C[hailonet] --> D[hailofilter] --> E[hailooverlay] --> F[Display]
```
A high level code example for building this pipeline using the helper functions:
```python
from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import (
    SOURCE_PIPELINE, INFERENCE_PIPELINE, DISPLAY_PIPELINE
)

pipeline_string = (
    f"{SOURCE_PIPELINE(video_source='input.mp4')} ! "
    f"{INFERENCE_PIPELINE(hef_path='model.hef', post_process_so='post.so')} ! "
    f"{DISPLAY_PIPELINE()}"
)
```

> **See also**: For complete examples and detailed function documentation, refer to the [GStreamer Helper Pipelines Reference](./gstreamer_helper_pipelines.md#complete-pipeline-examples).

#### 2. Wrapped Inference for Resolution Preservation
**Use case:** Run inference on a scaled-down version of the video for performance, but display the original high-res video with overlays.

```mermaid
flowchart LR
    A[Source] --> C
    subgraph INFERENCE_PIPELINE_WRAP
        C[hailocropper]
        C -- Resized Frame --> I[Inner Pipeline]
        C -- Original Frame --> Q[Queue]
        I --> AGG(hailoaggregator)
        Q --> AGG
    end
    AGG -- Original Frame with AI Metadata --> F[hailooverlay] --> G[Display]
```

A high level code example for building this pipeline using the helper functions:
```python
from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import (
    SOURCE_PIPELINE, INFERENCE_PIPELINE, INFERENCE_PIPELINE_WRAPPER, DISPLAY_PIPELINE
)

inner_infer = INFERENCE_PIPELINE(hef_path='model.hef', post_process_so='post.so')
inner_infer_wrapper = INFERENCE_PIPELINE_WRAPPER(inner_infer)
pipeline_string = (
    f"{SOURCE_PIPELINE(video_source='input.mp4')} ! "
    f"{inner_infer_wrapper} ! "
    f"{DISPLAY_PIPELINE()}"
)
```

> **See also**: For detailed documentation on `INFERENCE_PIPELINE_WRAPPER`, see the [GStreamer Helper Pipelines Reference](./gstreamer_helper_pipelines.md#inference-pipeline-functions).


#### 3. Cascaded Networks Pipeline
**Use case:** Run two or more models in series (e.g., detection followed by classification on detected regions).

```mermaid
flowchart LR
    A[Source] --> B[NN1: Detection] --> C
    subgraph CROPPER_PIPELINE
        C[hailocropper]
        C -- Cropped Detections --> D[NN2: Classification]
        C -- Original Frame --> Q[Queue]
        D --> AGG(hailoaggregator)
        Q --> AGG
    end
    AGG -- Detections with class --> F[Display]
```

A high level code example for building this pipeline using the helper functions.
Note that the `CROPPER_PIPELINE` is a helper function that is used to crop the detections from the first network and pass them to the second network. You can control the cropper by passing a custom C++ function to it.

```python
from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import (
    SOURCE_PIPELINE, INFERENCE_PIPELINE, CROPPER_PIPELINE, DISPLAY_PIPELINE
)

first_infer = INFERENCE_PIPELINE(hef_path='detector.hef', post_process_so='detector_post.so')
second_infer = INFERENCE_PIPELINE(hef_path='classifier.hef', post_process_so='classifier_post.so')
cropper = CROPPER_PIPELINE(second_infer, so_path='cropper.so', function_name='crop_func')
pipeline_string = (
    f"{SOURCE_PIPELINE(video_source='input.mp4')} ! "
    f"{first_infer} ! {cropper} ! {DISPLAY_PIPELINE()}"
)
```

> **See also**: For detailed documentation on `CROPPER_PIPELINE` and its parameters, see the [GStreamer Helper Pipelines Reference](./gstreamer_helper_pipelines.md#cropping-and-tiling-functions).

#### 4. Parallel Networks Pipeline
**Use case:** Run multiple models in parallel on the same input stream and combine their results.

```mermaid
flowchart TD
    A[Source] --> T(tee)
    T --> C1[NN Chain 1] --> M(hailomuxer)
    T --> C2[NN Chain 2] --> M
    M --> E[Overlay] --> F[Display]
```

A high level code example for building this pipeline using the helper functions.

```python
from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import (
    SOURCE_PIPELINE, INFERENCE_PIPELINE, DISPLAY_PIPELINE
)

pipeline_string = (
    f"{SOURCE_PIPELINE(video_source='input.mp4')} ! tee name=t "
    f"t. ! {INFERENCE_PIPELINE(hef_path='model1.hef', post_process_so='post1.so')} ! hailomuxer name=m "
    f"t. ! {INFERENCE_PIPELINE(hef_path='model2.hef', post_process_so='post2.so')} ! m. "
    f"m. ! {DISPLAY_PIPELINE()}"
)
```

#### 5. Tiled Inference Pipeline
**Use case:** Split a high-res frame into tiles, run inference on each, and aggregate results for display.

```mermaid
flowchart LR
    A[Source] --> C
    subgraph TILE_CROPPER_PIPELINE
        C[hailotilecropper]
        C -- Tile --> I[Inner Pipeline]
        C -- Original Frame --> Q[Queue]
        I --> AGG(hailotileaggregator)
        Q --> AGG
    end
    AGG -- Original Frame with AI Metadata --> F[hailooverlay] --> G[Display]
```

A high level code example for building this pipeline using the helper functions:

```python
from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import (
    SOURCE_PIPELINE, INFERENCE_PIPELINE, TILE_CROPPER_PIPELINE, DISPLAY_PIPELINE
)

tile_inference = INFERENCE_PIPELINE(hef_path='model.hef', post_process_so='post.so')
tile_cropper = TILE_CROPPER_PIPELINE(
    inner_pipeline=tile_inference,
    tiles_along_x_axis=4,
    tiles_along_y_axis=3,
    overlap_x_axis=0.1
)
pipeline_string = (
    f"{SOURCE_PIPELINE(video_source='input.mp4')} ! "
    f"{tile_cropper} ! {DISPLAY_PIPELINE()}"
)
```

> **See also**: For detailed documentation on `TILE_CROPPER_PIPELINE` and its parameters, see the [GStreamer Helper Pipelines Reference](./gstreamer_helper_pipelines.md#tile_cropper_pipeline).

## Additional Topics
### Retraining your own models
See [Retraining your own models](retraining_example.md) for more information.

### Bring Your Own Models
Compiling your own models is out of the scope of this repository.
See our [hailo_model_zoo](https://github.com/hailo-ai/hailo_model_zoo) for additional supported models.
For adding your own models, see our Data Flow Compiler (DFC) documentation in [Hailo Developer Zone (Requires registration)](https://developer.hailo.ai/docs/hailo-data-flow-compiler) for more information.

#### Post Process function
Once you have a new model, you will need to add a Post Process Function to the model.
See [Writing Your Own Post-Process for Hailo Apps](writing_postprocess.md) for more information.