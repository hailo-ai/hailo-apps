# GStreamer Helper Pipelines Reference

This document provides a comprehensive reference for the `gstreamer_helper_pipelines.py` module, which contains helper functions for building GStreamer pipeline strings in a modular and maintainable way.

> **Note**: This is a reference document. For an introduction to building applications and understanding the development workflow, see the [Application Development Guide](./app_development.md).

## Overview

The `gstreamer_helper_pipelines` module provides a set of Python functions that generate GStreamer pipeline strings. These functions abstract away the complexity of manually constructing GStreamer pipeline strings and provide a consistent, maintainable way to build pipelines for Hailo AI applications.

This reference complements the [Application Development Guide](./app_development.md) by providing detailed documentation for all helper functions, their parameters, and complete usage examples.

## Table of Contents

- [Utility Functions](#utility-functions)
- [Source Pipeline Functions](#source-pipeline-functions)
- [Inference Pipeline Functions](#inference-pipeline-functions)
- [Display and Output Functions](#display-and-output-functions)
- [Cropping and Tiling Functions](#cropping-and-tiling-functions)
- [Tracking Functions](#tracking-functions)
- [Streaming and Shared Memory Functions](#streaming-and-shared-memory-functions)
- [Callback Functions](#callback-functions)

---

## Utility Functions

### QUEUE

Creates a GStreamer queue element string with the specified parameters.

**Parameters:**
- `name` (str): The name of the queue element.
- `max_size_buffers` (int, optional): The maximum number of buffers that the queue can hold. Defaults to 3.
- `max_size_bytes` (int, optional): The maximum size in bytes that the queue can hold. Defaults to 0 (unlimited).
- `max_size_time` (int, optional): The maximum size in time that the queue can hold. Defaults to 0 (unlimited).
- `leaky` (str, optional): The leaky type of the queue. Can be 'no', 'upstream', or 'downstream'. Defaults to 'no'.

**Returns:**
- `str`: A string representing the GStreamer queue element with the specified parameters.

**Example:**
```python
queue_str = QUEUE(name="my_queue", max_size_buffers=10, leaky="downstream")
```

### get_source_type

Determines the source type based on the input source string.

**Parameters:**
- `input_source` (str): The input source identifier.

**Returns:**
- `str`: The source type, which can be one of: "usb", "rpi", "libcamera", "ximage", "rtsp", or "file".

**Source Type Detection:**
- `/dev/video*` → "usb"
- Starts with `rpi` → "rpi"
- Starts with `libcamera` → "libcamera"
- Starts with `0x` → "ximage"
- Starts with `rtsp://` → "rtsp"
- Otherwise → "file"

### get_camera_resolution

Returns a standard camera resolution based on the video resolution required.

**Parameters:**
- `video_width` (int, optional): The required width. Defaults to 640.
- `video_height` (int, optional): The required height. Defaults to 640.

**Returns:**
- `tuple`: A tuple of (width, height) representing the closest standard resolution.

**Standard Resolutions:**
- ≤ 640x480 → 640x480
- ≤ 1280x720 → 1280x720
- ≤ 1920x1080 → 1920x1080
- Otherwise → 3840x2160

---

## Source Pipeline Functions

### SOURCE_PIPELINE

Creates a GStreamer pipeline string for the video source with frame rate control.

**Parameters:**
- `video_source` (str): The path or device name of the video source (e.g., `/dev/video0`, `input.mp4`, `rtsp://...`).
- `video_width` (int, optional): The width of the video. Defaults to 640.
- `video_height` (int, optional): The height of the video. Defaults to 640.
- `name` (str, optional): The prefix name for the pipeline elements. Defaults to 'source'.
- `no_webcam_compression` (bool, optional): When True, uses uncompressed format for USB cameras (only low resolution supported). Defaults to False.
- `frame_rate` (int, optional): Target frame rate. Defaults to 30.
- `sync` (bool, optional): Whether to synchronize frame rate. Defaults to True.
- `video_format` (str, optional): The video format (e.g., 'RGB'). Defaults to 'RGB'.
- `mirror_image` (bool, optional): Whether to horizontally mirror the image for camera sources (USB and RPI cameras). Defaults to True. Set to False to disable mirroring.

**Returns:**
- `str`: A string representing the GStreamer pipeline for the video source.

**Supported Source Types:**
- **USB cameras** (`/dev/video*`): Uses `v4l2src` with optional JPEG compression
- **RPI cameras** (`rpi*`): Uses `appsrc` for Raspberry Pi camera interface
- **LibCamera** (`libcamera*`): Uses `libcamerasrc` element
- **XImage** (`0x*`): Uses `ximagesrc` for X11 screen capture
- **RTSP streams** (`rtsp://*`): Uses `rtspsrc` for network streams
- **File sources**: Uses `filesrc` with `decodebin` for video files

**Example:**
```python
# USB camera with mirroring disabled
source = SOURCE_PIPELINE(
    video_source="/dev/video0",
    video_width=1280,
    video_height=720,
    mirror_image=False
)

# RTSP stream
source = SOURCE_PIPELINE(
    video_source="rtsp://192.168.1.100:554/stream",
    video_width=1920,
    video_height=1080
)

# Video file
source = SOURCE_PIPELINE(video_source="input.mp4")
```

---

## Inference Pipeline Functions

### INFERENCE_PIPELINE

Creates a GStreamer pipeline string for inference and post-processing using a Hailo HEF file.

**Function Signature:**
```python
INFERENCE_PIPELINE(
    hef_path,
    post_process_so=None,
    batch_size=1,
    config_json=None,
    post_function_name=None,
    additional_params="",
    name="inference",
    scheduler_timeout_ms=None,
    scheduler_priority=None,
    vdevice_group_id="SHARED",
    multi_process_service=None
)
```

**Parameters:**
- `hef_path` (str): Path to the HEF (Hailo Executable Format) file.
- `post_process_so` (str or None): Path to the post-processing .so file. If None, post-processing is skipped.
- `batch_size` (int): Batch size for hailonet. Defaults to 1.
- `config_json` (str or None): Config JSON for post-processing (e.g., label mapping).
- `post_function_name` (str or None): Function name in the .so postprocess library.
- `additional_params` (str): Additional parameters appended to hailonet. Defaults to "".
- `name` (str): Prefix name for pipeline elements. Defaults to 'inference'.
- `scheduler_timeout_ms` (int or None): hailonet scheduler-timeout-ms. Defaults to None.
- `scheduler_priority` (int or None): hailonet scheduler-priority. Defaults to None.
- `vdevice_group_id` (str): hailonet vdevice-group-id. Defaults to "SHARED" (SHARED_VDEVICE_GROUP_ID constant).
- `multi_process_service` (bool or None): hailonet multi-process-service. Defaults to None.

**Returns:**
- `str`: A string representing the GStreamer pipeline for inference.

**Note:**
For a full list of hailonet options, run `gst-inspect-1.0 hailonet`.

**Example:**
```python
inference = INFERENCE_PIPELINE(
    hef_path="models/yolov5.hef",
    post_process_so="postprocess/yolov5_post.so",
    config_json="configs/yolov5_labels.json",
    post_function_name="yolov5_postprocess",
    batch_size=1
)
```

### INFERENCE_PIPELINE_WRAPPER

Creates a GStreamer pipeline string that wraps an inner pipeline with a hailocropper and hailoaggregator. This allows keeping the original video resolution and color-space (format) of the input frame.

**Parameters:**
- `inner_pipeline` (str): The inner pipeline string to be wrapped.
- `bypass_max_size_buffers` (int, optional): The maximum number of buffers for the bypass queue. Defaults to 20.
- `name` (str, optional): The prefix name for the pipeline elements. Defaults to 'inference_wrapper'.

**Returns:**
- `str`: A string representing the GStreamer pipeline for the inference wrapper.

**Use Case:**
Run inference on a scaled-down version of the video for performance, but display the original high-res video with overlays.

**Example:**
```python
inner_infer = INFERENCE_PIPELINE(
    hef_path="model.hef",
    post_process_so="post.so"
)
wrapped_infer = INFERENCE_PIPELINE_WRAPPER(inner_infer)
```

---

## Display and Output Functions

### OVERLAY_PIPELINE

Creates a GStreamer pipeline string for the hailooverlay element, which draws bounding boxes and labels on the video.

**Parameters:**
- `name` (str, optional): The prefix name for the pipeline elements. Defaults to 'hailo_overlay'.

**Returns:**
- `str`: A string representing the GStreamer pipeline for the hailooverlay element.

**Example:**
```python
overlay = OVERLAY_PIPELINE(name="my_overlay")
```

### DISPLAY_PIPELINE

Creates a GStreamer pipeline string for displaying the video with overlays.

**Parameters:**
- `video_sink` (str, optional): The video sink element to use. Defaults to 'autovideosink'.
- `sync` (str, optional): The sync property for the video sink. Defaults to 'true'.
- `show_fps` (str, optional): Whether to show the FPS on the video sink. Should be 'true' or 'false'. Defaults to 'false'.
- `name` (str, optional): The prefix name for the pipeline elements. Defaults to 'hailo_display'.

**Returns:**
- `str`: A string representing the GStreamer pipeline for displaying the video.

**Example:**
```python
display = DISPLAY_PIPELINE(
    video_sink="xvimagesink",
    show_fps="true"
)
```

### FILE_SINK_PIPELINE

Creates a GStreamer pipeline string for saving the video to a file in .mkv format.

**Parameters:**
- `output_file` (str): The path to the output file. Defaults to "output.mkv".
- `name` (str, optional): The prefix name for the pipeline elements. Defaults to 'file_sink'.
- `bitrate` (int, optional): The bitrate for the encoder in kbps. Defaults to 5000.

**Returns:**
- `str`: A string representing the GStreamer pipeline for saving the video to a file.

**Note:**
It is recommended to run `ffmpeg` to fix the file header after recording:
```bash
ffmpeg -i output.mkv -c copy fixed_output.mkv
```

**Note:** If your source is a file, looping will not work with this pipeline.

**Example:**
```python
file_sink = FILE_SINK_PIPELINE(
    output_file="recorded_video.mkv",
    bitrate=8000
)
```

### UI_APPSINK_PIPELINE

Creates a GStreamer pipeline string for the UI appsink element. This pipeline is used to send video frames to a UI application.

**Parameters:**
- `name` (str, optional): The prefix name for the pipeline elements. Defaults to 'ui_sink'.
- `sync` (str, optional): The sync property for the appsink. Defaults to 'true'.
- `show_fps` (str, optional): Whether to show FPS (currently unused). Defaults to 'false'.

**Returns:**
- `str`: A string representing the GStreamer pipeline for the UI appsink element.

**Example:**
```python
ui_sink = UI_APPSINK_PIPELINE(name="my_ui_sink", sync="false")
```

---

## Cropping and Tiling Functions

### CROPPER_PIPELINE

Wraps an inner pipeline with hailocropper and hailoaggregator. The cropper will crop detections made by earlier stages in the pipeline. Each detection is cropped and sent to the inner pipeline for further processing.

**Parameters:**
- `inner_pipeline` (str): The pipeline string to be wrapped.
- `so_path` (str): The path to the cropper .so library.
- `function_name` (str): The function name in the .so library.
- `use_letterbox` (bool): Whether to preserve aspect ratio. Defaults to True.
- `no_scaling_bbox` (bool): If True, bounding boxes are not scaled. Defaults to True.
- `internal_offset` (bool): If True, uses internal offsets. Defaults to True.
- `resize_method` (str): The resize method. Defaults to 'bilinear'.
- `bypass_max_size_buffers` (int): For the bypass queue. Defaults to 20.
- `name` (str): A prefix name for pipeline elements. Defaults to 'cropper_wrapper'.

**Returns:**
- `str`: A pipeline string representing hailocropper + aggregator around the inner_pipeline.

**Use Case:**
After face detection pipeline stage, crop the faces and send them to a face recognition pipeline.

**Example:**
```python
face_recognition = INFERENCE_PIPELINE(
    hef_path="face_recognition.hef",
    post_process_so="face_recognition_post.so"
)
cropper = CROPPER_PIPELINE(
    inner_pipeline=face_recognition,
    so_path="cropper.so",
    function_name="crop_faces"
)
```

### TILE_CROPPER_PIPELINE

Wraps an inner pipeline with hailotilecropper and hailotileaggregator. The tile cropper divides the input frame into tiles based on the specified tiling parameters.

**Parameters:**
- `inner_pipeline` (str): The pipeline string to be wrapped for processing each tile.
- `name` (str): A prefix name for pipeline elements. Defaults to 'tile_cropper_wrapper'.
- `internal_offset` (bool): If True, uses internal offsets for cropping. Defaults to True.
- `scale_level` (int): The scaling level for the tiles. Defaults to 2.
- `tiling_mode` (int): The tiling mode (e.g., 1 for uniform tiling). Defaults to 1.
- `tiles_along_x_axis` (int): Number of tiles along the x-axis. Defaults to 4.
- `tiles_along_y_axis` (int): Number of tiles along the y-axis. Defaults to 3.
- `overlap_x_axis` (float): Overlap percentage between tiles along the x-axis. Defaults to 0.1.
- `overlap_y_axis` (float): Overlap percentage between tiles along the y-axis. Defaults to 0.08.
- `iou_threshold` (float): Intersection-over-Union (IoU) threshold for combining detections. Defaults to 0.3.
- `border_threshold` (float): Threshold for handling detections near tile borders. Defaults to 0.1.

**Returns:**
- `str`: A pipeline string representing hailotilecropper + hailotileaggregator around the inner_pipeline.

**Note:**
Single scaling requires `tiling_mode=0` & `border_threshold=0`.

**Use Case:**
Split a high-res frame into tiles, run inference on each, and aggregate results.

**Example:**
```python
tile_inference = INFERENCE_PIPELINE(
    hef_path="model.hef",
    post_process_so="post.so"
)
tile_cropper = TILE_CROPPER_PIPELINE(
    inner_pipeline=tile_inference,
    tiles_along_x_axis=4,
    tiles_along_y_axis=3,
    overlap_x_axis=0.1
)
```

---

## Tracking Functions

### TRACKER_PIPELINE

Creates a GStreamer pipeline string for the HailoTracker element, which tracks detected objects across video frames.

**Parameters:**
- `class_id` (int, required): The class ID to track. Use -1 to track across all classes. This parameter is required (no default value).
- `kalman_dist_thr` (float, optional): Threshold used in Kalman filter to compare Mahalanobis cost matrix. Closer to 1.0 is looser. Defaults to 0.8.
- `iou_thr` (float, optional): Threshold used in Kalman filter to compare IOU cost matrix. Closer to 1.0 is looser. Defaults to 0.9.
- `init_iou_thr` (float, optional): Threshold used in Kalman filter to compare IOU cost matrix of newly found instances. Closer to 1.0 is looser. Defaults to 0.7.
- `keep_new_frames` (int, optional): Number of frames to keep without a successful match before a 'new' instance is removed. Defaults to 2.
- `keep_tracked_frames` (int, optional): Number of frames to keep without a successful match before a 'tracked' instance is considered 'lost'. Defaults to 15.
- `keep_lost_frames` (int, optional): Number of frames to keep without a successful match before a 'lost' instance is removed. Defaults to 2.
- `keep_past_metadata` (bool, optional): Whether to keep past metadata on tracked objects. Defaults to False.
- `qos` (bool, optional): Whether to enable QoS. Defaults to False.
- `name` (str, optional): The prefix name for the pipeline elements. Defaults to 'hailo_tracker'.

**Returns:**
- `str`: A string representing the GStreamer pipeline for the HailoTracker element.

**Note:**
For a full list of options and their descriptions, run `gst-inspect-1.0 hailotracker`.

**Example:**
```python
tracker = TRACKER_PIPELINE(
    class_id=-1,  # Track all classes
    kalman_dist_thr=0.8,
    iou_thr=0.9
)
```

---

## Streaming and Shared Memory Functions

### VIDEO_STREAM_PIPELINE

Creates a GStreamer pipeline string portion for encoding and streaming video over UDP.

**Parameters:**
- `port` (int): UDP port number. Defaults to 5004.
- `host` (str): Destination IP address. Defaults to "127.0.0.1".
- `bitrate` (int): Target bitrate for x264enc in kbps. Defaults to 2048.

**Returns:**
- `str`: GStreamer pipeline string fragment.

**Note:**
Using x264enc with zerolatency tune. Hardware encoders (e.g., `omxh264enc`, `v4l2h264enc`, `vaapih264enc`) are preferable on embedded systems.

**Example:**
```python
stream = VIDEO_STREAM_PIPELINE(
    port=5004,
    host="192.168.1.100",
    bitrate=4000
)
```

### VIDEO_SHMSINK_PIPELINE

Creates a GStreamer pipeline string portion for shared memory video transfer using the shm plugins. Shmsink creates a shared memory segment and socket.

**Parameters:**
- `socket_path` (str, optional): Socket path for the shared memory segment. Defaults to None.

**Returns:**
- `str`: GStreamer pipeline string fragment.

**Example:**
```python
shm_sink = VIDEO_SHMSINK_PIPELINE(socket_path="/tmp/video_socket")
```

### VIDEO_SHMSRC_PIPELINE

Creates a GStreamer pipeline string portion for shared memory video transfer using the shm plugins. Shmsrc connects to that segment and reads video frames.

**Parameters:**
- `socket_path` (str, optional): Socket path for the shared memory segment. Defaults to None.

**Returns:**
- `str`: GStreamer pipeline string fragment.

**Example:**
```python
shm_src = VIDEO_SHMSRC_PIPELINE(socket_path="/tmp/video_socket")
```

---

## Callback Functions

### USER_CALLBACK_PIPELINE

Creates a GStreamer pipeline string for the user callback element, which allows Python callbacks to be invoked during pipeline execution.

**Parameters:**
- `name` (str, optional): The prefix name for the pipeline elements. Defaults to 'identity_callback'.

**Returns:**
- `str`: A string representing the GStreamer pipeline for the user callback element.

**Example:**
```python
callback = USER_CALLBACK_PIPELINE(name="my_callback")
```

---

## Complete Pipeline Examples

### Example 1: Simple Detection Pipeline

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

### Example 2: Cascaded Networks (Detection + Classification)

```python
from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import (
    SOURCE_PIPELINE, INFERENCE_PIPELINE, CROPPER_PIPELINE, DISPLAY_PIPELINE
)

detector = INFERENCE_PIPELINE(
    hef_path='detector.hef',
    post_process_so='detector_post.so'
)
classifier = INFERENCE_PIPELINE(
    hef_path='classifier.hef',
    post_process_so='classifier_post.so'
)
cropper = CROPPER_PIPELINE(
    inner_pipeline=classifier,
    so_path='cropper.so',
    function_name='crop_func'
)

pipeline_string = (
    f"{SOURCE_PIPELINE(video_source='input.mp4')} ! "
    f"{detector} ! {cropper} ! {DISPLAY_PIPELINE()}"
)
```

### Example 3: Resolution Preservation with Wrapper

```python
from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import (
    SOURCE_PIPELINE, INFERENCE_PIPELINE, INFERENCE_PIPELINE_WRAPPER, DISPLAY_PIPELINE
)

inner_infer = INFERENCE_PIPELINE(
    hef_path='model.hef',
    post_process_so='post.so'
)
wrapped_infer = INFERENCE_PIPELINE_WRAPPER(inner_infer)

pipeline_string = (
    f"{SOURCE_PIPELINE(video_source='input.mp4')} ! "
    f"{wrapped_infer} ! {DISPLAY_PIPELINE()}"
)
```

### Example 4: Tracking Pipeline

```python
from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import (
    SOURCE_PIPELINE, INFERENCE_PIPELINE, TRACKER_PIPELINE, DISPLAY_PIPELINE
)

inference = INFERENCE_PIPELINE(
    hef_path='model.hef',
    post_process_so='post.so'
)
tracker = TRACKER_PIPELINE(class_id=-1)

pipeline_string = (
    f"{SOURCE_PIPELINE(video_source='/dev/video0')} ! "
    f"{inference} ! {tracker} ! {DISPLAY_PIPELINE()}"
)
```

### Example 5: Tiled Inference Pipeline

```python
from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import (
    SOURCE_PIPELINE, INFERENCE_PIPELINE, TILE_CROPPER_PIPELINE, DISPLAY_PIPELINE
)

tile_inference = INFERENCE_PIPELINE(
    hef_path='model.hef',
    post_process_so='post.so'
)
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

---

## Best Practices

1. **Use Helper Functions**: Always use the helper functions instead of manually constructing pipeline strings. This ensures consistency and maintainability.

2. **Naming Conventions**: Use descriptive `name` parameters to make debugging easier, especially when using multiple instances of the same pipeline component.

3. **Queue Sizing**: Adjust queue sizes (`max_size_buffers`) based on your use case. Larger queues can help with buffering but consume more memory.

4. **Camera Mirroring**: For camera sources, consider whether mirroring is appropriate for your use case. Use `mirror_image=False` when you need the actual camera orientation.

5. **Performance Tuning**: For embedded systems, consider using hardware encoders (e.g., `omxh264enc` for Raspberry Pi) instead of software encoders.

6. **Error Handling**: Always validate pipeline strings before passing them to GStreamer. Print the pipeline string for debugging purposes.

---

## Additional Resources

- [GStreamer Documentation](https://gstreamer.freedesktop.org/documentation/)
- [Hailo Tappas Elements Documentation](https://github.com/hailo-ai/tappas/tree/master/docs/elements)
- [Application Development Guide](./app_development.md)

