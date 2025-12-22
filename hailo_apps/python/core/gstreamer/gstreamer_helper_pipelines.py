import os

from hailo_apps.python.core.common.defines import (
    GST_VIDEO_SINK,
    TAPPAS_POSTPROC_PATH_DEFAULT,
    TAPPAS_POSTPROC_PATH_KEY,
    SHARED_VDEVICE_GROUP_ID
)


def get_source_type(input_source):
    # This function will return the source type based on the input source
    # return values can be "file", "mipi" or "usb"
    input_source = str(input_source)
    if input_source.startswith("/dev/video"):
        return "usb"
    elif input_source.startswith("rpi"):
        return "rpi"
    elif input_source.startswith("libcamera"):  # Use libcamerasrc element, not suggested
        return "libcamera"
    elif input_source.startswith("0x"):
        return "ximage"
    elif input_source.startswith('rtsp://'):
        return 'rtsp'
    else:
        return "file"


def QUEUE(name, max_size_buffers=3, max_size_bytes=0, max_size_time=0, leaky="no"):
    """Creates a GStreamer queue element string with the specified parameters.

    Args:
        name (str): The name of the queue element.
        max_size_buffers (int, optional): The maximum number of buffers that the queue can hold. Defaults to 3.
        max_size_bytes (int, optional): The maximum size in bytes that the queue can hold. Defaults to 0 (unlimited).
        max_size_time (int, optional): The maximum size in time that the queue can hold. Defaults to 0 (unlimited).
        leaky (str, optional): The leaky type of the queue. Can be 'no', 'upstream', or 'downstream'. Defaults to 'no'.

    Returns:
        str: A string representing the GStreamer queue element with the specified parameters.
    """
    q_string = f"queue name={name} leaky={leaky} max-size-buffers={max_size_buffers} max-size-bytes={max_size_bytes} max-size-time={max_size_time} "
    return q_string


def get_camera_resolution(video_width=640, video_height=640):
    # This function will return a standard camera resolution based on the video resolution required
    # Standard resolutions are 640x480, 1280x720, 1920x1080, 3840x2160
    # If the required resolution is not standard, it will return the closest standard resolution
    if video_width <= 640 and video_height <= 480:
        return 640, 480
    elif video_width <= 1280 and video_height <= 720:
        return 1280, 720
    elif video_width <= 1920 and video_height <= 1080:
        return 1920, 1080
    else:
        return 3840, 2160


def SOURCE_PIPELINE(
    video_source,
    video_width=640,
    video_height=640,
    name="source",
    no_webcam_compression=False,
    frame_rate=30,
    sync=True,
    video_format="RGB",
    mirror_image=True,
):
    """Creates a GStreamer pipeline string for the video source with a separate fps caps
    for frame rate control.

    Args:
        video_source (str): The path or device name of the video source.
        video_width (int, optional): The width of the video. Defaults to 640.
        video_height (int, optional): The height of the video. Defaults to 640.
        video_format (str, optional): The video format. Defaults to 'RGB'.
        name (str, optional): The prefix name for the pipeline elements. Defaults to 'source'.
        mirror_image (bool, optional): Whether to horizontally mirror the image (for camera sources). Defaults to True.

    Returns:
        str: A string representing the GStreamer pipeline for the video source.
    """
    source_type = get_source_type(video_source)

    # Build videoflip element string conditionally
    videoflip_str = f'videoflip name=videoflip_{name} video-direction=horiz ! ' if mirror_image else ''

    if source_type == "usb":
        if no_webcam_compression:
            # When using uncompressed format, only low resolution is supported
            source_element = (
                f'v4l2src device={video_source} name={name} ! '
                f'video/x-raw, width=640, height=480 ! '
                f'{videoflip_str}'
            )
        else:
            # Use compressed format for webcam
            width, height = get_camera_resolution(video_width, video_height)
            source_element = (
                f'v4l2src device={video_source} name={name} ! image/jpeg, framerate=30/1, width={width}, height={height} ! '
                f'{QUEUE(name=f"{name}_queue_decode")} ! '
                f'decodebin name={name}_decodebin ! '
                f'{videoflip_str}'
            )
    elif source_type == "rpi":
        rpi_videoflip_str = "videoflip name=videoflip video-direction=horiz ! " if mirror_image else ""
        source_element = (
            f"appsrc name=app_source is-live=true leaky-type=downstream max-buffers=3 ! "
            f"{rpi_videoflip_str}"
            f"video/x-raw, format={video_format}, width={video_width}, height={video_height} ! "
        )
    elif source_type == "libcamera":
        source_element = (
            f"libcamerasrc name={name} ! "
            f"video/x-raw, format={video_format}, width=1536, height=864 ! "
        )
    elif source_type == "ximage":
        source_element = (
            f"ximagesrc xid={video_source} ! {QUEUE(name=f'{name}queue_scale_')} ! videoscale ! "
        )
    elif source_type == 'rtsp':  # RTSP stream handling
        source_element = (
            f'rtspsrc location="{video_source}" name={name} ! '
            f'{QUEUE(name=f"{name}_queue_decode")} ! '
            f'decodebin name={name}_decodebin ! '
        )
    else:
        source_element = (
            f'filesrc location="{video_source}" name={name} ! '
            f"{QUEUE(name=f'{name}_queue_decode')} ! "
            f"decodebin name={name}_decodebin ! "
        )

    # Set up the fps caps.
    # If sync is True, constrain the rate with the given frame_rate.
    # Otherwise, pass through (no framerate limitation).
    if sync:
        fps_caps = f"video/x-raw, framerate={frame_rate}/1"
    else:
        fps_caps = "video/x-raw"

    source_pipeline = (
        f"{source_element} "
        f"{QUEUE(name=f'{name}_scale_q')} ! "
        f"videoscale name={name}_videoscale n-threads=2 ! "
        f"{QUEUE(name=f'{name}_convert_q')} ! "
        f"videoconvert n-threads=3 name={name}_convert qos=false ! "
        f"video/x-raw, pixel-aspect-ratio=1/1, format={video_format}, "
        f"width={video_width}, height={video_height} ! "
        f'videorate name={name}_videorate ! capsfilter name={name}_fps_caps caps="{fps_caps}" '
    )

    return source_pipeline


def INFERENCE_PIPELINE(
    hef_path,
    post_process_so=None,
    batch_size=1,
    config_json=None,
    post_function_name=None,
    additional_params="",
    name="inference",
    # Extra hailonet parameters
    scheduler_timeout_ms=None,
    scheduler_priority=None,
    vdevice_group_id=SHARED_VDEVICE_GROUP_ID,  # Don't change it - this is aligned across multiple apps
    multi_process_service=None,
):
    """Creates a GStreamer pipeline string for inference and post-processing using a user-provided shared object file.
    This pipeline includes videoscale and videoconvert elements to convert the video frame to the required format.
    The format and resolution are automatically negotiated based on the HEF file requirements.

    Args:
        hef_path (str): Path to the HEF file.
        post_process_so (str or None): Path to the post-processing .so file. If None, post-processing is skipped.
        batch_size (int): Batch size for hailonet (default=1).
        config_json (str or None): Config JSON for post-processing (e.g., label mapping).
        post_function_name (str or None): Function name in the .so postprocess.
        additional_params (str): Additional parameters appended to hailonet.
        name (str): Prefix name for pipeline elements (default='inference').

        # Extra hailonet parameters
        Run `gst-inspect-1.0 hailonet` for more information.
        vdevice_group_id (int): hailonet vdevice-group-id. Default=1.
        scheduler_timeout_ms (int or None): hailonet scheduler-timeout-ms. Default=None.
        scheduler_priority (int or None): hailonet scheduler-priority. Default=None.
        multi_process_service (bool or None): hailonet multi-process-service. Default=None.

    Returns:
        str: A string representing the GStreamer pipeline for inference.
    """
    # config & function strings
    config_str = f" config-path={config_json} " if config_json else ""
    function_name_str = f" function-name={post_function_name} " if post_function_name else ""
    vdevice_group_id_str = f" vdevice-group-id={vdevice_group_id} "
    multi_process_service_str = (
        f" multi-process-service={str(multi_process_service).lower()} "
        if multi_process_service is not None
        else ""
    )
    scheduler_timeout_ms_str = (
        f" scheduler-timeout-ms={scheduler_timeout_ms} " if scheduler_timeout_ms is not None else ""
    )
    scheduler_priority_str = (
        f" scheduler-priority={scheduler_priority} " if scheduler_priority is not None else ""
    )

    hailonet_str = (
        f"hailonet name={name}_hailonet "
        f"hef-path={hef_path} "
        f"batch-size={batch_size} "
        f"{vdevice_group_id_str}"
        # f"{multi_process_service_str}"
        f"{scheduler_timeout_ms_str}"
        f"{scheduler_priority_str}"
        f"{additional_params} "
        f"force-writable=true "
    )

    inference_pipeline = (
        f"{QUEUE(name=f'{name}_scale_q')} ! "
        f"videoscale name={name}_videoscale n-threads=2 qos=false ! "
        f"{QUEUE(name=f'{name}_convert_q')} ! "
        f"video/x-raw, pixel-aspect-ratio=1/1 ! "
        f"videoconvert name={name}_videoconvert n-threads=2 ! "
        f"{QUEUE(name=f'{name}_hailonet_q')} ! "
        f"{hailonet_str} ! "
    )

    if post_process_so:
        inference_pipeline += (
            f"{QUEUE(name=f'{name}_hailofilter_q')} ! "
            f"hailofilter name={name}_hailofilter so-path={post_process_so} {config_str} {function_name_str} qos=false ! "
        )

    inference_pipeline += f"{QUEUE(name=f'{name}_output_q')} "

    return inference_pipeline


def INFERENCE_PIPELINE_WRAPPER(
    inner_pipeline, bypass_max_size_buffers=20, name="inference_wrapper"
):
    """Creates a GStreamer pipeline string that wraps an inner pipeline with a hailocropper and hailoaggregator.
    This allows to keep the original video resolution and color-space (format) of the input frame.
    The inner pipeline should be able to do the required conversions and rescale the detection to the original frame size.

    Args:
        inner_pipeline (str): The inner pipeline string to be wrapped.
        bypass_max_size_buffers (int, optional): The maximum number of buffers for the bypass queue. Defaults to 20.
        name (str, optional): The prefix name for the pipeline elements. Defaults to 'inference_wrapper'.

    Returns:
        str: A string representing the GStreamer pipeline for the inference wrapper.
    """
    # Get the directory for post-processing shared objects
    tappas_post_process_dir = os.environ.get(TAPPAS_POSTPROC_PATH_KEY, TAPPAS_POSTPROC_PATH_DEFAULT)
    whole_buffer_crop_so = os.path.join(
        tappas_post_process_dir, "cropping_algorithms/libwhole_buffer.so"
    )

    # Construct the inference wrapper pipeline string
    inference_wrapper_pipeline = (
        f"{QUEUE(name=f'{name}_input_q')} ! "
        f"hailocropper name={name}_crop so-path={whole_buffer_crop_so} function-name=create_crops use-letterbox=true resize-method=inter-area internal-offset=true "
        f"hailoaggregator name={name}_agg "
        f"{name}_crop. ! {QUEUE(max_size_buffers=bypass_max_size_buffers, name=f'{name}_bypass_q')} ! {name}_agg.sink_0 "
        f"{name}_crop. ! {inner_pipeline} ! {name}_agg.sink_1 "
        f"{name}_agg. ! {QUEUE(name=f'{name}_output_q')} "
    )

    return inference_wrapper_pipeline


def OVERLAY_PIPELINE(name="hailo_overlay"):
    """Creates a GStreamer pipeline string for the hailooverlay element.
    This pipeline is used to draw bounding boxes and labels on the video.

    Args:
        name (str, optional): The prefix name for the pipeline elements. Defaults to 'hailo_overlay'.

    Returns:
        str: A string representing the GStreamer pipeline for the hailooverlay element.
    """
    # Construct the overlay pipeline string
    overlay_pipeline = f"{QUEUE(name=f'{name}_q')} ! hailooverlay name={name} "

    return overlay_pipeline


def DISPLAY_PIPELINE(
    video_sink=GST_VIDEO_SINK, sync="true", show_fps="false", name="hailo_display"
):
    """Creates a GStreamer pipeline string for displaying the video.
    It includes the hailooverlay plugin to draw bounding boxes and labels on the video.

    Args:
        video_sink (str, optional): The video sink element to use. Defaults to 'autovideosink'.
        sync (str, optional): The sync property for the video sink. Defaults to 'true'.
        show_fps (str, optional): Whether to show the FPS on the video sink. Should be 'true' or 'false'. Defaults to 'false'.
        name (str, optional): The prefix name for the pipeline elements. Defaults to 'hailo_display'.

    Returns:
        str: A string representing the GStreamer pipeline for displaying the video.
    """
    # Construct the display pipeline string
    display_pipeline = (
        f"{OVERLAY_PIPELINE(name=f'{name}_overlay')} ! "
        f"{QUEUE(name=f'{name}_videoconvert_q')} ! "
        f"videoconvert name={name}_videoconvert n-threads=2 qos=false ! "
        f"{QUEUE(name=f'{name}_q')} ! "
        f"fpsdisplaysink name={name} video-sink={video_sink} sync={sync} text-overlay={show_fps} signal-fps-measurements=true "
    )

    return display_pipeline


def FILE_SINK_PIPELINE(output_file="output.mkv", name="file_sink", bitrate=5000):
    """Creates a GStreamer pipeline string for saving the video to a file in .mkv format.
    It it recommended run ffmpeg to fix the file header after recording.
    example: ffmpeg -i output.mkv -c copy fixed_output.mkv
    Note: If your source is a file, looping will not work with this pipeline.

    Args:
        output_file (str): The path to the output file.
        name (str, optional): The prefix name for the pipeline elements. Defaults to 'file_sink'.
        bitrate (int, optional): The bitrate for the encoder. Defaults to 5000.

    Returns:
        str: A string representing the GStreamer pipeline for saving the video to a file.
    """
    # Construct the file sink pipeline string
    file_sink_pipeline = (
        f"{QUEUE(name=f'{name}_videoconvert_q')} ! "
        f"videoconvert name={name}_videoconvert n-threads=2 qos=false ! "
        f"{QUEUE(name=f'{name}_encoder_q')} ! "
        f"x264enc tune=zerolatency bitrate={bitrate} ! "
        f"matroskamux ! "
        f"filesink location={output_file} "
    )

    return file_sink_pipeline


def USER_CALLBACK_PIPELINE(name="identity_callback"):
    """Creates a GStreamer pipeline string for the user callback element.

    Args:
        name (str, optional): The prefix name for the pipeline elements. Defaults to 'identity_callback'.

    Returns:
        str: A string representing the GStreamer pipeline for the user callback element.
    """
    # Construct the user callback pipeline string
    user_callback_pipeline = f"{QUEUE(name=f'{name}_q')} ! identity name={name} "

    return user_callback_pipeline


def TRACKER_PIPELINE(
    class_id,
    kalman_dist_thr=0.8,
    iou_thr=0.9,
    init_iou_thr=0.7,
    keep_new_frames=2,
    keep_tracked_frames=15,
    keep_lost_frames=2,
    keep_past_metadata=False,
    qos=False,
    name="hailo_tracker",
):
    """Creates a GStreamer pipeline string for the HailoTracker element.

    Args:
        class_id (int): The class ID to track. Default is -1, which tracks across all classes.
        kalman_dist_thr (float, optional): Threshold used in Kalman filter to compare Mahalanobis cost matrix. Closer to 1.0 is looser. Defaults to 0.8.
        iou_thr (float, optional): Threshold used in Kalman filter to compare IOU cost matrix. Closer to 1.0 is looser. Defaults to 0.9.
        init_iou_thr (float, optional): Threshold used in Kalman filter to compare IOU cost matrix of newly found instances. Closer to 1.0 is looser. Defaults to 0.7.
        keep_new_frames (int, optional): Number of frames to keep without a successful match before a 'new' instance is removed from the tracking record. Defaults to 2.
        keep_tracked_frames (int, optional): Number of frames to keep without a successful match before a 'tracked' instance is considered 'lost'. Defaults to 15.
        keep_lost_frames (int, optional): Number of frames to keep without a successful match before a 'lost' instance is removed from the tracking record. Defaults to 2.
        keep_past_metadata (bool, optional): Whether to keep past metadata on tracked objects. Defaults to False.
        qos (bool, optional): Whether to enable QoS. Defaults to False.
        name (str, optional): The prefix name for the pipeline elements. Defaults to 'hailo_tracker'.

    Note:
        For a full list of options and their descriptions, run `gst-inspect-1.0 hailotracker`.

    Returns:
        str: A string representing the GStreamer pipeline for the HailoTracker element.
    """
    # Construct the tracker pipeline string
    tracker_pipeline = (
        f"hailotracker name={name} class-id={class_id} kalman-dist-thr={kalman_dist_thr} iou-thr={iou_thr} init-iou-thr={init_iou_thr} "
        f"keep-new-frames={keep_new_frames} keep-tracked-frames={keep_tracked_frames} keep-lost-frames={keep_lost_frames} keep-past-metadata={keep_past_metadata} qos={qos} ! "
        f"{QUEUE(name=f'{name}_q')} "
    )
    return tracker_pipeline


def CROPPER_PIPELINE(
    inner_pipeline,
    so_path,
    function_name,
    use_letterbox=True,
    no_scaling_bbox=True,
    internal_offset=True,
    resize_method="bilinear",
    bypass_max_size_buffers=20,
    name="cropper_wrapper",
):
    """Wraps an inner pipeline with hailocropper and hailoaggregator.
    The cropper will crop detections made by earlier stages in the pipeline.
    Each detection is cropped and sent to the inner pipeline for further processing.
    The aggregator will combine the cropped detections with the original frame.
    Example use case: After face detection pipeline stage, crop the faces and send them to a face recognition pipeline.

    Args:
        inner_pipeline (str): The pipeline string to be wrapped.
        so_path (str): The path to the cropper .so library.
        function_name (str): The function name in the .so library.
        use_letterbox (bool): Whether to preserve aspect ratio. Defaults True.
        no_scaling_bbox (bool): If True, bounding boxes are not scaled. Defaults True.
        internal_offset (bool): If True, uses internal offsets. Defaults True.
        resize_method (str): The resize method. Defaults to 'inter-area'.
        bypass_max_size_buffers (int): For the bypass queue. Defaults to 20.
        name (str): A prefix name for pipeline elements. Defaults 'cropper_wrapper'.

    Returns:
        str: A pipeline string representing hailocropper + aggregator around the inner_pipeline.
    """
    return (
        f"{QUEUE(name=f'{name}_input_q')} ! "
        f"hailocropper name={name}_cropper "
        f"so-path={so_path} "
        f"function-name={function_name} "
        f"use-letterbox={str(use_letterbox).lower()} "
        f"no-scaling-bbox={str(no_scaling_bbox).lower()} "
        f"internal-offset={str(internal_offset).lower()} "
        f"resize-method={resize_method} "
        f"hailoaggregator name={name}_agg "
        # bypass
        f"{name}_cropper. ! "
        f"{QUEUE(name=f'{name}_bypass_q', max_size_buffers=bypass_max_size_buffers)} ! {name}_agg.sink_0 "
        # pipeline for the actual inference
        f"{name}_cropper. ! {inner_pipeline} ! {name}_agg.sink_1 "
        # aggregator output
        f"{name}_agg. ! {QUEUE(name=f'{name}_output_q')} "
    )

def TILE_CROPPER_PIPELINE(
    inner_pipeline,
    name='tile_cropper_wrapper',
    internal_offset=True,
    scale_level=2,
    tiling_mode=1,
    tiles_along_x_axis=4,
    tiles_along_y_axis=3,
    overlap_x_axis=0.1,
    overlap_y_axis=0.08,
    iou_threshold=0.3,
    border_threshold=0.1
):
    """
    Wraps an inner pipeline with hailotilecropper and hailotileaggregator.
    The tile cropper divides the input frame into tiles based on the specified tiling parameters.
    Each tile is processed by the inner pipeline, and the aggregator combines the results from all tiles.

    Example use case: After a detection pipeline stage, crop the frame into tiles for further processing
    (e.g., object recognition or classification) and aggregate the results.

    Args:
        inner_pipeline (str): The pipeline string to be wrapped for processing each tile.
        name (str): A prefix name for pipeline elements. Defaults to 'tile_cropper_wrapper'.
        internal_offset (bool): If True, uses internal offsets for cropping. Defaults to True.
        scale_level (int): The scaling level for the tiles. Defaults to 2.
        tiling_mode (int): The tiling mode (e.g., 1 for uniform tiling). Defaults to 1.
        tiles_along_x_axis (int): Number of tiles along the x-axis. Defaults to 4.
        tiles_along_y_axis (int): Number of tiles along the y-axis. Defaults to 3.
        overlap_x_axis (float): Overlap percentage between tiles along the x-axis. Defaults to 0.1.
        overlap_y_axis (float): Overlap percentage between tiles along the y-axis. Defaults to 0.08.
        iou_threshold (float): Intersection-over-Union (IoU) threshold for combining detections. Defaults to 0.3.
        border_threshold (float): Threshold for handling detections near tile borders. Defaults to 0.1.

    Returns:
        str: A pipeline string representing hailotilecropper + hailotileaggregator around the inner_pipeline.

    Note:
        Single scaling requires tiling_mode=0 & border_threshold=0.
    """
    return (
        f'{QUEUE(name=f"{name}_input_q")} ! '
        f'hailotilecropper name={name}_cropper '
        f'internal-offset={str(internal_offset).lower()} '
        f'tiling-mode={str(tiling_mode).lower()} '
        + (f'scale-level={str(scale_level).lower()} ' if scale_level != 0 else '') +
        f'tiles-along-x-axis={str(tiles_along_x_axis).lower()} '
        f'tiles-along-y-axis={str(tiles_along_y_axis).lower()} '
        f'overlap-x-axis={str(overlap_x_axis).lower()} '
        f'overlap-y-axis={str(overlap_y_axis).lower()} '
        f'hailotileaggregator name={name}_agg '
        f'flatten-detections=true '
        f'iou-threshold={str(iou_threshold).lower()} '
        + (f'border-threshold={str(border_threshold).lower()} ' if border_threshold != 0 else '') +
        # bypass
        f'{name}_cropper. ! '
        f'{QUEUE(name=f"{name}_bypass_q")} ! {name}_agg. '
        # pipeline for the actual inference
        f'{name}_cropper. ! {inner_pipeline} ! {name}_agg. '
        # aggregator output
        f'{name}_agg. ! {QUEUE(name=f"{name}_output_q")} '
    )

def VIDEO_STREAM_PIPELINE(port=5004, host="127.0.0.1", bitrate=2048):
    """Creates a GStreamer pipeline string portion for encoding and streaming video over UDP.

    Args:
        port (int): UDP port number.
        host (str): Destination IP address.
        bitrate (int): Target bitrate for x264enc in kbps.

    Returns:
        str: GStreamer pipeline string fragment.
    """
    # Using x264enc with zerolatency tune. Adjust encoder/params as needed.
    # Hardware encoders (e.g., omxh264enc, v4l2h264enc, vaapih264enc) are preferable on embedded systems.
    # Example using omxh264enc (Raspberry Pi):
    # encoder = f'omxh264enc target-bitrate={bitrate*1000} control-rate=variable'
    # Example using vaapih264enc (Intel):
    # encoder = f'vaapih264enc rate-control=cbr bitrate={bitrate}' # May need caps negotiation
    encoder = f"x264enc tune=zerolatency bitrate={bitrate} speed-preset=ultrafast"
    return (
        f"videoconvert ! video/x-raw,format=I420 ! "  # x264enc often prefers I420
        f"{encoder} ! video/x-h264,profile=baseline ! "  # Add profile for better compatibility potentially
        f"rtph264pay config-interval=1 pt=96 ! "
        f"udpsink host={host} port={port} sync=false async=false"
    )


def VIDEO_SHMSINK_PIPELINE(socket_path=None):
    """Creates a GStreamer pipeline string portion for shared memory video transfer using the shm plugins.
    Shmsink creates a shared memory segment and socket.

    Args:
        socket_path (str): socket path.

    Returns:
        str: GStreamer pipeline string fragment.
    """
    return f"videoconvert ! video/x-raw,format=RGB,width=640,height=480,framerate=30/1 ! shmsink socket-path={socket_path}"


def VIDEO_SHMSRC_PIPELINE(socket_path=None):
    """Creates a GStreamer pipeline string portion for shared memory video transfer using the shm plugins.
    Shmsrc connects to that segment and reads video frames.

    Args:
        socket_path (str): socket path.

    Returns:
        str: GStreamer pipeline string fragment.
    """
    return f"shmsrc socket-path={socket_path} do-timestamp=true ! video/x-raw,format=RGB,width=640,height=480,framerate=30/1 ! videoconvert ! autovideosink"


def UI_APPSINK_PIPELINE(name="ui_sink", sync="true", show_fps="false"):
    """Creates a GStreamer pipeline string for the UI appsink element.
    This pipeline is used to send video frames to a UI application.
    And convert the video format to JPEG for display.
    It includes the hailooverlay plugin to draw bounding boxes and labels on the video.

    Args:
        name (str, optional): The prefix name for the pipeline elements. Defaults to 'ui_sink'.
        sync (str, optional): The sync property for the appsink. Defaults to 'true'.

    Returns:
        str: A string representing the GStreamer pipeline for the UI appsink element.
    """
    # Construct the UI appsink pipeline string
    ui_appsink_pipeline = (
        f"{OVERLAY_PIPELINE(name=f'{name}_overlay')} ! "
        f"{QUEUE(name=f'{name}_videoconvert_q')} ! "
        # f'videoconvert name={name}_videoconvert n-threads=2 qos=false ! '
        # f'encodebin name={name}_encodebin ! '
        # f'jpegenc name={name}_jpegenc quality=100 ! '
        # f'image/jpeg ! '
        # f'{QUEUE(name=f"{name}_q")} ! '
        f"video/x-raw, format=RGB ! "
        f"appsink name={name} sync={sync} drop=true emit-signals=true "
    )
    return ui_appsink_pipeline
