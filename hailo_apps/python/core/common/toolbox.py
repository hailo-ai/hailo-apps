import json
import os
import sys
import time
import queue
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Callable, Any

import cv2
import numpy as np

try:
    from hailo_apps.python.core.common.defines import (
        DEFAULT_COCO_LABELS_PATH,
        HAILO_ARCH_KEY,
        IMAGE_EXTENSIONS,
        VIDEO_SUFFIXES,
    )
    from hailo_apps.python.core.common.hailo_logger import get_logger
    from hailo_apps.python.core.common.camera_utils import (
        CapProcessingMode,
        is_stream_url,
        open_cv_capture,
        open_rpi_camera,
        open_usb_camera,
        get_source_fps,
        select_cap_processing_mode,
    )
    from hailo_apps.python.core.common.installation_utils import is_raspberry_pi
except ImportError:
    from .defines import (
        DEFAULT_COCO_LABELS_PATH,
        HAILO_ARCH_KEY,
        IMAGE_EXTENSIONS,
        VIDEO_SUFFIXES,
    )
    from .hailo_logger import get_logger
    from .camera_utils import (
        CapProcessingMode,
        is_stream_url,
        open_cv_capture,
        open_rpi_camera,
        open_usb_camera,
        get_source_fps,
        select_cap_processing_mode,
    )
    from .installation_utils import is_raspberry_pi

hailo_logger = get_logger(__name__)


class InputType(Enum):
    USB_CAMERA = "usb_camera"
    RPI_CAMERA = "rpi_camera"
    STREAM = "stream"
    VIDEO = "video"
    IMAGES = "images"
    UNKNOWN = "unknown"


@dataclass
class InputContext:
    # User configuration
    input_src: str
    batch_size: int
    resolution: Optional[str] = None
    frame_rate: Optional[float] = None
    video_unpaced: bool = False

    # Detected input type
    input_type: InputType = InputType.UNKNOWN

    # Runtime capture objects
    cap: Optional[Any] = None
    images: Optional[List[np.ndarray]] = None

    # Runtime metadata
    source_fps: Optional[float] = None
    cap_processing_mode: Optional[CapProcessingMode] = None
    width: Optional[int] = None
    height: Optional[int] = None

    @property
    def has_capture(self) -> bool:
        return self.cap is not None

    @property
    def has_images(self) -> bool:
        return self.images is not None and len(self.images) > 0

    @property
    def is_camera(self) -> bool:
        return self.input_type in {InputType.USB_CAMERA, InputType.RPI_CAMERA}

    @property
    def is_video(self) -> bool:
        return self.input_type == InputType.VIDEO

    @property
    def is_stream(self) -> bool:
        return self.input_type == InputType.STREAM

@dataclass
class VisualizationSettings:
    output_dir: str
    save_stream_output: bool = False
    output_resolution: Optional[Tuple[int, int]] = None
    side_by_side: bool = False
    no_display: bool = False



# -------------------------------------------------------------------
# Main entry: init_input_source
# -------------------------------------------------------------------
def init_input_source(input_context: InputContext) -> InputContext:
    """
    Initialize the input source according to the user-provided input.

    Supported input values:
        - "usb"                : Open a USB/UVC camera using OpenCV
        - "rpi"                : Open Raspberry Pi camera using Picamera2
        - "0", "1", ...        : Open camera by index (Windows)
        - "/dev/videoX"        : Open a specific V4L2 device (Linux)
        - http(s):// or rtsp://: Open a network stream
        - video file path      : Open a video file
        - image file / folder  : Load images from disk

    Returns:
        InputContext: Updated input context with initialized runtime fields.
    """
    src = input_context.input_src.strip()

    # ------------------------------------------------
    # 1) USB camera
    # ------------------------------------------------
    if src == "usb" or src.startswith("/dev/video") or src.isdigit():
        input_context.input_type = InputType.USB_CAMERA
        input_context.cap = open_usb_camera(src, input_context.resolution)
        input_context.source_fps = get_source_fps(input_context.cap, "USB camera")
    # ------------------------------------------------
    # 2) Raspberry Pi camera
    # ------------------------------------------------
    elif src == "rpi":
        if not is_raspberry_pi():
            hailo_logger.error("RPi camera requested, but this is not a Raspberry Pi system.")
            sys.exit(1)

        input_context.input_type = InputType.RPI_CAMERA
        input_context.cap = open_rpi_camera()
        input_context.source_fps = 30

        if input_context.cap is None:
            sys.exit(1)

        hailo_logger.info("Using Raspberry Pi camera at 800x600, 30 FPS")

    # ------------------------------------------------
    # 3) Network stream
    # ------------------------------------------------
    elif is_stream_url(src):
        input_context.input_type = InputType.STREAM
        input_context.cap = open_cv_capture(src, "stream")
        input_context.source_fps = get_source_fps(input_context.cap, "stream camera")

    # ------------------------------------------------
    # 4) Video file
    # ------------------------------------------------
    elif any(src.lower().endswith(suffix) for suffix in VIDEO_SUFFIXES):

        if not os.path.exists(src):
            hailo_logger.error(f"File not found: {src}")
            sys.exit(1)
        input_context.input_type = InputType.VIDEO
        input_context.cap = open_cv_capture(src, "video")
        input_context.source_fps = get_source_fps(input_context.cap, "video file")

    # ------------------------------------------------
    # 5) Image directory / image file
    # ------------------------------------------------
    elif not os.path.exists(src):
        hailo_logger.error(
            f"Invalid input '{src}'. Expected one of:\n"
            "  - 'usb'\n"
            "  - 'rpi'\n"
            "  - Linux camera device (e.g., /dev/video0)\n"
            "  - Windows camera index (e.g., 0, 1)\n"
            "  - http(s):// or rtsp:// stream\n"
            "  - video file path\n"
            "  - image directory / image file"
        )
        sys.exit(1)

    else:
        input_context.input_type = InputType.IMAGES
        input_context.images = load_images_opencv(src)

        try:
            validate_images(input_context.images, input_context.batch_size)
        except ValueError as error:
            hailo_logger.error(error)
            sys.exit(1)

        hailo_logger.info(f"Using image input: {src}")

    # Runtime metadata for capture-based inputs
    if input_context.cap is not None:
        input_context.width = int(input_context.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        input_context.height = int(input_context.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        input_context.cap_processing_mode = select_cap_processing_mode(
            input_type=input_context.input_type.value,
            frame_rate=input_context.frame_rate,
            source_fps=input_context.source_fps,
            video_unpaced=input_context.video_unpaced,
        )
        hailo_logger.info(f"Capture processing mode: {input_context.cap_processing_mode.value}")


    return input_context


def load_json_file(path: str) -> Dict[str, Any]:
    """
    Loads and parses a JSON file.

    Args:
        path (str): Path to the JSON file.

    Returns:
        Dict[str, Any]: Parsed contents of the JSON file.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        OSError: If the file cannot be read.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, 'r', encoding='utf-8') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON format in file '{path}': {e.msg}", e.doc, e.pos)

    return data


def resolve_onnx_config_from_hef(hef_path: str, caller_file: str) -> str:
    """
    Resolve ONNX config JSON path based on HEF filename.

    Uses caller_file to locate the app directory.
    """
    if not hef_path:
        raise ValueError("HEF path is required to resolve ONNX config.")

    hef_name = Path(hef_path).stem
    app_dir = Path(caller_file).resolve().parent

    config_path = app_dir / "onnx" / f"config_onnx_{hef_name}.json"

    if not config_path.exists():
        raise ValueError(
            f"Missing ONNX config for model '{hef_name}'.\n"
            f"Expected: {config_path}\n"
            "Place it under the app 'onnx/' directory or pass --onnxconfig."
        )

    return str(config_path)


def load_images_opencv(images_path: str) -> List[np.ndarray]:
    """
    Load images from the specified path as RGB.

    Args:
        images_path (str): Path to the input image or directory of images.

    Returns:
        List[np.ndarray]: List of images as NumPy arrays in RGB format.
    """
    path = Path(images_path)

    def read_rgb(p: Path):
        img = cv2.imread(str(p))
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None

    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        img = read_rgb(path)
        return [img] if img is not None else []

    elif path.is_dir():
        images = [
            read_rgb(img)
            for img in path.glob("*")
            if img.suffix.lower() in IMAGE_EXTENSIONS
        ]
        return [img for img in images if img is not None]

    return []

def load_input_images(images_path: str):
    """
    Load images from the specified path.

    Args:
        images_path (str): Path to the input image or directory of images.

    Returns:
        List[Image.Image]: List of PIL.Image.Image objects.
    """
    from PIL import Image
    path = Path(images_path)
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [Image.open(path)]
    elif path.is_dir():
        return [
            Image.open(img) for img in path.glob("*")
            if img.suffix.lower() in IMAGE_EXTENSIONS
        ]
    return []

def validate_images(images: List[np.ndarray], batch_size: int) -> None:
    """
    Validate that images exist and are properly divisible by the batch size.

    Args:
        images (List[np.ndarray]): List of images.
        batch_size (int): Number of images per batch.

    Raises:
        ValueError: If images list is empty or not divisible by batch size.
    """
    if not images:
        raise ValueError(
            'No valid images found in the specified path.'
        )

    if len(images) % batch_size != 0:
        raise ValueError(
            'The number of input images should be divisible by the batch size '
            'without any remainder.'
        )


def generate_color(class_id: int) -> tuple:
    """
    Generate a unique color for a given class ID.

    Args:
        class_id (int): The class ID to generate a color for.

    Returns:
        tuple: A tuple representing an RGB color.
    """
    np.random.seed(class_id)
    return tuple(np.random.randint(0, 255, size=3).tolist())

def get_labels(labels_path: str) -> list:
        """
        Load labels from a file.

        Args:
            labels_path (str): Path to the labels file.

        Returns:
            list: List of class names.
        """
        if labels_path is None or not os.path.exists(labels_path):
            labels_path = DEFAULT_COCO_LABELS_PATH
        with open(labels_path, 'r', encoding="utf-8") as f:
            class_names = f.read().splitlines()
        return class_names


def id_to_color(idx):
    np.random.seed(idx)
    return np.random.randint(0, 255, size=3, dtype=np.uint8)


def stop_after_timeout(stop_event: threading.Event, timeout_seconds: int) -> None:
    """
    Stop the pipeline after the requested timeout.

    Args:
        stop_event (threading.Event): Global stop signal for the pipeline.
        timeout_seconds (int): Number of seconds to wait before stopping.
    """
    time.sleep(timeout_seconds)
    if not stop_event.is_set():
        hailo_logger.info(f"Reached time limit ({timeout_seconds}s). Stopping application.")
        stop_event.set()

####################################################################
# PreProcess of Network Input
####################################################################
def preprocess(
    input_context: InputContext,
    input_queue: queue.Queue,
    model_input_width: int,
    model_input_height: int,
    preprocess_fn: Optional[Callable[[np.ndarray, int, int], np.ndarray]] = None,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """
    Preprocess and enqueue images or captured frames into the input queue.
    """
    preprocess_fn = preprocess_fn or default_preprocess

    if input_context.has_images:
        preprocess_images(
            images=input_context.images,
            batch_size=input_context.batch_size,
            input_queue=input_queue,
            model_input_width=model_input_width,
            model_input_height=model_input_height,
            preprocess_fn=preprocess_fn,
        )
    else:
        preprocess_from_capture(
            cap=input_context.cap,
            batch_size=input_context.batch_size,
            input_queue=input_queue,
            model_input_width=model_input_width,
            model_input_height=model_input_height,
            processing_mode=input_context.cap_processing_mode,
            preprocess_fn=preprocess_fn,
            target_fps=input_context.frame_rate,
            stop_event=stop_event,
        )

    input_queue.put(None)


def preprocess_from_capture(
    cap: Any,
    batch_size: int,
    input_queue: queue.Queue,
    model_input_width: int,
    model_input_height: int,
    processing_mode: Optional[CapProcessingMode],
    preprocess_fn: Callable[[np.ndarray, int, int], np.ndarray],
    target_fps: Optional[float] = None,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """
    Read frames from capture, preprocess them, and push batches to the queue.

    Supported modes:
    - CAMERA_NORMAL:
        Process camera frames as they arrive.
    - CAMERA_FRAME_DROP:
        Drop camera frames to match the requested target FPS.
    - VIDEO_PACE:
        Process video with normal playback pacing based on source timestamps.
    - VIDEO_UNPACED:
        Process video as fast as possible.
    - VIDEO_PACED_AND_FRAME_DROP:
        Pace video playback normally, but only keep frames that match the
        requested target FPS.
    """

    def should_stop() -> bool:
        return stop_event is not None and stop_event.is_set()

    if processing_mode in (
        CapProcessingMode.CAMERA_FRAME_DROP,
        CapProcessingMode.VIDEO_PACED_AND_FRAME_DROP,
    ):
        if not target_fps or target_fps <= 0:
            raise ValueError(
                f"{processing_mode.value} requires a positive target_fps"
            )

    # Camera frame-drop timing
    next_keep_timestamp = time.monotonic()
    keep_period = (
        1.0 / float(target_fps)
        if processing_mode == CapProcessingMode.CAMERA_FRAME_DROP
        else None
    )

    # Video pacing state
    video_start_ms: Optional[float] = None
    wall_start_time: Optional[float] = None

    # Video frame-drop timing (based on source video timestamps)
    next_keep_video_ms: Optional[float] = None
    video_keep_period_ms = (
        1000.0 / float(target_fps)
        if processing_mode == CapProcessingMode.VIDEO_PACED_AND_FRAME_DROP
        else None
    )

    raw_frames: List[np.ndarray] = []
    processed_frames: List[np.ndarray] = []

    frame_index = 0

    while not should_stop():
        ret, frame_bgr = cap.read()
        if not ret:
            hailo_logger.debug("[READ] End of stream")
            break

        frame_index += 1
        hailo_logger.debug(f"[READ] frame={frame_index}")

        current_pos_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)

        # Initialize video timing state on first frame for paced modes
        if processing_mode in (
            CapProcessingMode.VIDEO_PACE,
            CapProcessingMode.VIDEO_PACED_AND_FRAME_DROP,
        ):
            if video_start_ms is None:
                video_start_ms = current_pos_ms
                wall_start_time = time.monotonic()

        # Video paced + frame drop:
        # keep only frames that match the requested target FPS based on video timestamps
        if processing_mode == CapProcessingMode.VIDEO_PACED_AND_FRAME_DROP:
            if next_keep_video_ms is None:
                next_keep_video_ms = current_pos_ms

            if current_pos_ms + 1e-3 < next_keep_video_ms:
                hailo_logger.debug(
                    f"[DROP] frame={frame_index} "
                    f"mode={processing_mode.value} "
                    f"pos_ms={current_pos_ms:.1f} "
                    f"next_keep_ms={next_keep_video_ms:.1f}"
                )
                continue

            hailo_logger.debug(
                f"[KEEP] frame={frame_index} "
                f"mode={processing_mode.value} "
                f"pos_ms={current_pos_ms:.1f}"
            )

            while current_pos_ms + 1e-3 >= next_keep_video_ms:
                next_keep_video_ms += video_keep_period_ms

        # Paced video playback
        if processing_mode in (
            CapProcessingMode.VIDEO_PACE,
            CapProcessingMode.VIDEO_PACED_AND_FRAME_DROP,
        ):
            desired_wall_time = wall_start_time + (current_pos_ms - video_start_ms) / 1000.0
            current_wall_time = time.monotonic()

            if current_wall_time < desired_wall_time:
                sleep_seconds = desired_wall_time - current_wall_time
                hailo_logger.debug(
                    f"[PACE] frame={frame_index} "
                    f"sleep={sleep_seconds:.4f}s "
                    f"pos_ms={current_pos_ms:.1f}"
                )
                time.sleep(sleep_seconds)

        # Camera frame drop mode
        if processing_mode == CapProcessingMode.CAMERA_FRAME_DROP:
            current_time = time.monotonic()

            if current_time < next_keep_timestamp:
                hailo_logger.debug(f"[DROP] frame={frame_index} mode={processing_mode.value}")
                continue

            hailo_logger.debug(f"[KEEP] frame={frame_index} mode={processing_mode.value}")
            next_keep_timestamp += keep_period

        # Keep log for all other modes
        if processing_mode not in (
            CapProcessingMode.CAMERA_FRAME_DROP,
            CapProcessingMode.VIDEO_PACED_AND_FRAME_DROP,
        ):
            hailo_logger.debug(f"[KEEP] frame={frame_index} mode={processing_mode.value}")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        raw_frames.append(frame_rgb)
        processed_frames.append(
            preprocess_fn(frame_rgb, model_input_width, model_input_height)
        )

        if len(raw_frames) >= batch_size:
            hailo_logger.debug(f"[QUEUE] push batch size={len(raw_frames)}")
            input_queue.put((raw_frames, processed_frames))
            raw_frames, processed_frames = [], []

    if raw_frames and not should_stop():
        input_queue.put((raw_frames, processed_frames))


def preprocess_images(
    images: List[np.ndarray],
    batch_size: int,
    input_queue: queue.Queue,
    model_input_width: int,
    model_input_height: int,
    preprocess_fn: Callable[[np.ndarray, int, int], np.ndarray],
) -> None:
    """
    Process a list of images and enqueue them in batches.
    """
    for batch in divide_list_to_batches(images, batch_size):
        batch_tuple = (
            [image for image in batch],
            [preprocess_fn(image, model_input_width, model_input_height) for image in batch],
        )
        input_queue.put(batch_tuple)


def divide_list_to_batches(
        images_list: List[np.ndarray], batch_size: int
) -> Generator[List[np.ndarray], None, None]:
    """
    Divide the list of images into batches.

    Args:
        images_list (List[np.ndarray]): List of images.
        batch_size (int): Number of images in each batch.

    Returns:
        Generator[List[np.ndarray], None, None]: Generator yielding batches
                                                  of images.
    """
    for i in range(0, len(images_list), batch_size):
        yield images_list[i: i + batch_size]


def default_preprocess(image: np.ndarray, model_w: int, model_h: int) -> np.ndarray:
    """
    Resize image with unchanged aspect ratio using padding.

    Args:
        image (np.ndarray): Input image.
        model_w (int): Model input width.
        model_h (int): Model input height.

    Returns:
        np.ndarray: Preprocessed and padded image.
    """
    img_h, img_w, _ = image.shape[:3]
    scale = min(model_w / img_w, model_h / img_h)
    new_img_w, new_img_h = int(img_w * scale), int(img_h * scale)
    image = cv2.resize(image, (new_img_w, new_img_h), interpolation=cv2.INTER_CUBIC)

    padded_image = np.full((model_h, model_w, 3), (114, 114, 114), dtype=np.uint8)
    x_offset = (model_w - new_img_w) // 2
    y_offset = (model_h - new_img_h) // 2
    padded_image[y_offset:y_offset + new_img_h, x_offset:x_offset + new_img_w] = image

    return padded_image


####################################################################
# Visualization
####################################################################
def visualize(
    input_context: InputContext,
    visualization_settings: VisualizationSettings,
    output_queue: queue.Queue,
    callback: Callable[[Any, Any], None],
    fps_tracker: Optional["FrameRateTracker"] = None,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """
    Visualize inference results.

    Responsibilities:
        • Receive frames and inference outputs from the output_queue
        • Apply the visualization callback
        • Display frames in an OpenCV window (unless --no-display)
        • Optionally save output video
        • Support both capture-based pipelines and image-based pipelines
    """

    image_index = 0
    video_writer = None
    writer_frame_width = None
    writer_frame_height = None

    cap = input_context.cap

    # ------------------------------------------------------------
    # Initialize display window and video writer (if needed)
    # ------------------------------------------------------------
    if cap is not None:

        # Create display window only when display is enabled
        if not visualization_settings.no_display:
            cv2.namedWindow("Output", cv2.WINDOW_AUTOSIZE)

        # Determine output resolution
        if visualization_settings.output_resolution is not None:
            target_width, target_height = visualization_settings.output_resolution
        else:
            target_width, target_height = input_context.width, input_context.height

        # Side-by-side comparison doubles the width
        writer_frame_width = target_width * (2 if visualization_settings.side_by_side else 1)
        writer_frame_height = target_height

        # Initialize VideoWriter if saving output is enabled
        if visualization_settings.save_stream_output:

            camera_fps = input_context.source_fps

            # Choose output FPS
            output_fps = input_context.frame_rate or (
                camera_fps if camera_fps and camera_fps > 1 else 30.0
            )

            os.makedirs(visualization_settings.output_dir, exist_ok=True)

            output_video_path = os.path.join(
                visualization_settings.output_dir,
                "output.avi",
            )

            video_writer = cv2.VideoWriter(
                output_video_path,
                cv2.VideoWriter_fourcc(*"XVID"),
                output_fps,
                (writer_frame_width, writer_frame_height),
            )

    # ------------------------------------------------------------
    # Main visualization loop
    # ------------------------------------------------------------
    while True:

        result = output_queue.get()

        try:
            # Sentinel value indicates pipeline termination
            if result is None:
                break

            # Result format:
            #   (frame, inference_result)
            #   (frame, inference_result, metadata)
            original_frame, inference_result, *metadata = result

            # If a stop event was triggered, keep draining queue
            if stop_event is not None and stop_event.is_set():
                continue

            # Normalize inference output
            if isinstance(inference_result, list) and len(inference_result) == 1:
                inference_result = inference_result[0]

            # Run visualization callback
            if metadata:
                frame_with_detections = callback(
                    original_frame,
                    inference_result,
                    metadata[0],
                )
            else:
                frame_with_detections = callback(original_frame, inference_result)

            # Update FPS tracker
            if fps_tracker is not None:
                fps_tracker.increment()

            # Convert RGB → BGR for OpenCV display
            output_bgr_frame = cv2.cvtColor(frame_with_detections, cv2.COLOR_RGB2BGR)

            frame_to_show = resize_frame_for_output(
                output_bgr_frame,
                visualization_settings.output_resolution,
            )

            # ----------------------------------------------------
            # Capture-based pipelines (camera / video / stream)
            # ----------------------------------------------------
            if cap is not None:

                # Display output window
                if not visualization_settings.no_display:

                    cv2.imshow("Output", frame_to_show)

                    # Allow quitting with 'q'
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        if stop_event is not None:
                            stop_event.set()
                        continue

                # Save video output if enabled
                if (
                    visualization_settings.save_stream_output
                    and video_writer is not None
                    and writer_frame_width
                    and writer_frame_height
                ):
                    frame_to_write = cv2.resize(
                        frame_to_show,
                        (writer_frame_width, writer_frame_height),
                    )

                    video_writer.write(frame_to_write)

            # ----------------------------------------------------
            # Image pipelines
            # ----------------------------------------------------
            else:

                os.makedirs(visualization_settings.output_dir, exist_ok=True)

                output_image_path = os.path.join(
                    visualization_settings.output_dir,
                    f"output_{image_index}.png",
                )

                cv2.imwrite(output_image_path, frame_to_show)

            image_index += 1

        finally:
            output_queue.task_done()

    # ------------------------------------------------------------
    # Cleanup resources
    # ------------------------------------------------------------
    if video_writer is not None:
        video_writer.release()

    if cap is not None:
        cap.release()

    if not visualization_settings.no_display:
        cv2.destroyAllWindows()

def resize_frame_for_output(frame: np.ndarray,
                            resolution: Optional[Tuple[int, int]]) -> np.ndarray:
    """
    Resize a frame according to the selected output resolution while
    preserving aspect ratio. Only the target height is enforced.

    Args:
        frame (np.ndarray): Input RGB or BGR image.
        resolution (Optional[Tuple[int, int]]): (width, height) or None.

    Returns:
        np.ndarray: Resized frame, or the original frame if resolution is None.
    """
    if resolution is None:
        return frame

    _, target_h = resolution

    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return frame

    scale = target_h / float(h)
    new_w = int(round(w * scale))
    new_h = target_h

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized


####################################################################
# Frame Rate Tracker
####################################################################
class FrameRateTracker:
    """
    Tracks frame count and elapsed time to compute real-time FPS (frames per second).
    """

    def __init__(self):
        """Initialize the tracker with zero frames and no start time."""
        self._count = 0
        self._start_time = None

    def start(self) -> None:
        """Start or restart timing and reset the frame count."""
        self._start_time = time.time()

    def increment(self, n: int = 1) -> None:
        """Increment the frame count.

        Args:
            n (int): Number of frames to add. Defaults to 1.
        """
        self._count += n


    @property
    def count(self) -> int:
        """Returns:
            int: Total number of frames processed.
        """
        return self._count

    @property
    def elapsed(self) -> float:
        """Returns:
            float: Elapsed time in seconds since `start()` was called.
        """
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def fps(self) -> float:
        """Returns:
            float: Calculated frames per second (FPS).
        """
        elapsed = self.elapsed
        return self._count / elapsed if elapsed > 0 else 0.0

    def frame_rate_summary(self) -> str:
        """Return a summary of frame count and FPS.

        Returns:
            str: e.g. "Processed 200 frames at 29.81 FPS"
        """
        return f"Processed {self.count} frames at {self.fps:.2f} FPS, Total time: {self.elapsed:.2f} seconds"

    


####################################################################
# Resource Resolution Functions (for HACE compatibility)
####################################################################
def resolve_arch(arch: Optional[str]) -> str:
    """
    Resolve the target Hailo architecture using CLI, environment, or auto-detection.

    Order:
      1. Explicit --arch value
      2. Environment variable HAILO_ARCH_KEY (used by pipelines)
      3. Automatic detection via detect_hailo_arch()

    Exits with an error message if none of the above succeed.
    """
    if arch:
        return arch

    env_arch = os.getenv(HAILO_ARCH_KEY)
    if env_arch:
        return env_arch

    try:
        from hailo_apps.python.core.common.installation_utils import detect_hailo_arch

        detected_arch = detect_hailo_arch()
    except Exception as exc:  # pragma: no cover - defensive
        hailo_logger.debug(f"Failed to auto-detect Hailo architecture: {exc}")
        detected_arch = None

    if detected_arch:
        return detected_arch

    hailo_logger.error(
        "Could not determine the Hailo device architecture: no --arch value or "
        "'hailo_arch' environment variable was provided, and automatic detection failed. "
        "Specify the architecture matching your device with --arch hailo8, "
        "--arch hailo8l, or --arch hailo10h, or set the 'hailo_arch' environment variable. "
        "To check whether HailoRT can identify your device, run 'hailortcli fw-control identify'."
    )
    sys.exit(1)
