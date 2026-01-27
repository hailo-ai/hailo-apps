import json
import os
import sys
import time
import queue
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Callable, Any
import subprocess
import cv2
import numpy as np
import re
import threading

try:
    from hailo_apps.python.core.common.defines import (
        HAILO_ARCH_KEY,
        DEFAULT_COCO_LABELS_PATH,
        IMAGE_EXTENSIONS,
        CAMERA_RESOLUTION_MAP,
        VIDEO_SUFFIXES
    )
    from hailo_apps.python.core.common.hailo_logger import get_logger
    from hailo_apps.python.core.common.installation_utils import is_raspberry_pi
except ImportError:
    from .defines import (
        HAILO_ARCH_KEY,
        DEFAULT_COCO_LABELS_PATH,
        IMAGE_EXTENSIONS,
        CAMERA_RESOLUTION_MAP,
        VIDEO_SUFFIXES
    )
    from .hailo_logger import get_logger
    from .camera_utils import is_rpi_camera_available
    from .installation_utils import is_raspberry_pi

logger = get_logger(__name__)



class PiCamera2CaptureAdapter:
    """
    Adapter that makes Picamera2 behave like cv2.VideoCapture.

    Goals:
    - Provide read(), isOpened(), get(), release() APIs compatible with OpenCV code
    - Avoid deadlocks when release() is called while another thread is reading
    - Ensure stop()/close() never race with capture_array()
    """

    def __init__(self, picam2):
        self.picam2 = picam2
        self._opened = True
        self._io_lock = threading.Lock()

    def isOpened(self):
        return self._opened

    def read(self):
        if not self._opened:
            return False, None

        # prevent stop/close while capturing
        with self._io_lock:
            if not self._opened: # re-check after taking lock
                return False, None
            frame = self.picam2.capture_array()

        if frame is None:
            return False, None
        return True, frame

    def get(self, prop_id: int) -> float:
        if prop_id in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT):
            try:
                cfg = self.picam2.camera_configuration()
                size = cfg.get("main", {}).get("size", None)
                if size and len(size) == 2:
                    w, h = int(size[0]), int(size[1])
                    return float(w if prop_id == cv2.CAP_PROP_FRAME_WIDTH else h)
            except Exception:
                pass
            return 0.0
        if prop_id == cv2.CAP_PROP_FPS:
            return 30.0
        return None

    def release(self):
        # stop new reads ASAP
        self._opened = False

        # wait if a read() is currently inside capture_array()
        with self._io_lock:
            try:
                self.picam2.stop()
            except Exception:
                pass
            try:
                self.picam2.close()
            except Exception:
                pass


def get_usb_video_devices() -> dict[int, str]:
    """
    Return {video_index: device_header} for USB-backed V4L2 devices only.

    Works with v4l2-ctl output styles like:
      - "Camera Name (046d:0825):"
      - "Camera Name (usb-xhci-hcd.1-1):"
    """
    try:
        out = subprocess.check_output(
            ["v4l2-ctl", "--list-devices"],
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception as e:
        logger.error(f"Failed to run v4l2-ctl --list-devices: {e}")
        return {}

    usb_devices: dict[int, str] = {}
    current_header: str = ""
    is_usb_section = False

    for line in out.splitlines():
        if not line.strip():
            continue

        # Header lines are not tab-indented
        if not line.startswith("\t"):
            current_header = line.strip().rstrip(":")
            lower = current_header.lower()

            # USB detection: either VID:PID OR "(usb-...)" style
            has_vid_pid = bool(re.search(r"\([0-9a-f]{4}:[0-9a-f]{4}\)", current_header, re.I))
            has_usb_bus = ("(usb-" in lower) or (" usb-" in lower) or ("(usb:" in lower)

            is_usb_section = has_vid_pid or has_usb_bus
            continue

        # Device node lines (tab-indented)
        if is_usb_section and "/dev/video" in line:
            # line looks like "\t/dev/video8"
            m = re.search(r"/dev/video(\d+)", line)
            if m:
                idx = int(m.group(1))
                usb_devices[idx] = current_header

    return usb_devices


def open_usb_camera(resolution: Optional[str]):
    """
    USB camera open .

    Behavior:
      - Detect REAL USB cameras via v4l2-ctl
      - If CAMERA_INDEX env var exists -> use it
      - Else -> auto-pick FIRST USB camera
      - Ignore CSI/RPi cameras completely
      - Apply resolution if requested
      - Ensure camera actually streams frames
    """
    usb_devices = get_usb_video_devices()
    if not usb_devices:
        logger.error("USB mode requested, but NO USB cameras detected.")
        logger.error("Run: v4l2-ctl --list-devices")
        sys.exit(1)

    # --------------------------------------------
    # Select camera index (env override OR auto)
    # --------------------------------------------
    env_val = os.environ.get("CAMERA_INDEX")
    if env_val is None:
        camera_index = sorted(usb_devices.keys())[0]
        logger.debug(
                f"No CAMERA_INDEX provided. "
                f"Auto-selected USB camera index {camera_index} "
                f"({usb_devices[camera_index]})"
            )
    else:
        try:
            camera_index = int(env_val)
        except ValueError:
            logger.error(f"Invalid CAMERA_INDEX value: {env_val}")
            sys.exit(1)

        if camera_index not in usb_devices:
            logger.error(
                f"CAMERA_INDEX={camera_index} is NOT a USB camera.\n"
                f"Available USB camera indices: {sorted(usb_devices.keys())}"
            )
            sys.exit(1)

    # --------------------------------------------
    # Open camera
    # --------------------------------------------
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error(f"Failed to open USB camera index {camera_index}")
        sys.exit(1)

    # --------------------------------------------
    # Apply resolution (USB only)
    # --------------------------------------------
    if resolution in CAMERA_RESOLUTION_MAP:
        w, h = CAMERA_RESOLUTION_MAP[resolution]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        logger.debug(f"USB camera resolution forced to {w}x{h}")

    # --------------------------------------------
    # Validate stream (real camera test)
    # --------------------------------------------
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        logger.error("USB camera opened but produced no frames.")
        sys.exit(1)

    return cap


def open_rpi_camera():
    try:
        from picamera2 import Picamera2
    except Exception as e:
        logger.error(f"Picamera2 not available: {e}")
        return None

    try:
        picam2 = Picamera2()
        main = {"size": (800, 600), "format": "RGB888"}
        config = picam2.create_video_configuration(main=main, controls={"FrameRate": 30})

        picam2.configure(config)
        picam2.start()
        return PiCamera2CaptureAdapter(picam2)

    except Exception as e:
        logger.error(f"Failed to open RPi camera: {e}")
        try:
            picam2.stop()
        except Exception:
            pass
        try:
            picam2.close()
        except Exception:
            pass
        return None


def is_stream_url(input_arg: str) -> bool:
    return input_arg.startswith(("http://", "https://", "rtsp://"))


# -------------------------------------------------------------------
# Main entry: init_input_source
# -------------------------------------------------------------------
def init_input_source(input_src: str, batch_size: int, resolution: Optional[str]):
    """
    Initialize input source based on user-provided `input`.

    Supported values:
      - "usb" : Open a USB/UVC camera using OpenCV (cv2.VideoCapture).
              `resolution` applies here (sd/hd/fhd) or native if None.
      - "rpi" : Open Raspberry Pi camera using Picamera2 (fixed 1280x720).
              `resolution` is ignored by design.
      - http(s):// or rtsp://: Network stream.
      - Video file path (.mp4/.avi/.mov/.mkv) : Open as cv2.VideoCapture(file).
      - Directory path : Load images from the directory.

    Returns:
        (cap, images)
          cap: cv2.VideoCapture OR PiCamera2CaptureAdapter OR None
          images: List[np.ndarray] for image-dir mode, else None
    """
    src = input_src.strip()

    # ------------------------------------------------
    # 1) USB camera
    # ------------------------------------------------
    if src == "usb":
        cap = open_usb_camera(resolution)
        logger.info("Using USB camera")
        return cap, None

    # ------------------------------------------------
    # 2) Raspberry Pi camera
    # ------------------------------------------------
    if src == "rpi":
        if not is_raspberry_pi():
            logger.error("RPi camera requested, but this is not a Raspberry Pi system.")
            sys.exit(1)
        cap = open_rpi_camera()
        if cap is None:
            sys.exit(1)

        logger.info("Using Raspberry Pi camera at 800x600")
        return cap, None

    # ------------------------------------------------
    # 3) Network stream (RTSP / HTTP / HTTPS)
    # ------------------------------------------------
    if is_stream_url(src):
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            logger.error(f"Failed to open stream URL: {src}")
            sys.exit(1)

        logger.info(f"Using stream input: {src}")
        return cap, None

    # ------------------------------------------------
    # 4) Video file
    # ------------------------------------------------
    if any(src.endswith(suffix) for suffix in VIDEO_SUFFIXES):
        if not os.path.exists(input_src):
            logger.error(f"File not found: {input_src}")
            sys.exit(1)

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {src}")
            sys.exit(1)

        logger.info(f"Using video file input: {src}")
        return cap, None

    # ------------------------------------------------
    # 5) Image directory / Image file
    # ------------------------------------------------
    if not os.path.exists(src):
        logger.error(
            f"Invalid input '{src}'. Expected one of:\n"
            "  - 'usb'\n"
            "  - 'rpi'\n"
            "  - http(s):// or rtsp:// stream\n"
            "  - video file path\n"
            "  - image directory / image file"
        )
        sys.exit(1)

    images = load_images_opencv(src)
    try:
        validate_images(images, batch_size)
    except ValueError as e:
        logger.error(e)
        sys.exit(1)

    return None, images


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



####################################################################
# PreProcess of Network Input
####################################################################

def preprocess(images: List[np.ndarray], cap: cv2.VideoCapture, framerate: float, batch_size: int,
               input_queue: queue.Queue, width: int, height: int,
               preprocess_fn: Optional[Callable[[np.ndarray, int, int], np.ndarray]] = None) -> None:

    """
    Preprocess and enqueue images or camera frames into the input queue as they are ready.
    Args:
        images (List[np.ndarray], optional): List of images as NumPy arrays.
        cap (cv2.VideoCapture, optional): VideoCapture object for camera/video stream.
        framerate (float, optional): Target framerate for frame skipping. If provided, frames
                                     will be skipped to achieve approximately this FPS.
        batch_size (int): Number of images per batch.
        input_queue (queue.Queue): Queue for input images.
        width (int): Model input width.
        height (int): Model input height.
        preprocess_fn (Callable, optional): Custom preprocessing function that takes an image, width, and height,
                                            and returns the preprocessed image. If not provided, a default padding-based
                                            preprocessing function will be used.
    """
    preprocess_fn = preprocess_fn or default_preprocess

    if cap is None:
        preprocess_images(images, batch_size, input_queue, width, height, preprocess_fn)
    else:
        preprocess_from_cap(cap, batch_size, input_queue, width, height, preprocess_fn, framerate)

    input_queue.put(None)  #Add sentinel value to signal end of input


def preprocess_from_cap(cap: Any,
                        batch_size: int,
                        input_queue: queue.Queue,
                        width: int,
                        height: int,
                        preprocess_fn: Callable[[np.ndarray, int, int], np.ndarray],
                        framerate: Optional[float] = None) -> None:
    """
    Read frames from a capture source, optionally limit how often frames are
    allowed into the pipeline, preprocess them, and enqueue them in batches.

    Args:
        cap: VideoCapture object.
        batch_size (int): Number of images per batch.
        input_queue (queue.Queue): Queue for input images.
        width (int): Model input width.
        height (int): Model input height.
        preprocess_fn (Callable): Function to preprocess a single image (image, width, height) -> image.
        framerate (float, optional): Target framerate for frame skipping.
    """
    frames = []
    processed_frames = []

    # Estimate camera FPS
    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    if not cam_fps or cam_fps <= 0:
        cam_fps = 30.0  # sensible default

    # Decide how many frames to skip
    if framerate is not None and framerate > 0:
        # e.g. cam_fps=30, framerate=1  -> skip=30  (use every 30th frame)
        #      cam_fps=30, framerate=10 -> skip=3   (use every 3rd frame)
        skip = max(1, int(round(cam_fps / float(framerate))))
    else:
        skip = 1  # no frame skipping, use all frames
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Skip frames to achieve the desired effective FPS
        if frame_idx % skip != 0:
            continue

        # Process only the kept frames - convert to RGB and store
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        processed_frame = preprocess_fn(frame, width, height)
        processed_frames.append(processed_frame)

        if len(frames) == batch_size:
            input_queue.put((frames, processed_frames))
            processed_frames, frames = [], []


def preprocess_images(images: List[np.ndarray], batch_size: int, input_queue: queue.Queue, width: int, height: int,
                      preprocess_fn: Callable[[np.ndarray, int, int], np.ndarray]) -> None:
    """
    Process a list of images and enqueue them.
    Args:
        images (List[np.ndarray]): List of images as NumPy arrays.
        batch_size (int): Number of images per batch.
        input_queue (queue.Queue): Queue for input images.
        width (int): Model input width.
        height (int): Model input height.
        preprocess_fn (Callable): Function to preprocess a single image (image, width, height) -> image.
    """
    for batch in divide_list_to_batches(images, batch_size):
        input_tuple = ([image for image in batch], [preprocess_fn(image, width, height) for image in batch])

        input_queue.put(input_tuple)




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


def visualize(
    output_queue: queue.Queue,
    cap: Optional[Any],
    save_stream_output: bool,
    output_dir: str,
    callback: Callable[[Any, Any], None],
    fps_tracker: Optional["FrameRateTracker"] = None,
    output_resolution: Optional[Tuple[int, int]] = None,
    framerate: Optional[float] = None,
    side_by_side: bool = False
) -> None:
    """
    Visualize inference results: draw detections, show them on screen,
    and optionally save the output video.

    Args:
        output_queue: Queue with (frame, inference_result[, extra]).
        cap: VideoCapture for camera/video input, or None for image mode.
        save_stream_output: If True, write the visualization to a video file.
        output_dir: Directory to save output frames or videos.
        callback: Function that draws detections on the frame.
        fps_tracker: Tracks real-time FPS (optional).
        output_resolution: One of ['sd','hd','fhd'] or a custom resolution for final display/save size.
        framerate: Override output video FPS (optional).
        side_by_side: If True, the callback returns a wide comparison frame.
    """
    image_id = 0
    out = None
    frame_width = None
    frame_height = None

    # Window + writer init (only for camera/video, not images)
    if cap is not None:
        cv2.namedWindow("Output", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        base_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        base_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

        if output_resolution is not None:
            target_w, target_h = output_resolution
        else:
            target_w, target_h = base_width, base_height

        frame_width  = target_w * (2 if side_by_side else 1)
        frame_height = target_h

        if save_stream_output:
            cam_fps   = cap.get(cv2.CAP_PROP_FPS)
            final_fps = framerate or (cam_fps if cam_fps and cam_fps > 1 else 30.0)

            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, "output.avi")
            out = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*"XVID"),
                final_fps,
                (frame_width, frame_height),
            )

    # Main loop
    while True:
        result = output_queue.get()
        if result is None:
            output_queue.task_done()
            break

        original_frame, inference_result, *rest = result

        if isinstance(inference_result, list) and len(inference_result) == 1:
            inference_result = inference_result[0]

        if rest:
            frame_with_detections = callback(original_frame, inference_result, rest[0])
        else:
            frame_with_detections = callback(original_frame, inference_result)

        if fps_tracker is not None:
            fps_tracker.increment()

        # Convert RGB to BGR for OpenCV display/save
        bgr_frame = cv2.cvtColor(frame_with_detections, cv2.COLOR_RGB2BGR)
        frame_to_show = resize_frame_for_output(bgr_frame, output_resolution)

        if cap is not None:
            cv2.imshow("Output", frame_to_show)
            if save_stream_output and out is not None and frame_width and frame_height:
                frame_to_save = cv2.resize(frame_to_show, (frame_width, frame_height))
                out.write(frame_to_save)
        else:
            cv2.imwrite(os.path.join(output_dir, f"output_{image_id}.png"), frame_to_show)

        image_id += 1
        output_queue.task_done()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            if save_stream_output and out is not None:
                out.release()
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            break

    if cap is not None and save_stream_output and out is not None:
        out.release()




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
        return f"Processed {self.count} frames at {self.fps:.2f} FPS"
    



####################################################################
# Resource Resolution Functions (for HACE compatibility)
####################################################################
def resolve_arch(arch: str | None) -> str:
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
        logger.debug(f"Failed to auto-detect Hailo architecture: {exc}")
        detected_arch = None

    if detected_arch:
        return detected_arch

    logger.error(
        "Could not determine Hailo architecture. "
        "Please specify --arch or set the environment variable 'hailo_arch'."
    )
    sys.exit(1)
