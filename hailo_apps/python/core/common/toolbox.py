import json
import os
import sys
import time
import queue
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Callable, Any

import cv2
import numpy as np

try:
    from hailo_apps.python.core.common.defines import (
        HAILO_ARCH_KEY,
        RESOURCES_PHOTOS_DIR_NAME,
        RESOURCES_VIDEOS_DIR_NAME,
        DEFAULT_COCO_LABELS_PATH
    )
    from hailo_apps.python.core.common.core import get_resource_path
    from hailo_apps.python.core.common.hailo_logger import get_logger
except ImportError:
    from .defines import (
        HAILO_ARCH_KEY,
        RESOURCES_PHOTOS_DIR_NAME,
        RESOURCES_VIDEOS_DIR_NAME,
        DEFAULT_COCO_LABELS_PATH
    )
    from .core import get_resource_path
    from .hailo_logger import get_logger

logger = get_logger(__name__)

IMAGE_EXTENSIONS: Tuple[str, ...] = ('.jpg', '.png', '.bmp', '.jpeg')
CAMERA_RESOLUTION_MAP = {
    "sd": (640, 480),
    "hd": (1280, 720),
    "fhd": (1920, 1080)
}
CAMERA_INDEX = int(os.environ.get('CAMERA_INDEX', '0'))


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


def is_valid_camera_index(index):
    """
    Check if a camera index is available and can be opened.

    Args:
        index (int): Camera index to test.

    Returns:
        bool: True if the camera can be opened, else False.
    """
    cap = cv2.VideoCapture(index)
    valid = cap.isOpened()
    cap.release()
    return valid


def list_available_cameras(max_index=5):
    """
    List all available camera indices up to a maximum index.

    Args:
        max_index (int): Highest camera index to test.

    Returns:
        list[int]: List of available camera indices.
    """
    available = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def init_input_source(input_path, batch_size, resolution):
    """
    Initialize input source from camera, video file, or image directory.

    Args:
        input_path (str): "camera", video file path, or image directory.
        batch_size (int): Number of images to validate against.
        resolution (str or None): One of ['sd', 'hd', 'fhd'], or None to use native camera resolution.

    Returns:
        Tuple[Optional[cv2.VideoCapture], Optional[List[np.ndarray]]]
    """
    cap = None
    images = None

    def get_camera_native_resolution():
        cap = cv2.VideoCapture(CAMERA_INDEX)
        res = (int(cap.get(3)), int(cap.get(4))) if cap.isOpened() else (640, 480)
        cap.release()
        return res


    if input_path == "camera":

        if not is_valid_camera_index(CAMERA_INDEX):
            logger.error(f"CAMERA_INDEX {CAMERA_INDEX} not found.")
            available = list_available_cameras()
            logger.warning(f"Available camera indices: {available}")
            exit(1)

        if not resolution:
            CAMERA_CAP_WIDTH, CAMERA_CAP_HEIGHT = get_camera_native_resolution()
        else:
            CAMERA_CAP_WIDTH, CAMERA_CAP_HEIGHT = CAMERA_RESOLUTION_MAP.get(resolution, (640, 480))  # fallback to SD

        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CAP_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CAP_HEIGHT)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))

    elif any(input_path.lower().endswith(suffix) for suffix in ['.mp4', '.avi', '.mov', '.mkv']):
        if not os.path.exists(input_path):
            logger.error(f"File not found: {input_path}")
            sys.exit(1)
        cap = cv2.VideoCapture(input_path)
    else:
        images = load_images_opencv(input_path)
        try:
            validate_images(images, batch_size)
        except ValueError as e:
            logger.error(e)
            sys.exit(1)

    return cap, images


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


def preprocess_from_cap(cap: cv2.VideoCapture,
                        batch_size: int,
                        input_queue: queue.Queue,
                        width: int,
                        height: int,
                        preprocess_fn: Callable[[np.ndarray, int, int], np.ndarray],
                        framerate: Optional[float] = None) -> None:
    """
    Process frames from the camera stream and enqueue them.

    If `framerate` is provided, we *skip frames* so that only approximately
    `framerate` frames per second are processed and displayed.

    The camera can still run at its native FPS (e.g. 30 FPS), but we only
    use every N-th frame. This gives the effect of 1 FPS / 5 FPS / 10 FPS
    in the live view without adding artificial lag.

    Args:
        cap (cv2.VideoCapture): VideoCapture object.
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
    cap: Optional[cv2.VideoCapture],
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

def resolve_output_resolution_arg(res_arg: Optional[list[str]]) -> Optional[Tuple[int, int]]:
    """
    Parse --output-resolution argument.

    Supported:
      --output-resolution sd|hd|fhd
      --output-resolution 1920 1080
    """
    if res_arg is None:
        return None

    # Single token: preset name (sd/hd/fhd)
    if len(res_arg) == 1:
        key = res_arg[0]
        if key in CAMERA_RESOLUTION_MAP:
            return CAMERA_RESOLUTION_MAP[key]
        raise ValueError(
            f"Invalid --output-resolution value '{key}'. "
            "Use 'sd', 'hd', 'fhd' or two integers, e.g. '--output-resolution 1920 1080'."
        )

    # Two tokens: custom width/height
    if len(res_arg) == 2 and all(x.isdigit() for x in res_arg):
        w, h = map(int, res_arg)
        if w <= 0 or h <= 0:
            raise ValueError("Custom --output-resolution width/height must be positive integers.")
        return (w, h)

    raise ValueError(
        f"Invalid --output-resolution value: {res_arg}. "
        "Use 'sd', 'hd', 'fhd' or two integers, e.g. '--output-resolution 1920 1080'."
    )


def list_networks(app: str) -> None:
    """
    Print the supported networks for a given application.

    Note: This is a stub implementation for HACE compatibility.
    In hailo-apps, use config_manager.get_model_names() instead.
    """
    logger.warning(
        f"list_networks() called for app '{app}'. "
        "This is a compatibility stub. "
        "For full functionality, use hailo_apps.config.config_manager.get_model_names()"
    )
    try:
        from hailo_apps.config.config_manager import get_model_names

        arch = resolve_arch(None)
        models = get_model_names(app, arch, tier="all")
        if models:
            logger.info(f"Available models for {app} ({arch}):")
            for model in models:
                logger.info(f"  - {model}")
        else:
            logger.info(f"No models found for {app} ({arch})")
    except ImportError:
        logger.error("Could not import config_manager. Please ensure hailo-apps is properly installed.")


def list_inputs(app: str) -> None:
    """
    List predefined inputs for a given application.

    Note: This is a stub implementation for HACE compatibility.
    """
    logger.warning(
        f"list_inputs() called for app '{app}'. "
        "This is a compatibility stub. "
        "Inputs should be specified as file paths or 'camera'."
    )


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


def resolve_net_arg(app: str, net_arg: str | None, dest_dir: str = "hefs", arch: str | None = None) -> str:
    """
    Resolve the --net argument into a concrete HEF path.

    Note: This is a compatibility function for HACE apps.
    In hailo-apps, prefer using core.resolve_hef_path() directly.
    """
    if net_arg is None:
        logger.error("No --net was provided.")
        list_networks(app)
        sys.exit(1)

    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    candidate = Path(net_arg)

    # If it's an existing HEF file, use it
    if candidate.exists() and candidate.is_file() and candidate.suffix == ".hef":
        logger.info(f"Using local HEF file: {candidate.resolve()}")
        return str(candidate.resolve())

    # If it has .hef extension but doesn't exist, error
    if candidate.suffix == ".hef":
        logger.error(f"HEF file not found: {net_arg}")
        list_networks(app)
        sys.exit(1)

    # Treat as model name - try to resolve using hailo-apps mechanism
    model_name = net_arg
    existing_hef = dest_path / f"{model_name}.hef"

    if existing_hef.exists():
        logger.info(f"Using existing HEF: {existing_hef.resolve()}")
        return str(existing_hef.resolve())

    # Try to resolve using hailo-apps's config system
    resolved_arch = resolve_arch(arch)
    try:
        from hailo_apps.python.core.common.core import resolve_hef_path
        resolved = resolve_hef_path(model_name, app, resolved_arch)
        if resolved and resolved.exists():
            logger.info(f"Resolved model '{model_name}' to: {resolved} (arch={resolved_arch})")
            return str(resolved)
    except Exception as e:
        logger.debug(f"Could not resolve via config system: {e}")

    # Fallback: error and suggest listing networks
    logger.error(f"Model '{model_name}' not found locally and could not be resolved.")
    logger.info("Please provide a full path to a .hef file, or use --list-nets to see available models.")
    list_networks(app)
    sys.exit(1)


def resolve_input_arg(app: str, input_arg: str | None) -> str:
    """
    Resolve the --input argument into a concrete input source.

    Note: This is a compatibility function for HACE apps.
    """
    # Map standalone app names to their base resource tag names
    APP_NAME_MAPPING = {
        "object_detection": "detection",
        "simple_detection": "simple_detection",
        "instance_segmentation": "instance_segmentation",
        "super_resolution": "super_resolution",
    }

    def resolve_tagged_resource(app_name: str, preferred_name: str | None = None) -> str | None:
        """Resolve a resource listed in resources_config.yaml for this app."""
        try:
            from hailo_apps.config.config_manager import get_inputs_for_app
        except Exception:
            return None

        # Map app name to base resource tag if needed
        resource_app_name = APP_NAME_MAPPING.get(app_name, app_name)
        inputs = get_inputs_for_app(resource_app_name, is_standalone=True)

        def pick(section: str) -> str | None:
            for entry in inputs.get(section, []):
                name = entry.get("name")
                if preferred_name and preferred_name != name:
                    continue
                resource_type = RESOURCES_PHOTOS_DIR_NAME if section == "images" else RESOURCES_VIDEOS_DIR_NAME
                resolved = get_resource_path(
                    pipeline_name=None,
                    resource_type=resource_type,
                    arch=None,
                    model=name,
                )
                if resolved and resolved.exists():
                    return str(resolved)
            return None

        # Prefer images first, then videos
        resolved = pick("images")
        if resolved:
            return resolved
        return pick("videos")

    if input_arg is None:
        resolved = resolve_tagged_resource(app)
        if resolved:
            logger.info("No input provided; using default bundled resource for %s: %s", app, resolved)
            return resolved

        logger.error(
            "No --input was provided and no bundled resource was found. "
            "Please specify -i/--input with a file path, directory, or 'camera'."
        )
        sys.exit(1)

    # "camera" stays as is
    if input_arg == "camera":
        return input_arg

    path_candidate = Path(input_arg)

    # If it already exists (file or dir), just use it as-is
    if path_candidate.exists():
        return str(path_candidate)

    resource_path = resolve_tagged_resource(app, path_candidate.name)
    if resource_path:
        logger.info("Resolved input '%s' to bundled resource: %s", input_arg, resource_path)
        return resource_path

    # If it has an extension but does NOT exist -> error
    if path_candidate.suffix:
        logger.error(f"Input file not found: {input_arg}")
        logger.info("Please provide a valid file path, directory, or 'camera'.")
        sys.exit(1)

    # No extension and path does not exist -> treat as logical ID
    logger.warning(
        f"Input '{input_arg}' does not exist as a local file or directory. "
        "Treating as logical input ID, but download functionality is not implemented in this compatibility stub."
    )
    logger.info("Please provide a full file path, directory path, or 'camera'.")
    sys.exit(1)
