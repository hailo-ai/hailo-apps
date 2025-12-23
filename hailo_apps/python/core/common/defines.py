from pathlib import Path

# Base Defaults
HAILO8_ARCH = "hailo8"
HAILO8L_ARCH = "hailo8l"
HAILO10H_ARCH = "hailo10h"
AUTO_DETECT = "auto"
HAILO_TAPPAS_CORE = "hailo-tappas-core"
HAILO_TAPPAS_CORE_PYTHON_NAMES = [
    "hailo-tappas-core-python-binding",
    "tappas-core-python-binding",
    HAILO_TAPPAS_CORE,
]
HAILORT_PACKAGE_NAME = "hailort"
HAILORT_PACKAGE_NAME_RPI = "h10-hailort"
HAILO_FILE_EXTENSION = ".hef"
MODEL_ZOO_URL = "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled"
S3_RESOURCES_BASE_URL = "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources"
RESOURCES_ROOT_PATH_DEFAULT = "/usr/local/hailo/resources"  # Do Not Change!
SHARED_VDEVICE_GROUP_ID = "SHARED"  # Do Not Change!

# Core defaults
ARM_POSSIBLE_NAME = ["arm", "aarch64"]
X86_POSSIBLE_NAME = ["x86", "amd64", "x86_64"]
RPI_POSSIBLE_NAME = ["rpi", "raspberrypi", "pi"]
HAILO8_ARCH_CAPS = "HAILO8"
HAILO8L_ARCH_CAPS = "HAILO8L"
HAILO10H_ARCH_CAPS = "HAILO10H"
HAILO15H_ARCH_CAPS = "HAILO15H"
HAILO_FW_CONTROL_CMD = "hailortcli fw-control identify"
X86_NAME_I = "x86"
RPI_NAME_I = "rpi"
ARM_NAME_I = "arm"
LINUX_SYSTEM_NAME_I = "linux"
UNKNOWN_NAME_I = "unknown"
USB_CAMERA = "usb"
JSON_FILE_EXTENSION = ".json"

# CLI defaults
PIP_CMD = "pip3"

# Base project paths
REPO_ROOT = Path(__file__).resolve().parents[4]


def _get_package_config_path(filename: str) -> Path | None:
    """Get config file path from package location (when installed via pip)."""
    try:
        import hailo_apps.config
        package_dir = Path(hailo_apps.config.__file__).parent
        config_path = package_dir / filename
        if config_path.exists():
            return config_path
    except (ImportError, AttributeError):
        pass
    return None


def _get_repo_config_path(filename: str) -> Path:
    """Get config file path from repo location (when running from source)."""
    return REPO_ROOT / "hailo_apps" / "config" / filename


def _get_config_path(filename: str) -> str:
    """Get config path: first check package location, then fall back to repo location."""
    # First try package location (when installed via pip)
    package_path = _get_package_config_path(filename)
    if package_path:
        return str(package_path)
    # Fall back to repo location (when running from source)
    return str(_get_repo_config_path(filename))


# Default config paths - checks package location first, then repo location
DEFAULT_CONFIG_PATH = _get_config_path("config.yaml")
DEFAULT_RESOURCES_CONFIG_PATH = _get_config_path("resources_config.yaml")

# Symlink, dotenv, local resources defaults
DEFAULT_RESOURCES_SYMLINK_PATH = str(REPO_ROOT / "resources")  # e.g. created by post-install
DEFAULT_DOTENV_PATH = "/usr/local/hailo/resources/.env"  # your env file lives here
DEFAULT_LOCAL_RESOURCES_PATH = str(REPO_ROOT / "local_resources")  # bundled GIFs, JSON, etc.

# Supported config options (used for validation in config_utils.py)
VALID_HAILORT_VERSION = [AUTO_DETECT, "4.23.0", "5.1.1"]
VALID_TAPPAS_VERSION = [AUTO_DETECT, "5.1.0"]
VALID_H10_MODEL_ZOO_VERSION = ["v5.1.0", "v5.1.1"]
VALID_H8_MODEL_ZOO_VERSION = ["v2.17.0"]
VALID_MODEL_ZOO_VERSION = VALID_H10_MODEL_ZOO_VERSION + VALID_H8_MODEL_ZOO_VERSION
VALID_HOST_ARCH = [AUTO_DETECT, "x86", "rpi", "arm"]
VALID_HAILO_ARCH = [AUTO_DETECT, HAILO8_ARCH, HAILO8L_ARCH, HAILO10H_ARCH]

# Config key constants
HAILORT_VERSION_KEY = "hailort_version"
TAPPAS_VERSION_KEY = "tappas_version"
MODEL_ZOO_VERSION_KEY = "model_zoo_version"
HOST_ARCH_KEY = "host_arch"
HAILO_ARCH_KEY = "hailo_arch"
RESOURCES_PATH_KEY = "resources_path"
VIRTUAL_ENV_NAME_KEY = "virtual_env_name"
TAPPAS_POSTPROC_PATH_KEY = "tappas_postproc_path"
HAILO_APPS_PATH_KEY = "hailo_apps_path"
HAILO_LOG_LEVEL_KEY = "HAILO_LOG_LEVEL"

# Environment variable groups
DIC_CONFIG_VARIANTS = [
    HAILORT_VERSION_KEY,
    TAPPAS_VERSION_KEY,
    MODEL_ZOO_VERSION_KEY,
    HOST_ARCH_KEY,
    HAILO_ARCH_KEY,
    RESOURCES_PATH_KEY,
    VIRTUAL_ENV_NAME_KEY,
    TAPPAS_POSTPROC_PATH_KEY,
]

# Default config values
HAILORT_VERSION_DEFAULT = AUTO_DETECT
TAPPAS_VERSION_DEFAULT = AUTO_DETECT
HOST_ARCH_DEFAULT = AUTO_DETECT
HAILO_ARCH_DEFAULT = AUTO_DETECT
MODEL_ZOO_VERSION_DEFAULT = "v2.17.0"
RESOURCES_PATH_DEFAULT = RESOURCES_ROOT_PATH_DEFAULT
VIRTUAL_ENV_NAME_DEFAULT = "venv_hailo_apps"

# Default TAPPAS post-processing directory - set via environment variable during installation
# The installer runs: pkg-config --variable=tappas_postproc_lib_dir hailo-tappas-core
# and stores the result in the .env file as TAPPAS_POSTPROC_PATH
TAPPAS_POSTPROC_PATH_DEFAULT = ""  # Will be populated from environment at runtime

# Resources directory structure
RESOURCES_MODELS_DIR_NAME = "models"
RESOURCES_VIDEOS_DIR_NAME = "videos"
RESOURCES_SO_DIR_NAME = "so"
RESOURCES_PHOTOS_DIR_NAME = "images"  # Changed from "photos" to match actual directory name
RESOURCES_JSON_DIR_NAME = "json"

# Depth pipeline defaults
DEPTH_APP_TITLE = "Hailo Depth App"
DEPTH_PIPELINE = "depth"
DEPTH_POSTPROCESS_SO_FILENAME = "libdepth_postprocess.so"
DEPTH_POSTPROCESS_FUNCTION = "filter_scdepth"
DEPTH_MODEL_NAME = "scdepthv3"

# Simple detection pipeline defaults
SIMPLE_DETECTION_APP_TITLE = "Hailo Simple Detection App"
SIMPLE_DETECTION_PIPELINE = "simple_detection"
SIMPLE_DETECTION_VIDEO_NAME = "example_640.mp4"
SIMPLE_DETECTION_MODEL_NAME = "yolov6n"
SIMPLE_DETECTION_POSTPROCESS_SO_FILENAME = "libyolo_hailortpp_postprocess.so"
SIMPLE_DETECTION_POSTPROCESS_FUNCTION = "filter"

# Detection pipeline defaults
DETECTION_APP_TITLE = "Hailo Detection App"
DETECTION_PIPELINE = "detection"
DETECTION_MODEL_NAME_H8 = "yolov8m"
DETECTION_MODEL_NAME_H8L = "yolov8s"
DETECTION_POSTPROCESS_SO_FILENAME = "libyolo_hailortpp_postprocess.so"
DETECTION_POSTPROCESS_FUNCTION = "filter_letterbox"

# Instance segmentation pipeline defaults
INSTANCE_SEGMENTATION_APP_TITLE = "Hailo Instance Segmentation App"
INSTANCE_SEGMENTATION_PIPELINE = "instance_segmentation"
INSTANCE_SEGMENTATION_POSTPROCESS_SO_FILENAME = "libyolov5seg_postprocess.so"
INSTANCE_SEGMENTATION_POSTPROCESS_FUNCTION = "filter_letterbox"
INSTANCE_SEGMENTATION_MODEL_NAME_H8 = "yolov5m_seg"
INSTANCE_SEGMENTATION_MODEL_NAME_H8L = "yolov5n_seg"

# Pose estimation pipeline defaults
POSE_ESTIMATION_APP_TITLE = "Hailo Pose Estimation App"
POSE_ESTIMATION_PIPELINE = "pose_estimation"
POSE_ESTIMATION_POSTPROCESS_SO_FILENAME = "libyolov8pose_postprocess.so"
POSE_ESTIMATION_POSTPROCESS_FUNCTION = "filter_letterbox"
POSE_ESTIMATION_MODEL_NAME_H8 = "yolov8m_pose"
POSE_ESTIMATION_MODEL_NAME_H8L = "yolov8s_pose"

# Face recognition pipeline defaults
FACE_RECOGNITION_APP_TITLE = "Hailo Face Recognition App"
FACE_DETECTION_PIPELINE = "face_detection"
FACE_DETECTION_MODEL_NAME_H8 = "scrfd_10g"
FACE_DETECTION_MODEL_NAME_H8L = "scrfd_2.5g"
FACE_RECOGNITION_PIPELINE = "face_recognition"
FACE_RECOGNITION_MODEL_NAME_H8 = "arcface_mobilefacenet"
FACE_RECOGNITION_MODEL_NAME_H8L = "arcface_mobilefacenet"
FACE_DETECTION_POSTPROCESS_SO_FILENAME = "libscrfd.so"
FACE_RECOGNITION_POSTPROCESS_SO_FILENAME = "libface_recognition_post.so"
FACE_ALIGN_POSTPROCESS_SO_FILENAME = "libvms_face_align.so"
FACE_CROP_POSTPROCESS_SO_FILENAME = "libvms_croppers.so"
FACE_RECOGNITION_VIDEO_NAME = "face_recognition.mp4"
FACE_RECON_DATABASE_DIR_NAME = "database"
FACE_RECON_TRAIN_DIR_NAME = "train"
FACE_RECON_SAMPLES_DIR_NAME = "samples"
FACE_RECON_LOCAL_SAMPLES_DIR_NAME = "faces"
FACE_DETECTION_JSON_NAME = "scrfd.json"
VMS_CROPPER_POSTPROCESS_FUNCTION = "face_recognition"
ARCFACE_MOBILEFACENET_POSTPROCESS_FUNCTION = "filter"
SCRFD_10G_POSTPROCESS_FUNCTION = "scrfd_10g_letterbox"  # Used for hailo8 and hailo10h
SCRFD_2_5G_POSTPROCESS_FUNCTION = "scrfd_2_5g_letterbox"  # Used for hailo8l

# Clip pipeline defaults
CLIP_PIPELINE = "clip"
CLIP_MODEL_NAME = "clip_resnet_50x4_image_encoder"
CLIP_DETECTION_PIPELINE = "clip_detection"
CLIP_DETECTION_MODEL_NAME = "hailo_yolov8n_4_classes_vga"
CLIP_APP_TITLE = "Hailo CLIP App"
CLIP_VIDEO_NAME = "clip_example.mp4"
CLIP_POSTPROCESS_SO_FILENAME = 'libclip_postprocess.so'
CLIP_CROPPER_POSTPROCESS_SO_FILENAME = 'libclip_croppers_postprocess.so'
CLIP_POSTPROCESS_FUNCTION_NAME = 'filter'
CLIP_DETECTION_POSTPROCESS_FUNCTION_NAME = 'filter'
CLIP_CROPPER_PERSON_POSTPROCESS_FUNCTION_NAME = 'person_cropper'
CLIP_CROPPER_VEHICLE_POSTPROCESS_FUNCTION_NAME = 'vehicle_cropper'
CLIP_CROPPER_FACE_POSTPROCESS_FUNCTION_NAME = 'face_cropper'
CLIP_CROPPER_LICENSE_PLATE_POSTPROCESS_FUNCTION_NAME = 'license_plate_cropper'
CLIP_CROPPER_OBJECT_POSTPROCESS_FUNCTION_NAME = 'object_cropper'
CLIP_DETECTOR_TYPE_PERSON = 'person'
CLIP_DETECTOR_TYPE_VEHICLE = 'vehicle'
CLIP_DETECTOR_TYPE_FACE = 'face'
CLIP_DETECTOR_TYPE_LICENSE_PLATE = 'license-plate'

# Multisource pipeline defaults
MULTI_SOURCE_APP_TITLE = "Hailo Multisource App"
TAPPAS_STREAM_ID_TOOL_SO_FILENAME = 'libstream_id_tool.so'

# REID Multisource pipeline defaults
REID_MULTISOURCE_PIPELINE = "reid_multisource"
REID_MULTISOURCE_APP_TITLE = "Hailo REID Multisource App"
REID_MULTI_SOURCE_DATABASE_DIR_NAME = "database"
REID_POSTPROCESS_SO_FILENAME = "librepvgg_reid_postprocess.so"
ALL_DETECTIONS_CROPPER_POSTPROCESS_SO_FILENAME = "liball_detections_cropper_postprocess.so"
REID_CROPPER_POSTPROCESS_FUNCTION = 'all_detections'
REID_POSTPROCESS_FUNCTION = 'filter'
REID_CLASSIFICATION_TYPE = 'reid'

# TILING pipeline defaults
TILING_PIPELINE = "tiling"
TILING_APP_TITLE = "Hailo Tiling App"
TILING_VIDEO_EXAMPLE_NAME = "tiling_visdrone_720p.mp4"
TILING_MODEL_NAME = "hailo_yolov8n_4_classes_vga"
TILING_POSTPROCESS_SO_FILENAME = "libyolo_hailortpp_postprocess.so"
TILING_POSTPROCESS_FUNCTION = "filter"

# Testing defaults
TEST_RUN_TIME = 10  # seconds
TERM_TIMEOUT = 5  # seconds

# USB device discovery
UDEV_CMD = "udevadm"

# Video format defaults
HAILO_RGB_VIDEO_FORMAT = "RGB"
HAILO_YUYV_VIDEO_FORMAT = "YUYV"
HAILO_NV12_VIDEO_FORMAT = "NV12"

# Video examples
BASIC_PIPELINES_VIDEO_EXAMPLE_NAME = "example.mp4"

# Gstreamer pipeline defaults
GST_VIDEO_SINK = "autovideosink"

# Gen AI app defaults
VLM_CHAT_APP = "vlm_chat"
LLM_CHAT_APP = "llm_chat"
WHISPER_CHAT_APP = "whisper_chat"
AGENT_APP = "agent"

# Gen AI model defaults
VLM_MODEL_NAME_H10 = "Qwen2-VL-2B-Instruct"
LLM_MODEL_NAME_H10 = "Qwen2.5-1.5B-Instruct"
LLM_CODER_MODEL_NAME_H10 = "Qwen2.5-Coder-1.5B-Instruct"
WHISPER_MODEL_NAME_H10 = "Whisper-Base"

# Whisper defaults
TARGET_SR = 16000  # Target sample rate for audio recording (in Hz)
TARGET_PLAYBACK_SR = 48000  # Target sample rate for playback (standard for hardware compatibility)
CHUNK_SIZE = 1024  # Number of frames per buffer

VOICE_ASSISTANT_APP = "voice_assistant"
VOICE_ASSISTANT_MODEL_NAME = "Qwen2.5-1.5B-Instruct"

# Piper TTS defaults
TTS_MODEL_NAME = "en_US-amy-low"
TTS_MODELS_DIR = str(REPO_ROOT / "local_resources" / "piper_models")
TTS_ONNX_PATH = str(Path(TTS_MODELS_DIR) / f"{TTS_MODEL_NAME}.onnx")
TTS_JSON_PATH = str(Path(TTS_MODELS_DIR) / f"{TTS_MODEL_NAME}.onnx.json")
TTS_VOLUME = 0.8  # Volume (0.0 to 1.0)
TTS_LENGTH_SCALE = 0.6  # Speech rate (lower is faster)
TTS_NOISE_SCALE = 0.6  # Voice variability (lower is more consistent)
TTS_W_SCALE = 0.6  # Pronunciation variability (lower is more consistent)
LLM_PROMPT_PREFIX = "Respond in up to three sentences. "
TEMP_WAV_DIR = "/tmp"

# OCR pipeline defaults
PADDLE_OCR_PIPELINE = "paddle_ocr"
OCR_APP_TITLE = "Hailo OCR App"
OCR_DETECTION_MODEL_NAME = "ocr_det"
OCR_RECOGNITION_MODEL_NAME = "ocr"
OCR_POSTPROCESS_SO_FILENAME = "libocr_postprocess.so"
OCR_DETECTION_POSTPROCESS_FUNCTION = "paddleocr_det"
OCR_RECOGNITION_POSTPROCESS_FUNCTION = "paddleocr_recognize"
OCR_CROPPER_FUNCTION = "crop_text_regions"
OCR_VIDEO_NAME = "ocr.mp4"

DEFAULT_COCO_LABELS_PATH = str(REPO_ROOT / "local_resources" / "coco.txt")
