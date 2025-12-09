"""Core helpers: arch detection, parser, buffer utils."""

import argparse
import os
import queue
from pathlib import Path

from dotenv import load_dotenv

from .defines import (
    DEFAULT_DOTENV_PATH,
    DEFAULT_LOCAL_RESOURCES_PATH,
    DEPTH_MODEL_NAME,
    DEPTH_PIPELINE,
    DETECTION_MODEL_NAME_H8,
    DETECTION_MODEL_NAME_H8L,
    DETECTION_PIPELINE,
    DIC_CONFIG_VARIANTS,
    FACE_DETECTION_MODEL_NAME_H8,
    FACE_DETECTION_MODEL_NAME_H8L,
    FACE_DETECTION_PIPELINE,
    FACE_RECOGNITION_MODEL_NAME_H8,
    FACE_RECOGNITION_MODEL_NAME_H8L,
    FACE_RECOGNITION_PIPELINE,
    HAILO8_ARCH,
    HAILO10H_ARCH,
    HAILO_ARCH_KEY,
    HAILO_FILE_EXTENSION,
    INSTANCE_SEGMENTATION_MODEL_NAME_H8,
    INSTANCE_SEGMENTATION_MODEL_NAME_H8L,
    INSTANCE_SEGMENTATION_PIPELINE,
    POSE_ESTIMATION_MODEL_NAME_H8,
    POSE_ESTIMATION_MODEL_NAME_H8L,
    POSE_ESTIMATION_PIPELINE,
    RESOURCES_JSON_DIR_NAME,
    RESOURCES_MODELS_DIR_NAME,
    # for get_resource_path
    RESOURCES_PHOTOS_DIR_NAME,
    RESOURCES_ROOT_PATH_DEFAULT,
    RESOURCES_SO_DIR_NAME,
    RESOURCES_VIDEOS_DIR_NAME,
    SIMPLE_DETECTION_MODEL_NAME,
    SIMPLE_DETECTION_PIPELINE,
    CLIP_PIPELINE,
    CLIP_MODEL_NAME,
    CLIP_DETECTION_PIPELINE,
    CLIP_DETECTION_MODEL_NAME
)
from .hailo_logger import get_logger
from .installation_utils import detect_hailo_arch

hailo_logger = get_logger(__name__)


def load_environment(env_file=DEFAULT_DOTENV_PATH, required_vars=None) -> bool:
    hailo_logger.debug(f"Loading environment from: {env_file}")
    if env_file is None:
        env_file = DEFAULT_DOTENV_PATH
    load_dotenv(dotenv_path=env_file)

    env_path = Path(env_file)
    if not os.path.exists(env_path):
        hailo_logger.warning(f".env file not found: {env_file}")
        print(f"⚠️ .env file not found: {env_file}")
        return False
    if not os.access(env_path, os.R_OK):
        hailo_logger.warning(f".env file not readable: {env_file}")
        print(f"⚠️ .env file not readable: {env_file}")
        return False
    if not os.access(env_path, os.W_OK):
        hailo_logger.warning(f".env file not writable: {env_file}")
        print(f"⚠️ .env file not writable: {env_file}")
        return False
    if not os.access(env_path, os.F_OK):
        hailo_logger.warning(f".env file not found (F_OK): {env_file}")
        print(f"⚠️ .env file not found: {env_file}")
        return False

    if required_vars is None:
        required_vars = DIC_CONFIG_VARIANTS
    missing = []
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing.append(var)

    if missing:
        hailo_logger.warning(f"Missing environment variables: {missing}")
        print("⚠️ Missing environment variables: %s", ", ".join(missing))
        return False
    hailo_logger.info("All required environment variables loaded successfully.")
    print("✅ All required environment variables loaded.")
    return True


def get_base_parser():
    """
    Creates the base argument parser with core flags shared by all Hailo applications.

    This parser defines the standard interface for common functionality across
    all applications, ensuring consistent flag naming and behavior.

    Returns:
        argparse.ArgumentParser: Base parser with core flags
    """
    hailo_logger.debug("Creating base argparse parser.")
    parser = argparse.ArgumentParser(
        description="Hailo Application Base Parser",
        add_help=False  # Allow parent parsers to control help display
    )

    # Core input/output flags
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help=(
            "Input source for processing. Can be a file path (image or video), "
            "camera index (integer), folder path containing images, or RTSP URL. "
            "For USB cameras, use 'usb' to auto-detect or '/dev/video<X>' for a specific device. "
            "For Raspberry Pi camera, use 'rpi'. If not specified, defaults to application-specific source."
        )
    )

    parser.add_argument(
        "--hef-path", "-n",
        type=str,
        default=None,
        help=(
            "Path to Hailo Executable Format (HEF) model file. "
            "The HEF file contains the compiled neural network model optimized for Hailo processors. "
            "If not specified, the application will attempt to use a default model based on the pipeline type."
        )
    )

    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1,
        help=(
            "Number of frames or images to process in parallel during inference. "
            "Higher batch sizes can improve throughput but require more memory. "
            "Default is 1 (sequential processing)."
        )
    )

    parser.add_argument(
        "--labels", "-l",
        type=str,
        default=None,
        help=(
            "Path to a text file containing class labels, one per line. "
            "Used for mapping model output indices to human-readable class names. "
            "If not specified, default labels for the model will be used (e.g., COCO labels for detection models)."
        )
    )

    parser.add_argument(
        "--width", "-W",
        type=int,
        default=None,
        help=(
            "Custom output width in pixels for video or image output. "
            "If specified, the output will be resized to this width while maintaining aspect ratio. "
            "If not specified, uses the input resolution or model default."
        )
    )

    parser.add_argument(
        "--height", "-H",
        type=int,
        default=None,
        help=(
            "Custom output height in pixels for video or image output. "
            "If specified, the output will be resized to this height while maintaining aspect ratio. "
            "If not specified, uses the input resolution or model default."
        )
    )

    parser.add_argument(
        "--arch", "-a",
        type=str,
        default=None,
        choices=["hailo8", "hailo8l", "hailo10h"],
        help=(
            "Target Hailo architecture for model execution. "
            "Options: 'hailo8' (Hailo-8 processor), 'hailo8l' (Hailo-8L processor), "
            "'hailo10h' (Hailo-10H processor). "
            "If not specified, the architecture will be auto-detected from the connected device."
        )
    )

    parser.add_argument(
        "--show-fps",
        action="store_true",
        help=(
            "Enable FPS (frames per second) counter display. "
            "When enabled, the application will display real-time performance metrics "
            "showing the current processing rate. Useful for performance monitoring and optimization."
        )
    )

    parser.add_argument(
        "--save-output", "-s",
        action="store_true",
        help=(
            "Enable output file saving. When enabled, processed images or videos will be saved to disk. "
            "The output location is determined by the --output-dir flag (for standalone apps) "
            "or application-specific defaults. Without this flag, output is only displayed (if applicable)."
        )
    )

    parser.add_argument(
        "--frame-rate", "-f",
        type=int,
        default=30,
        help=(
            "Target frame rate for video processing in frames per second. "
            "Controls the playback speed and processing rate for video sources. "
            "Default is 30 FPS. Lower values reduce processing load, higher values increase throughput."
        )
    )

    return parser


def get_pipeline_parser():
    """
    Creates an argument parser for GStreamer pipeline applications.

    This parser extends the base parser with pipeline-specific flags for
    GStreamer-based applications that process video streams in real-time.

    Returns:
        argparse.ArgumentParser: Parser with base and pipeline-specific flags
    """
    hailo_logger.debug("Creating pipeline argparse parser.")
    base_parser = get_base_parser()
    parser = argparse.ArgumentParser(
        description="Hailo GStreamer Pipeline Application",
        parents=[base_parser],
        add_help=True  # Enable --help flag to show all available options
    )

    parser.add_argument(
        "--use-frame",
        action="store_true",
        help=(
            "Enable frame access in callback functions. "
            "When enabled, the callback function receives access to the raw frame data, "
            "allowing for custom processing, analysis, or visualization within the pipeline. "
            "Useful for applications that need to perform additional operations on individual frames."
        )
    )

    parser.add_argument(
        "--disable-sync",
        action="store_true",
        help=(
            "Disable display sink synchronization. "
            "When enabled, the pipeline will process frames as fast as possible without waiting "
            "for display synchronization. This is particularly useful when processing from file sources "
            "where you want maximum throughput rather than real-time playback speed."
        )
    )

    parser.add_argument(
        "--disable-callback",
        action="store_true",
        help=(
            "Skip user callback execution. "
            "When enabled, the pipeline will run without invoking custom callback functions, "
            "processing frames through the standard pipeline only. Useful for performance testing "
            "or when you want to run the pipeline without custom post-processing logic."
        )
    )

    parser.add_argument(
        "--dump-dot",
        action="store_true",
        help=(
            "Export pipeline graph to DOT file. "
            "When enabled, the GStreamer pipeline structure will be saved as a Graphviz DOT file "
            "(typically named 'pipeline.dot'). This file can be visualized using tools like 'dot' "
            "to understand the pipeline topology and debug pipeline configuration issues."
        )
    )

    return parser


def get_standalone_parser():
    """
    Creates an argument parser for standalone processing applications.

    This parser extends the base parser with standalone-specific flags for
    applications that process files or batches without GStreamer pipelines.

    Returns:
        argparse.ArgumentParser: Parser with base and standalone-specific flags
    """
    hailo_logger.debug("Creating standalone argparse parser.")
    base_parser = get_base_parser()
    parser = argparse.ArgumentParser(
        description="Hailo Standalone Processing Application",
        parents=[base_parser],
        add_help=True  # Enable --help flag to show all available options
    )

    parser.add_argument(
        "--track",
        action="store_true",
        help=(
            "Enable object tracking for detections. "
            "When enabled, detected objects will be tracked across frames using a tracking algorithm "
            "(e.g., ByteTrack). This assigns consistent IDs to objects over time, enabling temporal analysis, "
            "trajectory visualization, and multi-frame association. Useful for video processing applications."
        )
    )

    parser.add_argument(
        "--resolution", "-r",
        type=str,
        choices=["sd", "hd", "fhd"],
        default="sd",
        help=(
            "Predefined resolution for camera input sources. "
            "Options: 'sd' (640x480, Standard Definition), 'hd' (1280x720, High Definition), "
            "'fhd' (1920x1080, Full High Definition). "
            "Default is 'sd'. This flag is only applicable when using camera input sources."
        )
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help=(
            "Directory where output files will be saved. "
            "When --save-output is enabled, processed images, videos, or result files will be "
            "written to this directory. If not specified, outputs are saved to a default location "
            "or the current working directory. The directory will be created if it does not exist."
        )
    )

    return parser


def get_default_parser():
    """
    Legacy function for backward compatibility.

    Returns the pipeline parser as the default to maintain compatibility
    with existing code that uses get_default_parser().

    Returns:
        argparse.ArgumentParser: Pipeline parser (for backward compatibility)
    """
    hailo_logger.warning(
        "get_default_parser() is deprecated. Use get_pipeline_parser() or get_standalone_parser() instead."
    )
    return get_pipeline_parser()


def get_model_name(pipeline_name: str, arch: str) -> str:
    hailo_logger.debug(f"Getting model name for pipeline={pipeline_name}, arch={arch}")
    is_h8 = arch in (HAILO8_ARCH, HAILO10H_ARCH)
    pipeline_map = {
        DEPTH_PIPELINE: DEPTH_MODEL_NAME,
        CLIP_PIPELINE: CLIP_MODEL_NAME,
        CLIP_DETECTION_PIPELINE: CLIP_DETECTION_MODEL_NAME,
        SIMPLE_DETECTION_PIPELINE: SIMPLE_DETECTION_MODEL_NAME,
        DETECTION_PIPELINE: DETECTION_MODEL_NAME_H8 if is_h8 else DETECTION_MODEL_NAME_H8L,
        INSTANCE_SEGMENTATION_PIPELINE: INSTANCE_SEGMENTATION_MODEL_NAME_H8
        if is_h8
        else INSTANCE_SEGMENTATION_MODEL_NAME_H8L,
        POSE_ESTIMATION_PIPELINE: POSE_ESTIMATION_MODEL_NAME_H8
        if is_h8
        else POSE_ESTIMATION_MODEL_NAME_H8L,
        FACE_DETECTION_PIPELINE: FACE_DETECTION_MODEL_NAME_H8
        if is_h8
        else FACE_DETECTION_MODEL_NAME_H8L,
        FACE_RECOGNITION_PIPELINE: FACE_RECOGNITION_MODEL_NAME_H8
        if is_h8
        else FACE_RECOGNITION_MODEL_NAME_H8L
    }
    name = pipeline_map[pipeline_name]
    hailo_logger.debug(f"Resolved model name: {name}")
    return name


def get_resource_path(
    pipeline_name: str, resource_type: str, arch: str | None = None, model: str | None = None
) -> Path | None:
    hailo_logger.debug(
        f"Getting resource path for pipeline={pipeline_name}, resource_type={resource_type}, model={model}"
    )
    root = Path(RESOURCES_ROOT_PATH_DEFAULT)
    # Auto-detect arch if not provided and needed for RESOURCES_MODELS_DIR_NAME
    if arch is None and resource_type == RESOURCES_MODELS_DIR_NAME:
        arch = os.getenv(HAILO_ARCH_KEY) or detect_hailo_arch()
        hailo_logger.debug(f"Auto-detected arch: {arch}")
    if not arch and resource_type == RESOURCES_MODELS_DIR_NAME:
        hailo_logger.error("Could not detect Hailo architecture.")
        assert False, "Could not detect Hailo architecture."


    if resource_type == RESOURCES_SO_DIR_NAME and model:
        return root / RESOURCES_SO_DIR_NAME / model
    if resource_type == RESOURCES_VIDEOS_DIR_NAME and model:
        return root / RESOURCES_VIDEOS_DIR_NAME / model
    if resource_type == RESOURCES_PHOTOS_DIR_NAME and model:
        return root / RESOURCES_PHOTOS_DIR_NAME / model
    if resource_type == RESOURCES_JSON_DIR_NAME and model:
        return root / RESOURCES_JSON_DIR_NAME / model
    if resource_type == DEFAULT_LOCAL_RESOURCES_PATH and model:
        return root / DEFAULT_LOCAL_RESOURCES_PATH / model

    if resource_type == RESOURCES_MODELS_DIR_NAME:
        if model:
            model_path = root / RESOURCES_MODELS_DIR_NAME / arch / model
            if "." in model:
                return model_path.with_name(model_path.name + HAILO_FILE_EXTENSION)
            return model_path.with_suffix(HAILO_FILE_EXTENSION)
        if pipeline_name:
            name = get_model_name(pipeline_name, arch)
            name_path = root / RESOURCES_MODELS_DIR_NAME / arch / name
            if "." in name:
                return name_path.with_name(name_path.name + HAILO_FILE_EXTENSION)
            return name_path.with_suffix(HAILO_FILE_EXTENSION)
    return None


class FIFODropQueue(queue.Queue):
    def put(self, item, block=False, timeout=None):
        if self.full():
            hailo_logger.debug("Queue full, dropping oldest item.")
            self.get_nowait()
        super().put(item, block, timeout)
