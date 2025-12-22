
"""Shared CLI argument parsing utilities."""

from __future__ import annotations

import argparse

try:
    from hailo_apps.python.core.common.hailo_logger import add_logging_cli_args, get_logger
except ImportError:
    from .hailo_logger import add_logging_cli_args, get_logger

hailo_logger = get_logger(__name__)


def get_base_parser() -> argparse.ArgumentParser:
    """
    Create the base argument parser with core flags shared by all Hailo applications.

    This parser defines the standard interface for common functionality across
    all applications, ensuring consistent flag naming and behavior.
    """
    hailo_logger.debug("Creating base argparse parser.")
    parser = argparse.ArgumentParser(
        description="Hailo Application Base Parser",
        add_help=False,  # Allow parent parsers to control help display
    )

    # Logging configuration group
    log_group = parser.add_argument_group("logging options", "Configure logging behavior")
    add_logging_cli_args(log_group)

    # Core input/output flags
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help=(
            "Input source for processing. Can be a file path (image or video), "
            "camera index (integer), folder path containing images, or RTSP URL. "
            "For USB cameras, use 'usb' to auto-detect or '/dev/video<X>' for a specific device. "
            "For Raspberry Pi camera, use 'rpi'. If not specified, defaults to application-specific source."
        ),
    )

    parser.add_argument(
        "--hef-path",
        "-n",
        type=str,
        default=None,
        help=(
            "Path or name of Hailo Executable Format (HEF) model file. "
            "Can be: (1) full path to .hef file, (2) model name (will search in resources), "
            "or (3) model name from available models (will auto-download if not found). "
            "If not specified, uses the default model for this application."
        ),
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help=(
            "List all available models for this application and exit. "
            "Shows default and extra models that can be used with --hef-path."
        ),
    )

    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=1,
        help=(
            "Number of frames or images to process in parallel during inference. "
            "Higher batch sizes can improve throughput but require more memory. "
            "Default is 1 (sequential processing)."
        ),
    )

    parser.add_argument(
        "--width",
        "-W",
        type=int,
        default=None,
        help=(
            "Custom output width in pixels for video or image output. "
            "If specified, the output will be resized to this width while maintaining aspect ratio. "
            "If not specified, uses the input resolution or model default."
        ),
    )

    parser.add_argument(
        "--height",
        "-H",
        type=int,
        default=None,
        help=(
            "Custom output height in pixels for video or image output. "
            "If specified, the output will be resized to this height while maintaining aspect ratio. "
            "If not specified, uses the input resolution or model default."
        ),
    )

    parser.add_argument(
        "--arch",
        "-a",
        type=str,
        default=None,
        choices=["hailo8", "hailo8l", "hailo10h"],
        help=(
            "Target Hailo architecture for model execution. "
            "Options: 'hailo8' (Hailo-8 processor), 'hailo8l' (Hailo-8L processor), "
            "'hailo10h' (Hailo-10H processor). "
            "If not specified, the architecture will be auto-detected from the connected device."
        ),
    )

    parser.add_argument(
        "--show-fps",
        action="store_true",
        help=(
            "Enable FPS (frames per second) counter display. "
            "When enabled, the application will display real-time performance metrics "
            "showing the current processing rate. Useful for performance monitoring and optimization."
        ),
    )

    parser.add_argument(
        "--frame-rate",
        "-f",
        type=int,
        default=30,
        help=(
            "Target frame rate for video processing in frames per second. "
            "Controls the playback speed and processing rate for video sources. "
            "Default is 30 FPS. Lower values reduce processing load, higher values increase throughput."
        ),
    )

    parser.add_argument(
        "--labels",
        "-l",
        type=str,
        default=None,
        help=(
            "Path to a text file containing class labels, one per line. "
            "Used for mapping model output indices to human-readable class names. "
            "If not specified, default labels for the model will be used (e.g., COCO labels for detection models)."
        ),
    )

    return parser


def get_pipeline_parser() -> argparse.ArgumentParser:
    """
    Create an argument parser for GStreamer pipeline applications.

    This parser extends the base parser with pipeline-specific flags for
    GStreamer-based applications that process video streams in real-time.
    """
    hailo_logger.debug("Creating pipeline argparse parser.")
    base_parser = get_base_parser()
    parser = argparse.ArgumentParser(
        description="Hailo GStreamer Pipeline Application",
        parents=[base_parser],
        add_help=True,  # Enable --help flag to show all available options
    )

    parser.add_argument(
        "--use-frame",
        action="store_true",
        help=(
            "Enable frame access in callback functions. "
            "When enabled, the callback function receives access to the raw frame data, "
            "allowing for custom processing, analysis, or visualization within the pipeline. "
            "Useful for applications that need to perform additional operations on individual frames."
        ),
    )

    parser.add_argument(
        "--disable-sync",
        action="store_true",
        help=(
            "Disable display sink synchronization. "
            "When enabled, the pipeline will process frames as fast as possible without waiting "
            "for display synchronization. This is particularly useful when processing from file sources "
            "where you want maximum throughput rather than real-time playback speed."
        ),
    )

    parser.add_argument(
        "--disable-callback",
        action="store_true",
        help=(
            "Skip user callback execution. "
            "When enabled, the pipeline will run without invoking custom callback functions, "
            "processing frames through the standard pipeline only. Useful for performance testing "
            "or when you want to run the pipeline without custom post-processing logic."
        ),
    )

    parser.add_argument(
        "--dump-dot",
        action="store_true",
        help=(
            "Export pipeline graph to DOT file. "
            "When enabled, the GStreamer pipeline structure will be saved as a Graphviz DOT file "
            "(typically named 'pipeline.dot'). This file can be visualized using tools like 'dot' "
            "to understand the pipeline topology and debug pipeline configuration issues."
        ),
    )

    parser.add_argument(
        "--enable-watchdog",
        action="store_true",
        help=(
            "Enable pipeline watchdog. "
            "When enabled, the pipeline will be monitored for stalled frame processing. "
            "If no frames are processed for the configured timeout, the pipeline will be automatically "
            "rebuilt. Note: This requires the application callback to be enabled (i.e., not disabled via --disable-callback)."
        ),
    )

    return parser


def get_standalone_parser() -> argparse.ArgumentParser:
    """
    Create an argument parser for standalone processing applications.

    This parser extends the base parser with standalone-specific flags for
    applications that process files or batches without GStreamer pipelines.
    """
    hailo_logger.debug("Creating standalone argparse parser.")
    base_parser = get_base_parser()
    parser = argparse.ArgumentParser(
        description="Hailo Standalone Processing Application",
        parents=[base_parser],
        add_help=True,  # Enable --help flag to show all available options
    )

    parser.add_argument(
        "--track",
        action="store_true",
        help=(
            "Enable object tracking for detections. "
            "When enabled, detected objects will be tracked across frames using a tracking algorithm "
            "(e.g., ByteTrack). This assigns consistent IDs to objects over time, enabling temporal analysis, "
            "trajectory visualization, and multi-frame association. Useful for video processing applications."
        ),
    )

    parser.add_argument(
        "--list-inputs",
        action="store_true",
        help=(
            "List available demo inputs for this application and exit. "
            "This uses the shared resources catalog (images/videos) defined in resources_config.yaml."
        ),
    )

    parser.add_argument(
        "--list-nets",
        action="store_true",
        help=(
            "List available models for this application and exit. "
            "Alias for --list-models to align with legacy app flags."
        ),
    )

    parser.add_argument(
        "--resolution",
        "-r",
        type=str,
        choices=["sd", "hd", "fhd"],
        default="sd",
        help=(
            "Predefined resolution for camera input sources. "
            "Options: 'sd' (640x480, Standard Definition), 'hd' (1280x720, High Definition), "
            "'fhd' (1920x1080, Full High Definition). "
            "Default is 'sd'. This flag is only applicable when using camera input sources."
        ),
    )



    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help=(
            "Directory where output files will be saved. "
            "When --save-output is enabled, processed images, videos, or result files will be "
            "written to this directory. If not specified, outputs are saved to a default location "
            "or the current working directory. The directory will be created if it does not exist."
        ),
    )

    parser.add_argument(
        "--save-output",
        "-s",
        action="store_true",
        help=(
            "Enable output file saving. When enabled, processed images or videos will be saved to disk. "
            "The output location is determined by the --output-dir flag. Without this flag, output is only displayed (if applicable)."
        ),
    )

    return parser


def get_default_parser() -> argparse.ArgumentParser:
    """
    Legacy function for backward compatibility.

    Returns the pipeline parser as the default to maintain compatibility
    with existing code that uses get_default_parser().
    """
    import warnings

    warnings.warn(
        "get_default_parser() is deprecated. Use get_pipeline_parser() or get_standalone_parser() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_pipeline_parser()


def configure_multi_model_hef_path(parser: argparse.ArgumentParser) -> None:
    """
    Configure --hef-path argument for multi-model apps.
    
    Multi-model apps (face_recognition, reid, paddle_ocr, clip) require multiple
    HEF files. This function modifies the --hef-path argument to accept multiple
    values using action='append'.
    
    Usage:
        --hef-path model1 --hef-path model2
    
    Args:
        parser: The argument parser to modify
    """
    # Find and remove the existing --hef-path argument
    for action in parser._actions[:]:
        if hasattr(action, 'option_strings') and '--hef-path' in action.option_strings:
            parser._remove_action(action)
            # Also remove from _option_string_actions
            for opt in action.option_strings:
                if opt in parser._option_string_actions:
                    del parser._option_string_actions[opt]
            break
    
    # Add the multi-model version
    parser.add_argument(
        "--hef-path",
        "-n",
        action="append",
        default=None,
        help=(
            "HEF model name or path (repeat for multi-model apps). "
            "Can be: (1) full path to .hef file, (2) model name from resources, "
            "or (3) model name from available models (will auto-download if not found). "
            "For multi-model apps, repeat the flag: --hef-path model1 --hef-path model2. "
            "If not specified, uses the default models for this application."
        ),
    )
