#!/usr/bin/env python3
import os
import sys
import queue
import threading
from functools import partial
from types import SimpleNamespace
import numpy as np
from pathlib import Path
import collections
try:
    from hailo_apps.python.core.tracker.byte_tracker import BYTETracker
    from hailo_apps.python.core.common.hailo_inference import HailoInfer
    from hailo_apps.python.core.common.toolbox import (
        InputContext,
        VisualizationSettings,
        init_input_source,
        get_labels,
        load_json_file,
        preprocess,
        visualize,
        FrameRateTracker,
        resolve_onnx_config_from_hef
    )
    from hailo_apps.python.core.common.defines import (
        MAX_INPUT_QUEUE_SIZE,
        MAX_OUTPUT_QUEUE_SIZE,
        MAX_ASYNC_INFER_JOBS
    )
    from hailo_apps.python.core.common.parser import get_standalone_parser
    from hailo_apps.python.core.common.hailo_logger import get_logger, init_logging, level_from_args
    from hailo_apps.python.standalone_apps.yolo26.object_detection.object_detection_utils import inference_result_handler
    from hailo_apps.python.core.common.core import handle_and_resolve_args
    from hailo_apps.python.core.common.onnx_utils import (
        load_onnx_config,
        init_onnx_sessions,
        normalized_preprocess,
        infer_debug_ref_onnx,
    )
except ImportError:
    # Running as a plain script: add repo root so `import hailo_apps` works.
    repo_root = None
    for p in Path(__file__).resolve().parents:
        if (p / "hailo_apps" / "config" / "config_manager.py").exists():
            repo_root = p
            break
    if repo_root is not None:
        sys.path.insert(0, str(repo_root))

    from hailo_apps.python.core.tracker.byte_tracker import BYTETracker
    from hailo_apps.python.core.common.hailo_inference import HailoInfer
    from hailo_apps.python.core.common.core import handle_and_resolve_args
    from hailo_apps.python.core.common.toolbox import (
        InputContext,
        VisualizationSettings,
        init_input_source,
        get_labels,
        load_json_file,
        preprocess,
        visualize,
        FrameRateTracker,
        resolve_onnx_config_from_hef
    )
    from hailo_apps.python.core.common.defines import (
        MAX_INPUT_QUEUE_SIZE,
        MAX_OUTPUT_QUEUE_SIZE,
        MAX_ASYNC_INFER_JOBS
    )
    from hailo_apps.python.core.common.parser import get_standalone_parser
    from hailo_apps.python.core.common.hailo_logger import get_logger, init_logging, level_from_args
    from hailo_apps.python.standalone_apps.yolo26.object_detection.object_detection_utils import inference_result_handler
    from hailo_apps.python.core.common.onnx_utils import (
        load_onnx_config,
        init_onnx_sessions,
        normalized_preprocess,
        infer_debug_ref_onnx,
    )

APP_NAME = Path(__file__).stem
logger = get_logger(__name__)


def parse_args():
    """
    Parse command-line arguments for the detection application.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = get_standalone_parser()
    parser.description = "YOLO26 object detection with ONNX postprocessing."

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

    parser.add_argument(
        "--draw-trail",
        action="store_true",
        help=(
            "[Tracking only] Draw motion trails of tracked objects.\n"
            "Uses the last 30 positions from the tracker history."
        )
    )

    parser.add_argument(
        "--onnx",
        type=str,
        default=None,
        metavar="ONNX_PP_FILE",
        help=(
            "Optional override path to ONNX postprocessing model file. "
            "If omitted, standalone resolver uses model-sidecar placement near the HEF path."
        ),
    )


    parser.add_argument(
        "--onnx-config",
        type=str,
        default=None,
        metavar="ONNX_CONFIG_FILE",
        help=(
            "Optional override path to ONNX postprocessing config JSON file. "
            "If omitted, the config is resolved automatically based on the HEF model name "
            "(onnx/config_onnx_<model_name>.json)."
        ),
    )

    parser.add_argument(
        "--neural-onnx-ref",
        type=str,
        default=None,
        help=(
            "Debug reference mode: path to ONNX model that reproduces HEF-like output tensors. "
            "When set, HEF inference is bypassed and this ONNX model feeds the postprocessing ONNX. "
            "If --hef-path is omitted, provide both --onnxconfig and --onnx explicitly."
        ),
    )

    args = parser.parse_args()
    return args


def run_inference_pipeline(net, input_src, batch_size, labels, output_dir,  
          save_output=False, camera_resolution="sd", output_resolution=None,
          enable_tracking=False, show_fps=False, frame_rate=None, draw_trail=False,
          no_display=False, onnx_config=None, onnxconfig_args=None) -> None:
    """
    Initialize queues, HailoAsyncInference instance, and run the inference.
    """
    labels = get_labels(labels)
    config_data = load_json_file(str(Path(__file__).parent / "config.json"))
    
    # Load ONNX config and initialize sessions if specified
    onnx_session = None
    debug_ref_onnx_intermediate_session = None
    use_debug_ref_onnx = False

    onnx_config, config_path = load_onnx_config(onnx_config, caller_file=__file__)
    use_debug_ref_onnx = bool(getattr(onnxconfig_args, "neural_onnx_ref", None))
    sessions = init_onnx_sessions(
        onnx_config,
        config_path,
        use_debug_ref_onnx,
        postproc_onnx_path=getattr(onnxconfig_args, "onnx", None),
        neural_onnx_ref_path=getattr(onnxconfig_args, "neural_onnx_ref", None),
    )
    onnx_session = sessions["onnx_session"]
    debug_ref_onnx_intermediate_session = sessions["debug_ref_onnx_intermediate_session"]

    input_context = InputContext(
        input_src=input_src,
        batch_size=batch_size,
        resolution=camera_resolution,
        frame_rate=frame_rate,
    )
    input_context = init_input_source(input_context)

    visualization_settings = VisualizationSettings(
        output_dir=output_dir,
        save_stream_output=save_output,
        output_resolution=output_resolution,
        no_display=no_display,
    )

    stop_event = threading.Event()
    tracker = None
    fps_tracker = None
    if show_fps:
        fps_tracker = FrameRateTracker()

    if enable_tracking:
        # load tracker config from config_data
        tracker_config = config_data.get("visualization_params", {}).get("tracker", {})
        tracker = BYTETracker(SimpleNamespace(**tracker_config))

    input_queue = queue.Queue(MAX_INPUT_QUEUE_SIZE)
    output_queue = queue.Queue(MAX_OUTPUT_QUEUE_SIZE)

    post_process_callback_fn = partial(
        inference_result_handler, labels=labels,
        config_data=config_data, tracker=tracker, draw_trail=draw_trail,
        onnx_config=onnx_config, onnx_session=onnx_session
    )

    # Skip HEF initialization in debug reference ONNX mode
    if use_debug_ref_onnx:
        # Use debug reference ONNX intermediate model input shape
        debug_ref_onnx_input = debug_ref_onnx_intermediate_session.get_inputs()[0]
        # Assume NCHW or NHWC format - extract H, W
        input_shape = debug_ref_onnx_input.shape
        if len(input_shape) == 4:
            # NCHW: [batch, channels, height, width] or NHWC: [batch, height, width, channels]
            if input_shape[1] in [1, 3]:  # NCHW
                height, width = input_shape[2], input_shape[3]
            else:  # NHWC
                height, width = input_shape[1], input_shape[2]
        else:
            raise ValueError(f"Unexpected debug reference ONNX input shape: {input_shape}")
        hailo_inference = None
        logger.info(f"Debug reference ONNX mode enabled - using intermediate outputs + ONNX postprocessing (input size: {height}x{width})")
    else:
        logger.info("Using HEF + ONNX postprocessing mode")
        hailo_inference = HailoInfer(net, batch_size, input_type=None, output_type="FLOAT32")
        height, width, _ = hailo_inference.get_input_shape()

    preprocess_thread = threading.Thread(
        target=preprocess,
        args=(
            input_context,
            input_queue,
            width,
            height,
            normalized_preprocess if use_debug_ref_onnx else None,
            stop_event,
        )
    )
    postprocess_thread = threading.Thread(
        target=visualize,
        args=(
            input_context,
            visualization_settings,
            output_queue,
            post_process_callback_fn,
            fps_tracker,
            stop_event,
        ),
    )
    
    if use_debug_ref_onnx:
        # Use debug reference ONNX inference (intermediate model + postprocessing)
        infer_thread = threading.Thread(
            target=infer_debug_ref_onnx, args=(debug_ref_onnx_intermediate_session, onnx_session, onnx_config, input_queue, output_queue)
        )
    else:
        infer_thread = threading.Thread(
            target=infer, args=(hailo_inference, input_queue, output_queue, stop_event)
        )

    preprocess_thread.start()
    postprocess_thread.start()
    infer_thread.start()

    if show_fps:
        fps_tracker.start()

    try:
        preprocess_thread.join()
        infer_thread.join()
        postprocess_thread.join()

    except KeyboardInterrupt:
        logger.info("Interrupted (Ctrl+C). Shutting down...")
        stop_event.set()

    finally:
        if show_fps:
            logger.info(fps_tracker.frame_rate_summary())

        logger.success("Processing completed successfully.")

        if visualization_settings.save_stream_output or input_context.has_images:
            logger.info(f"Saved outputs to '{visualization_settings.output_dir}'.")


def infer(hailo_inference, input_queue, output_queue, stop_event):
    """
    Main inference loop that pulls data from the input queue, runs asynchronous
    inference, and pushes results to the output queue.

    Each item in the input queue is expected to be a tuple:
        (input_batch, preprocessed_batch)
        - input_batch: Original frames (used for visualization or tracking)
        - preprocessed_batch: Model-ready frames (e.g., resized, normalized)

    Args:
        hailo_inference (HailoInfer): The inference engine to run model predictions.
        input_queue (queue.Queue): Provides (input_batch, preprocessed_batch) tuples.
        output_queue (queue.Queue): Collects (input_frame, result) tuples for visualization.

    Returns:
        None
    """
    # Limit number of concurrent async inferences
    pending_jobs = collections.deque()

    while True:
        next_batch = input_queue.get()
        if not next_batch:
            break  # Stop signal received

        if stop_event.is_set():
            continue  # Skip processing if stop signal is set

        input_batch, preprocessed_batch = next_batch

        # Prepare the callback for handling the inference result
        inference_callback_fn = partial(
            inference_callback,
            input_batch=input_batch,
            output_queue=output_queue
        )


        while len(pending_jobs) >= MAX_ASYNC_INFER_JOBS:
            pending_jobs.popleft().wait(10000)

        # Run async inference
        job = hailo_inference.run(preprocessed_batch, inference_callback_fn)
        pending_jobs.append(job)

    # Release resources and context
    hailo_inference.close()
    output_queue.put(None)


def inference_callback(
    completion_info,
    bindings_list: list,
    input_batch: list,
    output_queue: queue.Queue
) -> None:
    """
    infernce callback to handle inference results and push them to a queue.

    Args:
        completion_info: Hailo inference completion info.
        bindings_list (list): Output bindings for each inference.
        input_batch (list): Original input frames.
        output_queue (queue.Queue): Queue to push output results to.
    """
    if completion_info.exception:
        logger.error(f'Inference error: {completion_info.exception}')
    else:
        for i, bindings in enumerate(bindings_list):
            if len(bindings._output_names) == 1:
                result = bindings.output().get_buffer()
            else:
                result = {
                    name: np.expand_dims(
                        bindings.output(name).get_buffer(), axis=0
                    )
                    for name in bindings._output_names
                }
            output_queue.put((input_batch[i], result))

def main() -> None:
    """
    Main function to run the script.
    """
    args = parse_args()
    init_logging(level=level_from_args(args))
    handle_and_resolve_args(args, APP_NAME, using_onnx_pp=True)

    if args.onnx_config is None:
        args.onnx_config = resolve_onnx_config_from_hef(
            args.hef_path,
            __file__
        )

    run_inference_pipeline(
        args.hef_path,
        args.input,
        args.batch_size,
        args.labels,
        args.output_dir,
        args.save_output,
        args.camera_resolution,
        args.output_resolution,
        args.track,
        args.show_fps,
        args.frame_rate,
        args.draw_trail,
        args.no_display,
        args.onnx_config,
        args  # Pass full args for --neural-onnx-ref debug mode
    )


if __name__ == "__main__":
    main()
