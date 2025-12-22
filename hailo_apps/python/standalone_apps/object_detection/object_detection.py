#!/usr/bin/env python3
import os
import sys
import queue
import threading
from functools import partial
from types import SimpleNamespace
import numpy as np
from pathlib import Path

try:
    from hailo_apps.python.core.tracker.byte_tracker import BYTETracker
    from hailo_apps.python.core.common.hailo_inference import HailoInfer
    from hailo_apps.python.core.common.toolbox import (
        init_input_source,
        get_labels,
        load_json_file,
        preprocess,
        visualize,
        FrameRateTracker,
        resolve_arch,
        resolve_input_arg,
        list_inputs,
    )
    from hailo_apps.python.core.common.core import handle_list_models_flag, resolve_hef_path
    from hailo_apps.python.core.common.parser import get_standalone_parser
    from hailo_apps.python.core.common.hailo_logger import get_logger, init_logging, level_from_args
    from hailo_apps.python.standalone_apps.object_detection.object_detection_post_process import inference_result_handler
except ImportError:
    core_dir = Path(__file__).resolve().parents[2] / "core"
    sys.path.insert(0, str(core_dir))
    from tracker.byte_tracker import BYTETracker
    from common.hailo_inference import HailoInfer
    from common.toolbox import (
        init_input_source,
        get_labels,
        load_json_file,
        preprocess,
        visualize,
        FrameRateTracker,
        resolve_arch,
        resolve_input_arg,
        list_inputs,
    )
    from common.core import handle_list_models_flag, resolve_hef_path
    from common.parser import get_standalone_parser
    from common.hailo_logger import get_logger, init_logging, level_from_args
    from object_detection_post_process import inference_result_handler

APP_NAME = Path(__file__).stem
logger = get_logger(__name__)


def parse_args():
    """
    Parse command-line arguments for the detection application.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = get_standalone_parser()
    parser.description = "Run object detection with optional tracking and performance measurement."

    handle_list_models_flag(parser, APP_NAME)

    args = parser.parse_args()

    # Handle --list-inputs and exit
    if args.list_inputs:
        list_inputs(APP_NAME)
        sys.exit(0)

    # Resolve network and input paths
    args.arch = resolve_arch(args.arch)
    args.hef_path = resolve_hef_path(
        hef_path=args.hef_path,
        app_name=APP_NAME,
        arch=args.arch,
    )
    if args.hef_path is None:
        logger.error("Failed to resolve HEF path for %s", APP_NAME)
        sys.exit(1)
    args.input = resolve_input_arg(APP_NAME, args.input)

    # Setup output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def run_inference_pipeline(net, input, batch_size, labels, output_dir,
          save_stream_output=False, resolution="sd",
          enable_tracking=False, show_fps=False) -> None:
    """
    Initialize queues, HailoAsyncInference instance, and run the inference.
    """
    labels = get_labels(labels)
    config_data = load_json_file("config.json")

    # Initialize input source from string: "camera", video file, or image folder.
    cap, images = init_input_source(input, batch_size, resolution)
    tracker = None
    fps_tracker = None
    if show_fps:
        fps_tracker = FrameRateTracker()

    if enable_tracking:
        # load tracker config from config_data
        tracker_config = config_data.get("visualization_params", {}).get("tracker", {})
        tracker = BYTETracker(SimpleNamespace(**tracker_config))

    input_queue = queue.Queue()
    output_queue = queue.Queue()

    post_process_callback_fn = partial(
        inference_result_handler, labels=labels,
        config_data=config_data, tracker=tracker
    )

    hailo_inference = HailoInfer(net, batch_size)
    height, width, _ = hailo_inference.get_input_shape()

    preprocess_thread = threading.Thread(
        target=preprocess, args=(images, cap, None, batch_size, input_queue, width, height)
    )
    postprocess_thread = threading.Thread(
        target=visualize, args=(output_queue, cap, save_stream_output,
                                output_dir, post_process_callback_fn, fps_tracker, None, None)
    )
    infer_thread = threading.Thread(
        target=infer, args=(hailo_inference, input_queue, output_queue)
    )

    preprocess_thread.start()
    postprocess_thread.start()
    infer_thread.start()

    if show_fps:
        fps_tracker.start()

    preprocess_thread.join()
    infer_thread.join()
    output_queue.put(None)  # Signal process thread to exit
    postprocess_thread.join()

    if show_fps:
        logger.debug(fps_tracker.frame_rate_summary())

    logger.info('Inference was successful!')


def infer(hailo_inference, input_queue, output_queue):
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
    while True:
        next_batch = input_queue.get()
        if not next_batch:
            break  # Stop signal received

        input_batch, preprocessed_batch = next_batch

        # Prepare the callback for handling the inference result
        inference_callback_fn = partial(
            inference_callback,
            input_batch=input_batch,
            output_queue=output_queue
        )

        # Run async inference
        hailo_inference.run(preprocessed_batch, inference_callback_fn)

    # Release resources and context
    hailo_inference.close()


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
    run_inference_pipeline(
        args.hef_path,
        args.input,
        args.batch_size,
        args.labels,
        args.output_dir,
        args.save_output,
        args.resolution,
        args.track,
        args.show_fps
    )




if __name__ == "__main__":
    main()
