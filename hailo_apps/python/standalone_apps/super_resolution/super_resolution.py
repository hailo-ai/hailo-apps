#!/usr/bin/env python3
import numpy as np
import cv2
from pathlib import Path
import os
import sys
from typing import List
import threading
import queue
from super_resolution_utils import SrganUtils, Espcnx4Utils, inference_result_handler
from functools import partial

try:
    from hailo_apps.python.core.common.hailo_inference import HailoInfer
    from hailo_apps.python.core.common.hailo_logger import get_logger, init_logging, level_from_args
    from hailo_apps.python.core.common.toolbox import (
        init_input_source,
        preprocess,
        visualize,
        FrameRateTracker,
        resolve_arch,
        resolve_input_arg,
        resolve_output_resolution_arg,
        list_inputs,
    )
    from hailo_apps.python.core.common.core import handle_list_models_flag, resolve_hef_path
    from hailo_apps.python.core.common.parser import get_standalone_parser
except ImportError:
    core_dir = Path(__file__).resolve().parents[2] / "core"
    sys.path.insert(0, str(core_dir))
    from common.hailo_inference import HailoInfer
    from common.hailo_logger import get_logger, init_logging, level_from_args
    from common.toolbox import (
        init_input_source,
        preprocess,
        visualize,
        FrameRateTracker,
        resolve_arch,
        resolve_input_arg,
        resolve_output_resolution_arg,
        list_inputs,
    )
    from common.core import handle_list_models_flag, resolve_hef_path
    from common.parser import get_standalone_parser

APP_NAME = Path(__file__).stem
logger = get_logger(__name__)


def parse_args():
    """
    Initialize argument parser for the script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = get_standalone_parser()
    parser.description = "Super Resolution using SRGAN or ESPCN models."

    # App-specific arguments
    parser.add_argument(
        "--camera-resolution",
        "-cr",
        type=str,
        choices=["sd", "hd", "fhd"],
        default=None,
        help="(Camera only) Input resolution: 'sd' (640x480), 'hd' (1280x720), or 'fhd' (1920x1080).",
    )

    parser.add_argument(
        "--output-resolution",
        "-or",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Output resolution. Use: 'sd', 'hd', 'fhd', "
            "or custom size like '--output-resolution 1920 1080'."
        ),
    )

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
    args.output_resolution = resolve_output_resolution_arg(args.output_resolution)

    # Setup output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(args.output_dir, exist_ok=True)

    return args



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



def run_inference_pipeline(
    net_path: str,
    input: str,
    batch_size: int,
    output_dir: str,
    camera_resolution: str,
    output_resolution: str,
    frame_rate: float,
    save_output: bool,
    show_fps: bool,
) -> None:
    """
    Initialize queues, create HailoAsyncInference instance, and run the inference pipeline.

    Args:
        net_path (str): Path to the HEF model file.
        input (str): Input source path (image directory, video file, or camera).
        batch_size (int): Number of frames to process per batch.
        output_dir (str): Directory path to save output visualizations.
        camera_resolution (str): Camera input resolution, e.g., 'sd', 'hd', or 'fhd'.
        output_resolution (str): Output resolution for display/saving.
        frame_rate (float): Target frame rate for processing.
        save_output (bool): Whether to save the processed stream to video/images.
        show_fps (bool): Whether to print/log FPS (frames per second) information during execution.

    Returns:
        None
    """

    utils = None
    # Initialize input source from string: "camera", video file, or image folder.
    cap, images = init_input_source(input, batch_size, camera_resolution)

    input_queue = queue.Queue()
    output_queue = queue.Queue()


    fps_tracker = None
    if show_fps:
        fps_tracker = FrameRateTracker()

    # Convert net_path to string if it's a Path object
    net_path = str(net_path)
    
    if 'espcn' in net_path:
        utils = Espcnx4Utils()
        hailo_inference = HailoInfer(net_path, batch_size, input_type="FLOAT32", output_type="FLOAT32")
    else:
        utils = SrganUtils()
        hailo_inference = HailoInfer(net_path, batch_size)
    
    height, width, _ = hailo_inference.get_input_shape()

    post_process_callback_fn = partial(
        inference_result_handler,model_height=height, model_width=width
    )

    preprocess_thread = threading.Thread(
        target=preprocess, args=(images, cap, frame_rate, batch_size, input_queue, width, height)
    )
    postprocess_thread = threading.Thread(
        target=visualize, args=(output_queue, cap, save_output,
                                output_dir, post_process_callback_fn, fps_tracker, output_resolution, frame_rate, True)
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

    logger.info("Inference was successful!")
    if save_output or input.lower() != "camera":
        logger.info(f"Results have been saved in {output_dir}")


def main() -> None:
    """
    Main function to run the script.
    """
    # Parse command line arguments
    args = parse_args()
    init_logging(level=level_from_args(args))

    # Start the inference
    run_inference_pipeline(
        args.hef_path,
        args.input,
        args.batch_size,
        args.output_dir,
        args.camera_resolution,
        args.output_resolution,
        args.frame_rate,
        args.save_output,
        args.show_fps,
    )
if __name__ == "__main__":
    main()
