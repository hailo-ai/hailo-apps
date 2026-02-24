#!/usr/bin/env python3
import os
import sys
import multiprocessing as mp
from queue import Queue
from functools import partial
import numpy as np
import threading
from pathlib import Path
from pose_estimation_utils import PoseEstPostProcessing
import collections
try:
    from hailo_apps.python.core.common.hailo_logger import get_logger, init_logging, level_from_args
    from hailo_apps.python.core.common.hailo_inference import HailoInfer
    from hailo_apps.python.core.common.core import handle_and_resolve_args
    from hailo_apps.python.core.common.parser import get_standalone_parser
    from hailo_apps.python.core.common.toolbox import (
        init_input_source,
        preprocess,
        visualize,
        select_cap_processing_mode,
        FrameRateTracker,
    )
    from hailo_apps.python.core.common.defines import (
        MAX_INPUT_QUEUE_SIZE,
        MAX_OUTPUT_QUEUE_SIZE,
        MAX_ASYNC_INFER_JOBS
    )
except ImportError:
    repo_root = None
    for p in Path(__file__).resolve().parents:
        if (p / "hailo_apps" / "config" / "config_manager.py").exists():
            repo_root = p
            break
    if repo_root is not None:
        sys.path.insert(0, str(repo_root))

    from hailo_apps.python.core.common.hailo_logger import get_logger, init_logging, level_from_args
    from hailo_apps.python.core.common.hailo_inference import HailoInfer
    from hailo_apps.python.core.common.core import handle_and_resolve_args
    from hailo_apps.python.core.common.parser import get_standalone_parser
    from hailo_apps.python.core.common.toolbox import (
        init_input_source,
        preprocess,
        visualize,
        select_cap_processing_mode,
        FrameRateTracker,
    )
    from hailo_apps.python.core.common.defines import (
        MAX_INPUT_QUEUE_SIZE,
        MAX_OUTPUT_QUEUE_SIZE,
        MAX_ASYNC_INFER_JOBS
    )

APP_NAME = Path(__file__).stem
logger = get_logger(__name__)


def parse_args():
    """
    Initialize argument parser for the script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = get_standalone_parser()
    parser.description = "Pose estimation using Hailo inference with YOLOv8-pose models."

    # App-specific arguments
    parser.add_argument(
        "--class-num",
        "-cn",
        type=int,
        default=1,
        help="The number of classes the model is trained on. Defaults to 1.",
    )

    args = parser.parse_args()
    return args


def inference_callback(
        completion_info,
        bindings_list: list,
        input_batch: list,
        output_queue: mp.Queue
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


def run_inference_pipeline(
    net_path: str,
    input_src: str,
    batch_size: int,
    class_num: int,
    output_dir: str,
    camera_resolution: str,
    output_resolution: str,
    frame_rate: float,
    save_output: bool,
    show_fps: bool,
) -> None:
    """
    Run the inference pipeline using HailoInfer.

    Args:
        net_path (str): Path to the HEF model file.
        input_src (str): Path to the input source (image, video, folder, or camera).
        batch_size (int): Number of frames to process per batch.
        class_num (int): Number of output classes expected by the model.
        output_dir (str): Directory where processed output will be saved.
        camera_resolution (str): Camera only, input resolution (e.g., 'sd', 'hd', 'fhd').
        output_resolution (str): Output resolution for display/saving.
        frame_rate (float): Target frame rate for processing.
        save_output (bool): If True, saves the output stream as a video file.
        show_fps (bool): If True, display real-time FPS on the output.

    Returns:
        None
    """
    input_queue = Queue(MAX_INPUT_QUEUE_SIZE)
    output_queue = Queue(MAX_OUTPUT_QUEUE_SIZE)


    pose_post_processing = PoseEstPostProcessing(
        max_detections=300,
        score_threshold=0.001,
        nms_iou_thresh=0.7,
        regression_length=15,
        strides=[8, 16, 32]
    )

    # Initialize input source from string: "camera", video file, or image folder.
    cap, images, input_type = init_input_source(input_src, batch_size, camera_resolution)
    cap_processing_mode = None
    if cap is not None:
        cap_processing_mode = select_cap_processing_mode(input_type, save_output, frame_rate)

    stop_event = threading.Event()
    fps_tracker = None
    if show_fps:
        fps_tracker = FrameRateTracker()

    hailo_inference = HailoInfer(
        net_path, batch_size, output_type="FLOAT32")
    height, width, _ = hailo_inference.get_input_shape()

    post_process_callback_fn = partial(
        pose_post_processing.inference_result_handler,
        model_height=height,
        model_width=width,
        class_num = class_num
    )

    preprocess_thread = threading.Thread(
        target=preprocess,
        args=(images, cap, frame_rate, batch_size, input_queue, width, height, cap_processing_mode, None, stop_event)
    )

    postprocess_thread = threading.Thread(
        target=visualize,
        args=(output_queue, cap, save_output,
            output_dir, post_process_callback_fn, fps_tracker, output_resolution, frame_rate, False, stop_event)
        )

    infer_thread = threading.Thread(
        target=infer,
        args=(hailo_inference, input_queue, output_queue, stop_event)
    )

    infer_thread.start()
    preprocess_thread.start()
    postprocess_thread.start()

    if show_fps:
        fps_tracker.start()
    infer_thread.join()
    preprocess_thread.join()
    postprocess_thread.join()

    if show_fps:
        logger.info(fps_tracker.frame_rate_summary())

    logger.success("Inference was successful!")
    if save_output or input_src.lower() not in ("usb", "rpi"):
        logger.info(f"Results have been saved in {output_dir}")


def main() -> None:
    args = parse_args()
    init_logging(level=level_from_args(args))
    handle_and_resolve_args(args, APP_NAME)
    run_inference_pipeline(
        args.hef_path,
        args.input,
        args.batch_size,
        args.class_num,
        args.output_dir,
        args.camera_resolution,
        args.output_resolution,
        args.frame_rate,
        args.save_output,
        args.show_fps,
    )


if __name__ == "__main__":
    main()
