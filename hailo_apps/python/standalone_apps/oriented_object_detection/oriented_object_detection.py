#!/usr/bin/env python3

import argparse
import os
import sys
import queue
import threading
from functools import partial
import numpy as np
from pathlib import Path
import cv2
import collections

def _ensure_repo_root_on_syspath() -> None:
    """
    This allows `import hailo_apps...` to work without requiring users to
    `pip install -e .` or `source setup_env.sh`.
    """

    this_file = Path(__file__).resolve()
    for parent in this_file.parents:
        if (parent / "hailo_apps").is_dir():
            sys.path.insert(0, str(parent))
            return

_ensure_repo_root_on_syspath()

try:
    from hailo_apps.python.core.common.hailo_inference import HailoInfer
    from hailo_apps.python.core.common.core import handle_and_resolve_args
    from hailo_apps.python.core.common.toolbox import (
        init_input_source,
        get_labels,
        load_json_file,
        preprocess,
        visualize,
        select_cap_processing_mode,
        FrameRateTracker
    )
    from hailo_apps.python.core.common.defines import (
        MAX_INPUT_QUEUE_SIZE,
        MAX_OUTPUT_QUEUE_SIZE,
        MAX_ASYNC_INFER_JOBS
    )
    from hailo_apps.python.core.common.defines import REPO_ROOT
    from hailo_apps.python.core.common.parser import get_standalone_parser
    from hailo_apps.python.core.common.hailo_logger import get_logger, init_logging, level_from_args
    from oriented_object_detection_post_process import inference_result_handler

except ImportError:
    # Running as a plain script: add repo root so `import hailo_apps` works.
    repo_root = None
    for p in Path(__file__).resolve().parents:
        if (p / "hailo_apps" / "config" / "config_manager.py").exists():
            repo_root = p
            break
    if repo_root is not None:
        sys.path.insert(0, str(repo_root))
    from hailo_apps.python.core.common.hailo_inference import HailoInfer
    from hailo_apps.python.core.common.core import handle_and_resolve_args
    from hailo_apps.python.core.common.toolbox import (
        init_input_source,
        get_labels,
        load_json_file,
        preprocess,
        visualize,
        select_cap_processing_mode,
        FrameRateTracker
    )
    from hailo_apps.python.core.common.defines import (
        MAX_INPUT_QUEUE_SIZE,
        MAX_OUTPUT_QUEUE_SIZE,
        MAX_ASYNC_INFER_JOBS
    )
    from hailo_apps.python.core.common.defines import REPO_ROOT
    from hailo_apps.python.core.common.parser import get_standalone_parser
    from hailo_apps.python.core.common.hailo_logger import get_logger, init_logging, level_from_args
    from oriented_object_detection_post_process import inference_result_handler


APP_NAME = Path(__file__).stem
logger = get_logger(__name__)

def parse_args() -> argparse.Namespace:
    """
    Initialize argument parser for the script.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = get_standalone_parser()
    parser.description = "Run oriented object detection."
    parser.add_argument(
        "--labels",
        "-l",
        type=str,
        default=str(REPO_ROOT / "local_resources" / "dota.txt"),
        help=(
            "Path to a text file containing class labels, one per line. "
            "Used for mapping model output indices to human-readable class names. "
            "If not specified, default labels for the model will be used (e.g., COCO labels for detection models)."
        ),
    )

    args = parser.parse_args()
    return args


def oriented_object_detection_preprocess(image: np.ndarray, model_w: int, model_h: int, config_data: dict) -> np.ndarray:
    # run letterbox resize
    h0, w0 = image.shape[:2]
    new_w, new_h = model_w, model_h
    r = min(new_w / w0, new_h / h0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw = (new_w - new_unpad[0]) / 2
    dh = (new_h - new_unpad[1]) / 2
    
    # calculate padding to ensure exact output dimensions
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    
    # adjust padding to ensure exact output shape
    if new_unpad[1] + top + bottom != new_h:
        bottom = new_h - new_unpad[1] - top
    if new_unpad[0] + left + right != new_w:
        right = new_w - new_unpad[0] - left
    
    color = (114, 114, 114)
    resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    padded_image = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded_image


def run_inference_pipeline(
    net,
    input_src,
    batch_size,
    labels_file,
    output_dir,
    camera_resolution,
    output_resolution,
    framerate,
    save_output=False,
    show_fps=False
) -> None:

    labels = get_labels(labels_file)
    # load local config.json from this example folder
    config_path = str(Path(__file__).parent / "config.json")
    config_data = load_json_file(config_path)

    # Initialize input source from string: "camera", video file, or image folder.
    cap, images, input_type = init_input_source(input_src, batch_size, camera_resolution)
    cap_processing_mode = None
    if cap is not None:
        cap_processing_mode = select_cap_processing_mode(input_type, save_output, framerate)

    stop_event = threading.Event()
    fps_tracker = None
    if show_fps:
        fps_tracker = FrameRateTracker()

    input_queue = queue.Queue(MAX_INPUT_QUEUE_SIZE)
    output_queue = queue.Queue(MAX_OUTPUT_QUEUE_SIZE)

    preprocess_callback_fn = partial(
        oriented_object_detection_preprocess,
        config_data=config_data,
    )
    
    post_process_callback_fn = partial(
        inference_result_handler, 
        labels=labels,
        config_data=config_data,
    )
    hailo_inference = HailoInfer(net, batch_size, input_type="UINT8", output_type="FLOAT32")
    height, width, _ = hailo_inference.get_input_shape()

    preprocess_thread = threading.Thread(
        target=preprocess, args=(images, cap, framerate, batch_size, input_queue, width, height, cap_processing_mode, preprocess_callback_fn, stop_event))
    postprocess_thread = threading.Thread(
        target=visualize, args=(output_queue, cap, save_output,
                                output_dir, post_process_callback_fn, fps_tracker, output_resolution, framerate, False, stop_event)
    )
    infer_thread = threading.Thread(
        target=infer, args=(hailo_inference, input_queue, output_queue, stop_event)
    )

    preprocess_thread.start()
    postprocess_thread.start()
    infer_thread.start()

    if show_fps:
        fps_tracker.start()

    preprocess_thread.join()
    infer_thread.join()
    postprocess_thread.join()

    if show_fps:
        logger.info(fps_tracker.frame_rate_summary())

    logger.success("Inference was successful!")
    if save_output or input_src.lower() not in ("usb", "rpi"):
        logger.info(f"Results have been saved in {output_dir}")



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
    max_async_jobs = 20
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
    args = parse_args()
    init_logging(level=level_from_args(args))
    handle_and_resolve_args(args, APP_NAME)
    run_inference_pipeline(
        args.hef_path,
        args.input,
        args.batch_size,
        args.labels,
        args.output_dir,
        args.camera_resolution,
        args.output_resolution,
        args.frame_rate,
        args.save_output,
        args.show_fps
    )


if __name__ == "__main__":
    main()
