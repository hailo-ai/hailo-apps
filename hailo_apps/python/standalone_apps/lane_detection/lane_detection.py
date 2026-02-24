#!/usr/bin/env python3
import multiprocessing as mp
import sys
import os
from functools import partial
from pathlib import Path
import numpy as np
import cv2
import threading
import argparse
import collections
from lane_detection_utils import (UFLDProcessing, check_process_errors, compute_scaled_radius)

try:
    from hailo_apps.python.core.common.hailo_logger import get_logger, init_logging, level_from_args
    from hailo_apps.python.core.common.hailo_inference import HailoInfer
    from hailo_apps.python.core.common.core import handle_and_resolve_args
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
    from hailo_apps.python.core.common.defines import (
        MAX_INPUT_QUEUE_SIZE,
        MAX_OUTPUT_QUEUE_SIZE,
        MAX_ASYNC_INFER_JOBS
    )

APP_NAME = Path(__file__).stem
logger = get_logger(__name__)


def parser_init():    
    
    parser = argparse.ArgumentParser(description="UFLD_v2 inference",
        formatter_class=argparse.RawTextHelpFormatter
    )
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
        "--list-inputs",
        action="store_true",
        help=(
            "List available demo inputs for this application and exit. "
            "This uses the shared resources catalog (images/videos) defined in resources_config.yaml."
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

    args = parser.parse_args()    
    return args


def get_video_info(video_path):
    """
    Get the dimensions (width and height).

    Args:
        video_path (str): Path to the input video file.

    Returns:
        Tuple[int, int]: A tuple containing frame width and frame height.

    Raises:
        ValueError: If the video file cannot be opened.
    """
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        vidcap.release()
        logger.error(f"Cannot open video file {video_path}")
        raise ValueError(f"Cannot open video file {video_path}")
    frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    vidcap.release()
    return frame_width, frame_height, frame_count


def preprocess_input(video_path: str,
                     input_queue: mp.Queue, width: int, height: int,
                     ufld_processing: UFLDProcessing) -> None:
    """
    Read video frames, preprocess them, and put them into the input queue for inference.

    Args:
        video_path (str): Path to the input video.
        input_queue (mp.Queue): Queue for input frames.
        width (int): Input frame width for resizing.
        height (int): Input frame height for resizing.
        ufld_processing (UFLDProcessing): Lane detection preprocessing class.
    """
    vidcap = cv2.VideoCapture(video_path)
    success, frame = vidcap.read()

    while success:
        resized_frame = ufld_processing.resize(frame, height, width)
        input_queue.put(([frame], [resized_frame]))


        success, frame = vidcap.read()

    input_queue.put(None)  # Sentinel value to signal the end of processing


def postprocess_output(output_queue: mp.Queue,
                       output_dir: str,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
                       ufld_processing: UFLDProcessing) -> None:
    """
    Post-process inference results, draw lane detections, and write output to a video.

    Args:
        output_queue (mp.Queue): Queue for output results.
        output_dir (str): Path to the output video file.
        ufld_processing (UFLDProcessing): Lane detection post-processing class.
    """
    # Import tqdm here to avoid issues with multiprocessing
    from tqdm import tqdm

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width, height = ufld_processing.get_original_frame_size()

    out_path = os.path.join(output_dir, "output.mp4")
    output_video = cv2.VideoWriter(out_path, fourcc, 20, (width, height))

    # Compute the scaled radius for the lane detection points
    radius = compute_scaled_radius(width, height)

    pbar = tqdm(total=total_frames, desc="Processing frames")

    while True:
        result = output_queue.get()
        if result is None:
            break  # Exit when the sentinel value is received
        original_frame, inference_output = result
        slices = list(inference_output.values())
        output_tensor = np.concatenate(slices, axis=1)  # Shape: (1, total_features)
        lanes = ufld_processing.get_coordinates(output_tensor)


        for lane in lanes:
            for coord in lane:
                cv2.circle(original_frame, coord, radius, (0, 255, 0), -1)
        output_video.write(original_frame.astype('uint8'))
        pbar.update(1)

    pbar.close()
    output_video.release()
    
    # Convert to H.264 for better compatibility
    import subprocess
    logger.info("Converting video to H.264 format...")
    temp_path = out_path.replace('.mp4', '_temp.mp4')
    try:
        result = subprocess.run([
            'ffmpeg', '-y', '-loglevel', 'error', '-i', out_path, 
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
            temp_path
        ], check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        os.replace(temp_path, out_path)
        logger.info("Video conversion complete!")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to convert video to H.264")
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except FileNotFoundError:
        logger.warning("ffmpeg not found, keeping original mp4v format")


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
    # Limit number of concurrent async inferences
    pending_jobs = collections.deque()

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


        while len(pending_jobs) >= MAX_ASYNC_INFER_JOBS:
            pending_jobs.popleft().wait(10000)

        # Run async inference
        job = hailo_inference.run(preprocessed_batch, inference_callback_fn)
        pending_jobs.append(job)

    # Release resources and context
    hailo_inference.close()
    output_queue.put(None)



def run_inference_pipeline(
    video_path: str,
    net_path: str,
    batch_size: int,
    output_dir: str,
    ufld_processing: UFLDProcessing
) -> None:
    """
    Run lane detection inference using HailoAsyncInference and manage the video processing pipeline.

    Args:
        video_path (str): Path to the input video.
        net_path (str): Path to the HEF model file.
        batch_size (int): Number of frames per batch.
        output_dir (str): Path to save the output video.
        ufld_processing (UFLDProcessing): Lane detection processing class.
    """

    input_queue = mp.Queue(MAX_INPUT_QUEUE_SIZE)
    output_queue = mp.Queue(MAX_OUTPUT_QUEUE_SIZE)
    hailo_inference = HailoInfer(net_path, batch_size, output_type="FLOAT32")


    preprocessed_frame_height, preprocessed_frame_width, _ = hailo_inference.get_input_shape()
    preprocess_thread = threading.Thread(
        target=preprocess_input,
        args=(video_path,
              input_queue,
              preprocessed_frame_width,
              preprocessed_frame_height,
              ufld_processing)
    )
    postprocess_thread = threading.Thread(
        target=postprocess_output,
        args=(output_queue, output_dir, ufld_processing)
    )

    infer_thread = threading.Thread(
        target=infer, args=(hailo_inference, input_queue, output_queue)
    )

    preprocess_thread.start()
    postprocess_thread.start()
    infer_thread.start()

    infer_thread.join()
    preprocess_thread.join()
    postprocess_thread.join()

    logger.success(f"Inference was successful! Results saved in {output_dir}")



if __name__ == "__main__":

    # Parse command-line arguments
    args = parser_init()
    init_logging(level=level_from_args(args))
    handle_and_resolve_args(args, APP_NAME)
    print(args.input)
    try:
        original_frame_width, original_frame_height, total_frames = get_video_info(args.input)
    except ValueError as e:
        logger.error(e)

    ufld_processing = UFLDProcessing(
        num_cell_row=100,
        num_cell_col=100,
        num_row=56,
        num_col=41,
        num_lanes=4,
        crop_ratio=0.8,
        original_frame_width=original_frame_width,
        original_frame_height=original_frame_height,
        total_frames=total_frames,
    )

    run_inference_pipeline(
        args.input,
        args.hef_path,
        batch_size=1,
        output_dir=args.output_dir,
        ufld_processing=ufld_processing,
    )
