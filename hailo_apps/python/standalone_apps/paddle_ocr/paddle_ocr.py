#!/usr/bin/env python3
import os
import sys
import queue
import threading
from functools import partial
from pathlib import Path

try:
    from hailo_apps.python.core.common.hailo_logger import get_logger, init_logging, level_from_args
except ImportError:
    core_dir = Path(__file__).resolve().parents[2] / "core"
    sys.path.insert(0, str(core_dir))
    from common.hailo_logger import get_logger, init_logging, level_from_args

# Check OCR dependencies before importing OCR-specific modules
def check_ocr_dependencies():
    """
    Check if all required OCR dependencies are installed.
    
    Exits the program with installation instructions if any dependencies are missing.
    """
    missing_deps = []
    # Map package names to their import names
    ocr_deps = {
        "paddlepaddle": "paddle",  # pip install paddlepaddle, but import paddle
        "shapely": "shapely",
        "pyclipper": "pyclipper",
        "symspellpy": "symspellpy",
    }
    
    for package_name, import_name in ocr_deps.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_deps.append(package_name)

    if missing_deps:
        print("\n" + "="*70)
        print("❌ MISSING REQUIRED DEPENDENCIES")
        print("="*70)
        print("\nThe following dependencies are required but not installed:")
        for dep in missing_deps:
            print(f"  • {dep}")
        print("\n" + "-"*70)
        print("INSTALLATION INSTRUCTIONS:")
        print("-"*70)
        print("\nTo install all dependencies (recommended):")
        print("  1. Navigate to the repository root directory")
        print("  2. Run: pip install -e \".[ocr]\"")
        print("\n" + "="*70)
        sys.exit(1)

# Check dependencies early
check_ocr_dependencies()
from paddle_ocr_utils import det_postprocess, resize_with_padding, inference_result_handler, OcrCorrector, map_bbox_to_original_image
import uuid
from collections import defaultdict

try:
    from hailo_apps.python.core.common.hailo_inference import HailoInfer
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
    from hailo_apps.python.core.common.core import (
        configure_multi_model_hef_path,
        handle_list_models_flag,
        resolve_hef_paths,
    )
    from hailo_apps.python.core.common.parser import get_standalone_parser
except ImportError:
    core_dir = Path(__file__).resolve().parents[2] / "core"
    sys.path.insert(0, str(core_dir))
    from common.hailo_inference import HailoInfer
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
    from common.core import (
        configure_multi_model_hef_path,
        handle_list_models_flag,
        resolve_hef_paths,
    )
    from common.parser import get_standalone_parser

APP_NAME = Path(__file__).stem
logger = get_logger(__name__)
# A dictionary that accumulates all OCR crops and their results for a single frame.
ocr_results_dict = defaultdict(lambda: {"frame": None, "results": [], "boxes": [], "count": 0})
ocr_expected_counts = {}


def parse_args():
    """
    Initialize argument parser for the script.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = get_standalone_parser()
    parser.description = "Paddle OCR Example with detection + OCR networks."
    configure_multi_model_hef_path(parser)

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

    parser.add_argument(
        "--use-corrector",
        action="store_true",
        help="Enable text correction after OCR (e.g., for spelling or formatting).",
    )

    handle_list_models_flag(parser, APP_NAME)

    args = parser.parse_args()

    # Handle --list-inputs and exit
    if args.list_inputs:
        list_inputs(APP_NAME)
        sys.exit(0)

    # Resolve the two networks
    args.arch = resolve_arch(args.arch)
    try:
        models = resolve_hef_paths(
            hef_paths=args.hef_path,
            app_name=APP_NAME,
            arch=args.arch,
        )
    except Exception as exc:
        logger.error("Failed to resolve HEF paths for %s: %s", APP_NAME, exc)
        sys.exit(1)

    args.det_net, args.ocr_net = [model.path for model in models]
    args.input = resolve_input_arg(APP_NAME, args.input)
    args.output_resolution = resolve_output_resolution_arg(args.output_resolution)

    # Setup output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def detector_hailo_infer(hailo_inference, input_queue, output_queue):
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
            detector_inference_callback,
            input_batch=input_batch,
            output_queue=output_queue
        )

        # Run async inference
        hailo_inference.run(preprocessed_batch, inference_callback_fn)

    # Release resources and context
    hailo_inference.close()



def ocr_hailo_infer(hailo_inference, input_queue, output_queue):
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

        input_batch, preprocessed_batch, extra_context = next_batch

        # Prepare the callback for handling the inference result
        inference_callback_fn = partial(
            ocr_inference_callback,
            input_batch=input_batch,
            output_queue=output_queue,
            extra_context = extra_context
        )

        # Run async inference
        hailo_inference.run(preprocessed_batch, inference_callback_fn)

    # Release resources and context
    hailo_inference.close()


def run_inference_pipeline(
    det_net,
    ocr_net,
    input,
    batch_size,
    output_dir,
    camera_resolution,
    output_resolution,
    frame_rate,
    save_output=False,
    show_fps=False,
    use_corrector=False,
) -> None:
    """
    Run full detector + OCR inference pipeline with multi-threading and streaming.

    Args:
        det_net: model path for the detection network.
        ocr_net: model path for the OCR network.
        input (str): Input source — 'camera', image directory, or video file path.
        batch_size (int): Number of frames to process in each batch.
        output_dir (str): Directory where output images or videos will be saved.
        camera_resolution (str): Camera input resolution (e.g., 'sd', 'hd', 'fhd').
        output_resolution (str): Output resolution for display/saving.
        frame_rate (float): Target frame rate for processing.
        save_output (bool): Whether to save the output stream. Defaults to False.
        show_fps (bool): Whether to display frames-per-second performance. Defaults to False.
        use_corrector (bool): Whether to enable text spell correction. Defaults to False.

    Returns:
        None
    """
    # Initialize capture handle for video/camera or load image folder
    cap, images = init_input_source(input, batch_size, camera_resolution)

    # Queues for passing data between threads
    det_input_queue = queue.Queue()
    ocr_input_queue = queue.Queue()

    det_postprocess_queue = queue.Queue()
    ocr_postprocess_queue = queue.Queue()

    vis_output_queue = queue.Queue()


    fps_tracker=None
    if show_fps:
        fps_tracker = FrameRateTracker()

    ocr_corrector = None
    if use_corrector:
        ocr_corrector = OcrCorrector()


    ####### CALLBACKS ########

    # Final visualization callback function with optional correction
    post_process_callback_fn = partial(
        inference_result_handler,
        ocr_corrector=ocr_corrector
    )


    # Detector inference callbacks
    detector_inference_callback_fn = partial(
        detector_inference_callback,
        det_postprocess_queue=det_postprocess_queue,
    )

    # ocr inference callbacks
    ocr_inference_callback_fn = partial(
        ocr_inference_callback,
        ocr_postprocess_queue=ocr_postprocess_queue
    )


    ###### THREADS ########

    # Start detector with async Hailo inference
    detector_hailo_inference = HailoInfer(det_net, batch_size)

    # Start ocr with async Hailo inference
    ocr_hailo_inference = HailoInfer(ocr_net, batch_size, priority=1)

    height, width, _ = detector_hailo_inference.get_input_shape()

    # input postprocess
    preprocess_thread = threading.Thread(
        target=preprocess, args=(images, cap, frame_rate, batch_size, det_input_queue, width, height)
    )

    # detector output postprocess
    detection_postprocess_thread = threading.Thread(
        target=detection_postprocess,
        args=(det_postprocess_queue, ocr_input_queue, vis_output_queue, height, width),
    )

    # ocr output postprocess
    ocr_postprocess_thread = threading.Thread(
        target=ocr_postprocess,
        args=(ocr_postprocess_queue, vis_output_queue),
    )

    # visualisation postprocess
    vis_postprocess_thread = threading.Thread(
        target=visualize,
        args=(vis_output_queue, cap, save_output, output_dir,
              post_process_callback_fn, fps_tracker, output_resolution, frame_rate, True)
    )

    det_thread = threading.Thread(
        target=detector_hailo_infer, args=(detector_hailo_inference, det_input_queue, det_postprocess_queue)
    )

    ocr_thread = threading.Thread(
        target=ocr_hailo_infer, args=(ocr_hailo_inference, ocr_input_queue, ocr_postprocess_queue)
    )

    if show_fps:
        fps_tracker.start()

    ##### Start threads ######
    preprocess_thread.start()
    det_thread.start()
    detection_postprocess_thread.start()
    ocr_thread.start()
    ocr_postprocess_thread.start()
    vis_postprocess_thread.start()


    ##### Join Threads and Shutdown Queues ######

    # Wait for input preprocessing to finish
    preprocess_thread.join()

    # Wait for detector inference to finish
    det_thread.join()

    # Tell detection postprocess thread to exit
    det_postprocess_queue.put(None)
    detection_postprocess_thread.join()

    # Signal OCR inference thread to stop (no more crops coming)
    ocr_input_queue.put(None)
    ocr_thread.join()

    # Signal OCR postprocess thread to stop
    ocr_postprocess_queue.put(None)
    ocr_postprocess_thread.join()

    # Signal visualization thread that everything is done
    vis_output_queue.put(None)
    vis_postprocess_thread.join()

    logger.info('Inference was successful!')

    if show_fps:
        logger.debug(fps_tracker.frame_rate_summary())



def detector_inference_callback(
    completion_info,
    bindings_list: list,
    input_batch: list,
    output_queue,
) -> None:
    """
    Callback triggered after detection inference completes.

    Args:
        completion_info: Info about whether inference succeeded or failed.
        bindings_list (list): Output buffer objects for each input.
        input_batch (list): input frames.
        output_queue (queue.Queue): Queue to pass cropped regions to the OCR pipeline.
    Returns:
        None
    """
    if completion_info.exception:
        logger.error(f'Inference error: {completion_info.exception}')
    else:
        for i, bindings in enumerate(bindings_list):
            result = bindings.output().get_buffer()
            output_queue.put(([input_batch[i], result]))



def detection_postprocess(
    det_postprocess_queue: queue.Queue,
    ocr_input_queue: queue.Queue,
    vis_output_queue: queue.Queue,
    model_height,
    model_width,
) -> None:
    """
    Worker thread to handle postprocessing of detection results.

    Args:
        det_postprocess_queue (queue.Queue): Queue containing tuples of (input_frame, preprocessed_img, result).
        ocr_input_queue (queue.Queue): Queue to send cropped and resized regions along with metadata to OCR stage.
        vis_output_queue (queue.Queue): Queue to send empty OCR results directly to visualization if no detections.
        model_height (int): The height of the model input used for scaling detection boxes.
        model_width (int): The width of the model input used for scaling detection boxes.

    Returns:
        None
    """
    while True:
        item = det_postprocess_queue.get()
        if item is None:
            break  # Shutdown signal

        input_frame, result = item

        det_pp_res, boxes = det_postprocess(result, input_frame, model_height, model_width)

        frame_id = str(uuid.uuid4())
        # Register how many OCR crops are expected from this frame
        ocr_expected_counts[frame_id] = len(det_pp_res)

        # If no text regions were detected, skip OCR and go straight to visualization
        if len(det_pp_res) == 0:
            vis_output_queue.put((input_frame, [], []))
            continue

        # For each detected text region:
        for idx, cropped in enumerate(det_pp_res):
            # Resize the cropped region to match OCR input size (with padding)
            resized = resize_with_padding(cropped)
            # Push one OCR task to the OCR input queue
            ocr_input_queue.put((input_frame, [resized], (frame_id, boxes[idx])))



def ocr_inference_callback(
    completion_info,
    bindings_list: list,
    input_batch: list,
    output_queue: queue.Queue,
    extra_context=None
) -> None:
    """
    Callback triggered after OCR inference completes. Extracts the result, attaches metadata,
    and pushes it to the OCR postprocessing queue.

    Args:
        completion_info: Info about whether inference succeeded or failed.
        bindings_list (list): Output buffer objects from the OCR model.
        input_batch (list): input frame (only one image per batch).
        output_queue (queue.Queue): Queue used to send the OCR results and metadata to the postprocessing stage.
        extra_context (tuple, optional): A tuple of (frame_id, [box]), where `box` is the denormalized detection
                                         bounding box from the detector. Used to group OCR results by frame.

    Returns:
        None
    """
    if completion_info.exception:
        logger.error(f"OCR Inference error: {completion_info.exception}")
        return

    # Handle the single result
    result = bindings_list[0].output().get_buffer()

    # Unpack inputs
    original_frame = input_batch
    frame_id, box = extra_context
    output_queue.put((frame_id, original_frame, result, box))


def ocr_postprocess(
    ocr_postprocess_queue: queue.Queue,
    vis_output_queue: queue.Queue
) -> None:
    """
    Worker thread to handle postprocessing of OCR model results.

    Args:
        ocr_postprocess_queue (queue.Queue): Queue containing tuples of (frame_id, input_frame, ocr_output, denorm_box).
        vis_output_queue (queue.Queue): Queue to pass the final results to visualization.

    Returns:
        None
    """
    while True:

        item = ocr_postprocess_queue.get()
        if item is None:
            break  # Shutdown signal

        frame_id, original_frame, ocr_output, denorm_box = item
        ocr_results_dict[frame_id]["results"].append(ocr_output)
        ocr_results_dict[frame_id]["boxes"].append(denorm_box)
        ocr_results_dict[frame_id]["count"] += 1
        ocr_results_dict[frame_id]["frame"] = original_frame

        expected = ocr_expected_counts.get(frame_id, None)

        # If all OCR results for this frame are collected
        if expected is not None and ocr_results_dict[frame_id]["count"] == expected:
            # Push the grouped results to the visualization queue
            vis_output_queue.put((
                ocr_results_dict[frame_id]["frame"],   # The full input frame
                ocr_results_dict[frame_id]["results"], # All OCR outputs for this frame
                ocr_results_dict[frame_id]["boxes"]    # All box positions for this frame
            ))

            # Clean up to free memory
            del ocr_results_dict[frame_id]
            del ocr_expected_counts[frame_id]


def main() -> None:
    """
    Main function to run the script.
    """
    args = parse_args()
    init_logging(level=level_from_args(args))
    run_inference_pipeline(
        args.det_net,
        args.ocr_net,
        args.input,
        args.batch_size,
        args.output_dir,
        args.camera_resolution,
        args.output_resolution,
        args.frame_rate,
        args.save_output,
        args.show_fps,
        args.use_corrector,
    )


if __name__ == "__main__":
    main()