import os
import sys
import queue
import threading
from functools import partial
import numpy as np
from pathlib import Path
from hailo_apps.python.core.common.hailo_logger import get_logger, init_logging, level_from_args

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.hailo_inference import HailoInfer
from common.toolbox import (
    init_input_source,
    get_labels,
    load_json_file,
    preprocess,
    visualize,
    FrameRateTracker,
    resolve_net_arg,
    resolve_arch,
    resolve_input_arg,
    resolve_output_resolution_arg,
    list_inputs,
    oriented_object_detection_preprocess,
)
from common.core import handle_list_models_flag
from common.parser import get_standalone_parser
from oriented_object_detection_post_process import inference_result_handler

APP_NAME = Path(__file__).stem
logger = get_logger(__name__)


def parse_args():
    """
    Initialize argument parser for the script.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = get_standalone_parser()
    parser.description = "Oriented object detection using rotated bounding boxes."

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
    args.hef_path = resolve_net_arg(APP_NAME, args.hef_path, ".", args.arch)
    args.input = resolve_input_arg(APP_NAME, args.input)
    args.output_resolution = resolve_output_resolution_arg(args.output_resolution)

    # Setup output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def run_inference_pipeline(
    net,
    input,
    batch_size,
    labels_file,
    output_dir,
    camera_resolution,
    output_resolution,
    frame_rate,
    save_output=False,
    show_fps=False,
) -> None:

    labels = get_labels(labels_file)
    # load local config.json from this example folder
    config_path = str(Path(__file__).parent / "config.json")
    config_data = load_json_file(config_path)

    cap, images = init_input_source(input, batch_size, camera_resolution)
    fps_tracker = None
    if show_fps:
        fps_tracker = FrameRateTracker()

    input_queue = queue.Queue()
    output_queue = queue.Queue()

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
        target=preprocess, args=(images, cap, frame_rate, batch_size, input_queue, width, height, preprocess_callback_fn)
    )
    postprocess_thread = threading.Thread(
        target=visualize, args=(output_queue, cap, save_output,
                                output_dir, post_process_callback_fn, fps_tracker, output_resolution, frame_rate)
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
    output_queue.put(None)
    postprocess_thread.join()

    if show_fps:
        logger.debug(fps_tracker.frame_rate_summary())

    logger.info('Oriented inference finished')


def infer(hailo_inference, input_queue, output_queue):
    while True:
        next_batch = input_queue.get()
        if not next_batch:
            break
        input_batch, preprocessed_batch = next_batch

        inference_callback_fn = partial(
            inference_callback,
            input_batch=input_batch,
            output_queue=output_queue
        )

        hailo_inference.run(preprocessed_batch, inference_callback_fn)

    hailo_inference.close()


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
        args.show_fps,
    )


if __name__ == "__main__":
    main()
