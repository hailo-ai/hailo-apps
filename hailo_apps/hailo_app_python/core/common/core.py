"""  
Core helpers: arch detection, parser, buffer utils.  
"""  
import os
from pathlib import Path
import argparse
import queue
from dotenv import load_dotenv

from .installation_utils import detect_hailo_arch

from .defines import (
    DEFAULT_DOTENV_PATH,
    DIC_CONFIG_VARIANTS,
    HAILO8_ARCH,
    # for get_resource_path
    RESOURCES_PATH_KEY,
    RESOURCES_ROOT_PATH_DEFAULT,
    HAILO_ARCH_KEY,
    RESOURCES_MODELS_DIR_NAME,
    RESOURCES_SO_DIR_NAME,
    RESOURCES_VIDEOS_DIR_NAME,
    DEPTH_PIPELINE,
    SIMPLE_DETECTION_PIPELINE,
    DETECTION_PIPELINE,
    INSTANCE_SEGMENTATION_PIPELINE,
    POSE_ESTIMATION_PIPELINE,
    DEPTH_MODEL_NAME,
    SIMPLE_DETECTION_MODEL_NAME,
    DETECTION_MODEL_NAME_H8,
    DETECTION_MODEL_NAME_H8L,
    INSTANCE_SEGMENTATION_MODEL_NAME_H8,
    INSTANCE_SEGMENTATION_MODEL_NAME_H8L,
    POSE_ESTIMATION_MODEL_NAME_H8,
    POSE_ESTIMATION_MODEL_NAME_H8L,
    HAILO_FILE_EXTENSION,
    RESOURCES_JSON_DIR_NAME,
    FACE_DETECTION_PIPELINE,
    FACE_DETECTION_MODEL_NAME_H8,
    FACE_DETECTION_MODEL_NAME_H8L,
    FACE_RECOGNITION_PIPELINE,
    FACE_RECOGNITION_MODEL_NAME_H8,
    FACE_RECOGNITION_MODEL_NAME_H8L,
    FACE_RECON_DIR_NAME,
    HAILO10H_ARCH,
    RESOURCES_PHOTOS_DIR_NAME,
    DEFAULT_LOCAL_RESOURCES_PATH,
)

def load_environment(env_file=DEFAULT_DOTENV_PATH, required_vars=None) -> bool:
    """
    Loads environment variables from a .env file and verifies required ones.

    Args:
        env_file (str): Path to the .env file.
        required_vars (list): List of required variable names to validate.
    """
    if env_file is None:
        env_file = DEFAULT_DOTENV_PATH
    load_dotenv(dotenv_path=env_file)

    # check if the virtual env is activated and has all dependencies installed
    print(f"Loading environment variables from {env_file}...")
    env_path = Path(env_file)
    if not os.path.exists(env_path):
        print(f"⚠️ .env file not found: {env_file}")
        return
    if not os.access(env_path, os.R_OK):
        print(f"⚠️ .env file not readable: {env_file}")
        return
    if not os.access(env_path, os.W_OK):
        print(f"⚠️ .env file not writable: {env_file}")
        return
    if not os.access(env_path, os.F_OK):
        print(f"⚠️ .env file not found: {env_file}")
        return

    
    if required_vars is None:
        required_vars = DIC_CONFIG_VARIANTS 
    missing = []
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing.append(var)

    if missing:
        print("⚠️ Missing environment variables: %s", ", ".join(missing))
        return False
    print("✅ All required environment variables loaded.")
    return True    
  
def get_default_parser():
    parser = argparse.ArgumentParser(description="Hailo App Help")
    parser.add_argument(
        "--input", "-i", type=str, default=None,
        help="Input source. Can be a file, USB (webcam), RPi camera (CSI camera module) or ximage. \
        For RPi camera use '-i rpi' \
        For automatically detect a connected usb camera, use '-i usb' \
        For manually specifying a connected usb camera, use '-i /dev/video<X>' \
        Defaults to application specific video."
    )
    parser.add_argument("--use-frame", "-u", action="store_true", help="Use frame from the callback function")
    parser.add_argument("--show-fps", "-f", action="store_true", help="Print FPS on sink")
    parser.add_argument(
            "--arch",
            default=None,
            choices=['hailo8', 'hailo8l', 'hailo10h'],
            help="Specify the Hailo architecture (hailo8 or hailo8l or hailo10h). Default is None , app will run check.",
    )
    parser.add_argument(
            "--hef-path",
            default=None,
            help="Path to HEF file",
    )
    parser.add_argument(
        "--disable-sync", action="store_true",
        help="Disables display sink sync, will run as fast as possible. Relevant when using file source."
    )
    parser.add_argument(
        "--disable-callback", action="store_true",
        help="Disables the user's custom callback function in the pipeline. Use this option to run the pipeline without invoking the callback logic."
    )
    parser.add_argument("--dump-dot", action="store_true", help="Dump the pipeline graph to a dot file pipeline.dot")
    parser.add_argument(
        "--frame-rate", "-r", type=int, default=30,
        help="Frame rate of the video source. Default is 30."
    )
    return parser

def get_model_name(pipeline_name: str, arch: str) -> str:
    # treat both Hailo-8 and Hailo-10H the same
    is_h8 = arch in (HAILO8_ARCH, HAILO10H_ARCH)

    pipeline_map = {
        DEPTH_PIPELINE: DEPTH_MODEL_NAME,
        SIMPLE_DETECTION_PIPELINE: SIMPLE_DETECTION_MODEL_NAME,

        DETECTION_PIPELINE:
            DETECTION_MODEL_NAME_H8 if is_h8 else DETECTION_MODEL_NAME_H8L,

        INSTANCE_SEGMENTATION_PIPELINE:
            INSTANCE_SEGMENTATION_MODEL_NAME_H8 if is_h8 else INSTANCE_SEGMENTATION_MODEL_NAME_H8L,

        POSE_ESTIMATION_PIPELINE:
            POSE_ESTIMATION_MODEL_NAME_H8 if is_h8 else POSE_ESTIMATION_MODEL_NAME_H8L,

        FACE_DETECTION_PIPELINE:
            FACE_DETECTION_MODEL_NAME_H8 if is_h8 else FACE_DETECTION_MODEL_NAME_H8L,

        FACE_RECOGNITION_PIPELINE:
            FACE_RECOGNITION_MODEL_NAME_H8 if is_h8 else FACE_RECOGNITION_MODEL_NAME_H8L,
    }

    return pipeline_map[pipeline_name]

    
def get_resource_path(pipeline_name: str,
                      resource_type: str,
                      model: str = None
                      ) -> Path | None:
    """
    Returns the path to a resource (model, shared object, or video) under the
    base resources directory, using defaults or environment overrides.

    Args:
        pipeline_name: logical name of the pipeline (e.g. 'depth')
        resource_type: one of 'models', 'so', or 'videos'
        model: specific filename (without extension for models)
    """
    # 1) Base resources root
    root = Path(RESOURCES_ROOT_PATH_DEFAULT)

    # 2) Hailo architecture (for model directory)
    arch = os.getenv(HAILO_ARCH_KEY, detect_hailo_arch())
    if not arch:
        return None

    # 3) Shared object (so) and videos: model parameter is full filename
    if resource_type == RESOURCES_SO_DIR_NAME and model:
        return (root / RESOURCES_SO_DIR_NAME / model)
    if resource_type == RESOURCES_VIDEOS_DIR_NAME and model:
        return (root / RESOURCES_VIDEOS_DIR_NAME / model)
    if resource_type == RESOURCES_PHOTOS_DIR_NAME and model:
        return (root / RESOURCES_PHOTOS_DIR_NAME / model)
    if resource_type == RESOURCES_JSON_DIR_NAME and model:
        return (root / RESOURCES_JSON_DIR_NAME / model)
    if resource_type == FACE_RECON_DIR_NAME and model:
        return (root / FACE_RECON_DIR_NAME / model)
    if resource_type == DEFAULT_LOCAL_RESOURCES_PATH and model:
        return (root / DEFAULT_LOCAL_RESOURCES_PATH / model)

    # 4) Models: append architecture and .hef extension edited by oscar madegwa
    if resource_type == RESOURCES_MODELS_DIR_NAME:
        # specific model name provided
        if model:
            return root / RESOURCES_MODELS_DIR_NAME / arch / f"{model}{HAILO_FILE_EXTENSION}"
        # derive model name from pipeline
        if pipeline_name:
            name = get_model_name(pipeline_name, arch)
            return root / RESOURCES_MODELS_DIR_NAME / arch / f"{name}{HAILO_FILE_EXTENSION}"

    return None

class FIFODropQueue(queue.Queue):  # helper class implementing a FIFO queue that drops the oldest item when full (leaky queue)
    def put(self, item, block=False, timeout=None):
        if self.full():
            self.get_nowait()  # remove the oldest frame
        super().put(item, block, timeout)
