import argparse
import sys

import cv2
import numpy as np

from hailo_platform import VDevice
from hailo_platform.genai import VLM

from hailo_apps.python.core.common.core import handle_list_models_flag, resolve_hef_path
from hailo_apps.python.core.common.defines import VLM_CHAT_APP, SHARED_VDEVICE_GROUP_ID, HAILO10H_ARCH, REPO_ROOT
from hailo_apps.python.core.common.hailo_logger import get_logger

# Initialize logger
logger = get_logger(__name__)


def main():
    """Main function for VLM Chat Example."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="VLM Chat Example")
    parser.add_argument("--hef-path", type=str, default=None, help="Path to HEF model file")
    parser.add_argument("--list-models", action="store_true", help="List available models")

    # Handle --list-models flag before full initialization
    handle_list_models_flag(parser, VLM_CHAT_APP)

    args = parser.parse_args()

    # Resolve HEF path with auto-download (VLM is Hailo-10H only)
    hef_path = resolve_hef_path(args.hef_path, app_name=VLM_CHAT_APP, arch=HAILO10H_ARCH)
    if hef_path is None:
        logger.error("Failed to resolve HEF path for VLM model.")
        sys.exit(1)

    logger.info(f"Using HEF: {hef_path}")
    print(f"✓ Model file found: {hef_path}")

    vdevice = None
    vlm = None

    try:
        print("\n[1/5] Initializing Hailo device...")
        params = VDevice.create_params()
        params.group_id = SHARED_VDEVICE_GROUP_ID
        vdevice = VDevice(params)
        print("✓ Hailo device initialized")

        print("[2/5] Loading VLM model...")
        vlm = VLM(vdevice, str(hef_path))
        print("✓ Model loaded successfully")

        prompt = [
            {
                "role": "system",
                "content": [{"type": "text", "text": 'You are a helpful assistant that analyzes images and answers questions about them.'}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": 'How many people in the image?.'}
                ]
            }
        ]

        # Load and convert image
        # Use standard REPO_ROOT from defines
        image_path = REPO_ROOT / 'doc' / 'images' / 'barcode-example.png'

        print(f"[3/5] Loading image from: {image_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image file: {image_path}")
        print(f"✓ Image loaded (size: {image.shape[1]}x{image.shape[0]})")

        print("[4/5] Preprocessing image...")
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (336, 336), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        print("✓ Image preprocessed (resized to 336x336, converted to RGB)")

        print("[5/5] Sending prompt with image to VLM...")
        print(f"   User question: '{prompt[1]['content'][1]['text']}'")
        response = vlm.generate_all(prompt=prompt, frames=[image], temperature=0.1, seed=42, max_generated_tokens=200)

        print("\nResponse received:")
        print("-" * 60)
        print(response.split(". [{'type'")[0].split("<|im_end|>")[0])
        print("-" * 60)
        print("\n✓ Example completed successfully")

    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
        sys.exit(1)

    finally:
        # Clean up resources
        if vlm:
            try:
                vlm.clear_context()
                vlm.release()
            except Exception as e:
                logger.warning(f"Error releasing VLM: {e}")

        if vdevice:
            try:
                vdevice.release()
            except Exception as e:
                logger.warning(f"Error releasing VDevice: {e}")


if __name__ == "__main__":
    main()

