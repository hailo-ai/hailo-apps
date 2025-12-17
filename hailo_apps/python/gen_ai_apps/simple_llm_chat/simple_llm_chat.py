import argparse
import sys

from hailo_platform import VDevice
from hailo_platform.genai import LLM

from hailo_apps.python.core.common.core import handle_list_models_flag, resolve_hef_path
from hailo_apps.python.core.common.defines import LLM_CHAT_APP, SHARED_VDEVICE_GROUP_ID, HAILO10H_ARCH
from hailo_apps.python.core.common.hailo_logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Parse arguments
parser = argparse.ArgumentParser(description="LLM Chat Example")
parser.add_argument("--hef-path", type=str, default=None, help="Path to HEF model file")
parser.add_argument("--list-models", action="store_true", help="List available models")

# Handle --list-models flag before full initialization
handle_list_models_flag(parser, LLM_CHAT_APP)

args = parser.parse_args()

# Resolve HEF path with auto-download (LLM is Hailo-10H only)
hef_path = resolve_hef_path(args.hef_path, app_name=LLM_CHAT_APP, arch=HAILO10H_ARCH)
if hef_path is None:
    logger.error("Failed to resolve HEF path for LLM model.")
    sys.exit(1)

logger.info(f"Using HEF: {hef_path}")
print(f"✓ Model file found: {hef_path}")

vdevice = None
llm = None

try:
    print("\n[1/4] Initializing Hailo device...")
    params = VDevice.create_params()
    params.group_id = SHARED_VDEVICE_GROUP_ID
    vdevice = VDevice(params)
    print("✓ Hailo device initialized")

    print("[2/4] Loading LLM model...")
    llm = LLM(vdevice, str(hef_path))
    print("✓ Model loaded successfully")

    prompt = [
        {"role": "system", "content": [{"type": "text", "text": 'You are a helpful assistant.'}]},
        {"role": "user", "content": [{"type": "text", "text": 'Tell a short joke.'}]}
    ]

    print("[3/4] Sending prompt to LLM...")
    print(f"   User prompt: '{prompt[1]['content'][0]['text']}'")
    response = llm.generate_all(prompt=prompt, temperature=0.1, seed=42, max_generated_tokens=200)

    print("[4/4] Response received:")
    print("-" * 60)
    print(response.split(". [{'type'")[0])
    print("-" * 60)
    print("\n✓ Example completed successfully")

except Exception as e:
    logger.error(f"Error occurred: {e}", exc_info=True)

finally:
    if llm:
        try:
            llm.clear_context()
            llm.release()
        except Exception as e:
            logger.warning(f"Error releasing LLM: {e}")

    if vdevice:
        try:
            vdevice.release()
        except Exception as e:
            logger.warning(f"Error releasing VDevice: {e}")

