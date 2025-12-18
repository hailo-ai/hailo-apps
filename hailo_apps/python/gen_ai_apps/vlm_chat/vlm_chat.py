import threading
import signal
import os
import cv2
import sys
import concurrent.futures
import select
import time
from typing import Optional, Callable, Any

os.environ["QT_QPA_PLATFORM"] = 'xcb'
from hailo_apps.python.gen_ai_apps.vlm_chat.backend import Backend
from hailo_apps.python.core.common.core import get_standalone_parser, get_resource_path, get_logger, handle_list_models_flag, resolve_hef_path
from hailo_apps.python.core.common.camera_utils import get_usb_video_devices
from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import get_source_type
from hailo_apps.python.core.common.defines import (
    VLM_CHAT_APP,
    VLM_MODEL_NAME_H10,
    RESOURCES_MODELS_DIR_NAME,
    HAILO10H_ARCH,
    RPI_NAME_I,
    USB_CAMERA
)

# Configuration Constants
MAX_TOKENS = 200
TEMPERATURE = 0.1
SEED = 42
SYSTEM_PROMPT = "You are a helpful assistant that analyzes images and answers questions about them."
INFERENCE_TIMEOUT = 60
SAVE_FRAMES = False

# App States
STATE_STREAMING = "STREAMING"
STATE_CAPTURED = "CAPTURED"
STATE_PROCESSING = "PROCESSING"
STATE_RESULT = "RESULT"

# Initialize logger
logger = get_logger(__name__)

class VLMChatApp:
    """
    Main application class for VLM Chat.
    Handles video display, user input, and interaction with the VLM backend.
    """
    def __init__(self, camera: Any, camera_type: str):
        """
        Initialize the VLM Chat Application.

        Args:
            camera (Any): Camera source (device index or connection object).
            camera_type (str): Type of camera ('usb' or 'rpi').
        """
        self.camera = camera
        self.camera_type = camera_type
        self.running = True
        self.executor = concurrent.futures.ThreadPoolExecutor()
        signal.signal(signal.SIGINT, self.signal_handler)
        self.frozen_frame = None
        self.current_state = STATE_STREAMING
        self.user_question = ''
        self.backend: Optional[Backend] = None
        self.video_thread: Optional[threading.Thread] = None

    def signal_handler(self, sig, frame):
        """Handle interrupt signals."""
        print('')
        logger.info("Signal received, shutting down...")
        self.stop()

    def stop(self):
        """Stop the application and clean up resources."""
        self.running = False
        if self.backend:
            self.backend.close()
        self.executor.shutdown(wait=True)

    def _get_user_input(self) -> Optional[str]:
        """
        Check for user input from stdin without blocking.

        Returns:
            Optional[str]: User input string if available, else None.
        """
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.readline().strip()
        return None

    def _init_camera(self) -> tuple[Callable[[], Any], Callable[[], None], str]:
        """
        Initialize the camera based on type.

        Returns:
            tuple: (get_frame_callback, cleanup_callback, camera_name)
        """
        if self.camera_type == RPI_NAME_I:
            try:
                from picamera2 import Picamera2
                picam2 = Picamera2()
                config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
                picam2.configure(config)
                picam2.start()
                # Convert RGB to BGR to match OpenCV/USB standard
                get_frame = lambda: cv2.cvtColor(picam2.capture_array(), cv2.COLOR_RGB2BGR)
                cleanup = lambda: picam2.stop()
                camera_name = "RPI"
                return get_frame, cleanup, camera_name
            except (ImportError, Exception) as e:
                logger.error(f"Error initializing RPI camera: {e}")
                raise
        else:
            cap = cv2.VideoCapture(self.camera)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            get_frame = lambda: (lambda r: r[1] if r[0] else None)(cap.read())
            cleanup = lambda: cap.release()
            camera_name = "USB"
            return get_frame, cleanup, camera_name

    def _print_state_prompt(self):
        """Print prompt based on current state."""
        print("\n" + "="*80)
        if self.current_state == STATE_STREAMING:
            print("  🎥  LIVE VIDEO  |  Press Enter to CAPTURE image ('q' to quit)")
            print("="*80)
        elif self.current_state == STATE_CAPTURED:
            print("  📷  IMAGE CAPTURED  |  Type question (Enter='Describe the image', 'q' to Cancel)")
            print("="*80)
            print("Question: ", end="", flush=True)
        elif self.current_state == STATE_PROCESSING:
            print("  ⏳  PROCESSING...  |  Please wait")
            print("="*80)
        elif self.current_state == STATE_RESULT:
            print("  ✅  RESULT READY  |  Press Enter to continue")
            print("="*80)

    def show_video(self):
        """Main loop for displaying video and handling user interaction."""
        try:
            get_frame, cleanup, _ = self._init_camera()
        except Exception:
            logger.error("Failed to initialize camera. Exiting.")
            self.running = False
            return

        # Initialize Backend
        try:
            # Use globally resolved hef_path
            self.backend = Backend(
                hef_path=str(hef_path),
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                seed=SEED,
                system_prompt=SYSTEM_PROMPT
            )
        except Exception as e:
            logger.error(f"Failed to initialize backend: {e}")
            cleanup()
            self.running = False
            return

        vlm_future = None

        # Initial Prompt
        self._print_state_prompt()

        # Ensure frame is defined in outer scope for safety
        frame = None

        try:
            while self.running:
                # Display Logic
                if self.current_state == STATE_STREAMING:
                    raw_frame = get_frame()
                    if raw_frame is None:
                        logger.error("Failed to read frame from camera")
                        break

                    # Pre-process frame to show user exactly what the model sees
                    # This ensures live video matches the aspect ratio (central crop)
                    rgb_frame = Backend.convert_resize_image(raw_frame)
                    frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

                    cv2.imshow('Video', frame)
                elif self.current_state in [STATE_CAPTURED, STATE_PROCESSING, STATE_RESULT]:
                    if self.frozen_frame is not None:
                        cv2.imshow('Video', self.frozen_frame)

                # Key Handling (Window)
                key = cv2.waitKey(25) & 0xFF
                if key == ord('q'):
                    self.stop()
                    break

                # Input Handling (Terminal)
                user_input = self._get_user_input()

                # State Machine Logic
                if self.current_state == STATE_STREAMING:
                    if user_input is not None: # User pressed Enter (or typed something)
                        stripped = user_input.strip()
                        if stripped.lower() in ['q', 'quit']:
                            self.stop()
                            break

                        # Capture current frame
                        # Note: frame variable holds the last captured frame from the display block
                        if frame is not None:
                            self.frozen_frame = frame.copy()
                            self.current_state = STATE_CAPTURED
                            self._print_state_prompt()

                elif self.current_state == STATE_CAPTURED:
                    if user_input is not None:
                        stripped = user_input.strip()
                        if stripped.lower() in ['q', 'quit']:
                            self.current_state = STATE_STREAMING
                            self.frozen_frame = None
                            self._print_state_prompt()
                            continue

                        self.user_question = stripped or "Describe the image"
                        if not stripped:
                            print(f"Using default prompt: '{self.user_question}'")

                        if SAVE_FRAMES and self.frozen_frame is not None:
                            timestamp = time.strftime("%Y%m%d-%H%M%S")
                            cv2.imwrite(f"frame_{timestamp}.jpg", self.frozen_frame)
                            print(f"Frame saved as frame_{timestamp}.jpg")

                        self.current_state = STATE_PROCESSING
                        self._print_state_prompt()

                        vlm_future = self.executor.submit(
                            self.backend.vlm_inference,
                            self.frozen_frame.copy(),
                            self.user_question,
                            INFERENCE_TIMEOUT
                        )

                elif self.current_state == STATE_PROCESSING:
                    if vlm_future and vlm_future.done():
                        try:
                            # We get the result but we don't print the full answer again
                            # because it was streamed by the worker process.
                            # We only handle errors or unexpected cases here.
                            # You can get the full answer by calling result.get('answer')
                            # print(f"\n\nAnswer: {result.get('answer')}")
                            result = vlm_future.result()
                        except Exception as e:
                            logger.error(f"Error getting future result: {e}")
                            print(f"\nError processing request: {e}")

                        vlm_future = None
                        self.current_state = STATE_RESULT
                        self._print_state_prompt()

                elif self.current_state == STATE_RESULT:
                    if user_input is not None: # User pressed Enter
                        self.current_state = STATE_STREAMING
                        self.frozen_frame = None
                        self._print_state_prompt()

        finally:
            cleanup()
            cv2.destroyAllWindows()
            self.stop()

    def run(self):
        """Start the application thread."""
        self.video_thread = threading.Thread(target=self.show_video)
        self.video_thread.start()
        try:
            self.video_thread.join()
        except KeyboardInterrupt:
            self.stop()
            self.video_thread.join()

if __name__ == "__main__":
    parser = get_standalone_parser()

    # Handle --list-models flag before full initialization
    handle_list_models_flag(parser, VLM_CHAT_APP)

    options_menu = parser.parse_args()

    # Resolve HEF path with auto-download (VLM is Hailo-10H only)
    hef_path = resolve_hef_path(
        options_menu.hef_path if hasattr(options_menu, 'hef_path') else None,
        app_name=VLM_CHAT_APP,
        arch=HAILO10H_ARCH
    )
    if hef_path is None:
        logger.error("Failed to resolve HEF path for VLM model. Exiting.")
        sys.exit(1)

    video_source = options_menu.input
    if video_source == USB_CAMERA:
        logger.debug("USB_CAMERA detected; scanning USB devices...")
        video_source = get_usb_video_devices()
        if not video_source:
            logger.error("No USB camera found for '--input usb'")
            print(
                'Provided argument "--input" is set to "usb", however no available USB cameras found. Please connect a camera or specifiy different input method.'
            )
            sys.exit(1)
        else:
            logger.debug(f"Using USB camera: {video_source[0]}")
            video_source = video_source[0]

    # Determine source type (usb, rpi, file, etc.)
    source_type = get_source_type(video_source) if video_source else None

    if not video_source:
        print('Please provide an input source using the "--input" argument: "usb" for USB camera or "rpi" for Raspberry Pi camera.')
        sys.exit(1)

    app = VLMChatApp(camera=video_source, camera_type=source_type)
    app.run()
    sys.exit(0)
