import logging
import threading
import traceback
from typing import Callable, Optional

import numpy as np

from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils.terminal_ui import TerminalUI
from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.audio_recorder import AudioRecorder

# Setup logger
logger = logging.getLogger(__name__)


class VoiceInteractionManager:
    """
    Manages the interactive voice loop (recording, UI, events).

    This class abstracts the common pattern of:
    1. Waiting for user input (SPACE to record, Q to quit, C to clear context).
    2. Managing the AudioRecorder.
    3. Delegating processing to callbacks.
    """

    def __init__(
        self,
        title: str,
        on_audio_ready: Callable[[np.ndarray], None],
        on_processing_start: Optional[Callable[[], None]] = None,
        on_clear_context: Optional[Callable[[], None]] = None,
        on_shutdown: Optional[Callable[[], None]] = None,
        on_abort: Optional[Callable[[], None]] = None,
        debug: bool = False,
    ):
        """
        Args:
            title (str): Title for the terminal banner.
            on_audio_ready (Callable): Callback when recording finishes with audio data.
            on_processing_start (Callable): Callback when recording starts (e.g. to stop TTS).
            on_clear_context (Callable): Callback when 'C' is pressed.
            on_shutdown (Callable): Callback when 'Q' or Ctrl+C is pressed.
            on_abort (Callable): Callback when 'X' is pressed.
            debug (bool): Enable debug logging for recorder.
        """
        self.title = title
        self.on_audio_ready = on_audio_ready
        self.on_processing_start = on_processing_start
        self.on_clear_context = on_clear_context
        self.on_shutdown = on_shutdown
        self.on_abort = on_abort
        self.debug = debug

        self.recorder = AudioRecorder(debug=debug)
        self.is_recording = False
        self.lock = threading.Lock()

        self.controls = {
            "SPACE": "start/stop recording",
            "Q": "quit",
            "C": "clear context",
            "X": "abort generation",
        }

        # Check if we have a valid input device
        if self.recorder.device_id is None:
            print("\n⚠️  WARNING: No audio input device detected!")
            print("Run 'hailo-audio-troubleshoot' to diagnose audio issues.")

    def run(self):
        """Starts the main interaction loop."""
        TerminalUI.show_banner(title=self.title, controls=self.controls)
        logger.debug("Voice interaction started")

        try:
            while True:
                ch = TerminalUI.get_char().lower()
                if ch == "q":
                    logger.info("Quit requested")
                    self.close()
                    break
                elif ch == " ":
                    self.toggle_recording()
                elif ch == "\x03":  # Ctrl+C
                    logger.info("Ctrl+C received")
                    self.close()
                    break
                elif ch == "c":
                    logger.info("Clear context requested")
                    if self.on_clear_context:
                        self.on_clear_context()
                elif ch == "x":
                    logger.info("Abort requested")
                    if self.on_abort:
                        self.on_abort()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt in main loop")
            self.close()
        except Exception as e:
            logger.error("Unexpected error in main loop: %s", e)
            if self.debug:
                logger.debug("Traceback: %s", traceback.format_exc())

    def toggle_recording(self):
        with self.lock:
            if not self.is_recording:
                self.start_recording()
            else:
                self.stop_recording()

    def start_recording(self):
        if self.on_processing_start:
            try:
                self.on_processing_start()
            except Exception as e:
                logger.warning("Failed to execute start callback: %s", e)

        try:
            self.recorder.start()
            self.is_recording = True
            print("\n🔴 Recording started. Press SPACE to stop.")
        except Exception as e:
            logger.error("Failed to start recording: %s", e)
            print("\n❌ Error starting recording!")
            print(f"Error: {e}")
            print("Tip: Run 'hailo-audio-troubleshoot' to check your microphone.")
            self.is_recording = False

    def stop_recording(self):
        print("\nProcessing... Please wait.")
        try:
            audio = self.recorder.stop()
        except Exception as e:
            logger.error("Failed to stop recording: %s", e)
            self.is_recording = False
            return

        self.is_recording = False

        if audio.size > 0:
            logger.debug("Audio captured: %d samples", audio.size)
            if self.on_audio_ready:
                # Run processing in a separate thread to keep UI responsive (for abort)
                def processing_wrapper():
                    try:
                        self.on_audio_ready(audio)
                    except Exception as e:
                        logger.error("Processing failed: %s", e)
                        if self.debug:
                            logger.debug("Traceback: %s", traceback.format_exc())
                    finally:
                        # Show banner again after processing is done
                        TerminalUI.show_banner(title=self.title, controls=self.controls)

                threading.Thread(target=processing_wrapper, daemon=True).start()
        else:
            logger.warning("No audio recorded")
            print("⚠️  No audio recorded.")
            TerminalUI.show_banner(title=self.title, controls=self.controls)

    def close(self):
        logger.debug("Shutting down voice interaction")
        if self.is_recording:
            try:
                self.recorder.stop()
            except Exception as e:
                logger.debug("Error stopping recorder during shutdown: %s", e)

        try:
            self.recorder.close()
        except Exception as e:
            logger.debug("Error closing recorder: %s", e)

        if self.on_shutdown:
            try:
                self.on_shutdown()
            except Exception as e:
                logger.error("Failed during shutdown callback: %s", e)
