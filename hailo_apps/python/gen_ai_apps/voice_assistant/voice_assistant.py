import argparse
import threading
from io import StringIO
from contextlib import redirect_stderr

from hailo_platform import VDevice
from hailo_platform.genai import LLM

from hailo_apps.python.core.common.defines import LLM_PROMPT_PREFIX, SHARED_VDEVICE_GROUP_ID, RESOURCES_MODELS_DIR_NAME, LLM_MODEL_NAME_H10
from hailo_apps.python.core.common.core import get_resource_path
from hailo_apps.python.core.common.hailo_logger import add_logging_cli_args, init_logging, level_from_args
from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.interaction import VoiceInteractionManager
from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.speech_to_text import SpeechToTextProcessor
from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.text_to_speech import (
    TextToSpeechProcessor,
    PiperModelNotFoundError,
)
from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils import streaming


class VoiceAssistantApp:
    """
    Manages the main application logic for the voice assistant.
    Builds the pipeline using common AI components.
    """

    def __init__(self, debug=False, no_tts=False):
        self.debug = debug
        self.no_tts = no_tts
        self.abort_event = threading.Event()

        print("Initializing AI components... (This might take a moment)")

        # Suppress noisy ALSA messages during initialization
        with redirect_stderr(StringIO()):
            # 1. VDevice
            params = VDevice.create_params()
            params.group_id = SHARED_VDEVICE_GROUP_ID
            self.vdevice = VDevice(params)

            # 2. Speech to Text
            self.s2t = SpeechToTextProcessor(self.vdevice)

            # 3. LLM
            # USER CONFIGURATION: You can change the LLM model here.
            # By default, it uses the model defined in LLM_MODEL_NAME_H10.
            # To use a custom HEF, provide the absolute path to your .hef file.
            model_path = str(
                get_resource_path(
                    pipeline_name=None,
                    resource_type=RESOURCES_MODELS_DIR_NAME,
                    model=LLM_MODEL_NAME_H10,
                )
            )
            # Example of using a custom path:
            # model_path = "/path/to/your/custom_model.hef"

            self.llm = LLM(self.vdevice, model_path)

            # 4. TTS
            self.tts = None
            if not no_tts:
                self.tts = TextToSpeechProcessor()

        print("✅ AI components ready!")

    def on_processing_start(self):
        self.on_abort()
        if self.tts:
            self.tts.interrupt()

    def on_abort(self):
        """Abort current generation and speech."""
        self.abort_event.set()
        if self.tts:
            self.tts.interrupt()

    def on_audio_ready(self, audio):
        self.abort_event.clear()

        # 1. Transcribe
        user_text = self.s2t.transcribe(audio)
        if not user_text:
            print("No speech detected.")
            return

        print(f"\nYou: {user_text}")
        print("\nLLM response:\n")

        # 2. Prepare TTS
        current_gen_id = None
        # Use a mutable container to track state inside callback
        state = {
            'sentence_buffer': "",
            'first_chunk_sent': False
        }

        if self.tts:
            self.tts.clear_interruption()
            current_gen_id = self.tts.get_current_gen_id()

        # 3. Generate Response
        prompt_text = LLM_PROMPT_PREFIX + user_text

        # Format prompt as a list of messages for the LLM
        formatted_prompt = [{'role': 'user', 'content': prompt_text}]

        def tts_callback(chunk: str):
            if self.tts:
                state['sentence_buffer'] += chunk
                # Chunk and queue speech using the centralized method
                state['sentence_buffer'] = self.tts.chunk_and_queue(
                    state['sentence_buffer'], current_gen_id, not state['first_chunk_sent']
                )

                if not state['first_chunk_sent'] and not self.tts.speech_queue.empty():
                    state['first_chunk_sent'] = True

        # Use streaming utility to handle generation, printing, and TTS callback
        # Note: simple voice assistant might not use tools, so filtered output should be clean text
        streaming.generate_and_stream_response(
            llm=self.llm,
            prompt=formatted_prompt,
            prefix="", # No prefix for this app
            show_raw_stream=self.debug,  # Show raw stream in debug mode
            token_callback=tts_callback,
            abort_callback=self.abort_event.is_set
        )

        # 4. Send remaining text
        if self.tts and state['sentence_buffer'].strip():
            self.tts.queue_text(state['sentence_buffer'].strip(), current_gen_id)

        print()

    def on_clear_context(self):
        self.llm.clear_context()
        print("Context cleared.")

    def close(self):
        if self.tts:
            self.tts.stop()

        # Clean up LLM resources
        try:
            self.llm.release()
        except Exception:
            pass

        # We rely on process cleanup for VDevice release or Python's GC,
        # matching original pattern.
        # Ideally self.vdevice.release() but it's shared.


def main():
    parser = argparse.ArgumentParser(
        description='A simple, voice-controlled AI assistant for your terminal.')
    add_logging_cli_args(parser)
    parser.add_argument('--no-tts', action='store_true',
                        help='Disable text-to-speech output for lower resource usage.')

    args = parser.parse_args()

    # Initialize logging
    init_logging(level=level_from_args(args))

    # Check if debug mode is enabled (for logging purposes)
    debug_mode = getattr(args, 'debug', False)

    if args.no_tts:
        print("TTS disabled: Running in low-resource mode.")

    # Initialize the app
    try:
        app = VoiceAssistantApp(debug=debug_mode, no_tts=args.no_tts)
    except PiperModelNotFoundError as e:
        # Piper model not found - exit with error message
        print("ERROR: TTS model not found. Use --no-tts to run without TTS, or install the Piper model.")
        print(str(e))
        return 1

    # Initialize the interaction manager
    interaction = VoiceInteractionManager(
        title="Voice Assistant",
        on_audio_ready=app.on_audio_ready,
        on_processing_start=app.on_processing_start,
        on_clear_context=app.on_clear_context,
        on_shutdown=app.close,
        on_abort=app.on_abort,
        debug=debug_mode
    )

    interaction.run()


if __name__ == "__main__":
    main()
