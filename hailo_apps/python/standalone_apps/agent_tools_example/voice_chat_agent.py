"""
Voice-enabled Interactive CLI chat agent.

Combines voice input/output capabilities with the tool-using chat agent.

Usage:
  python -m hailo_apps.python.standalone_apps.agent_tools_example.voice_chat_agent
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from pathlib import Path

from hailo_platform import VDevice
from hailo_platform.genai import LLM

from hailo_apps.python.core.gen_ai_utils.voice_processing.interaction import VoiceInteractionManager
from hailo_apps.python.core.gen_ai_utils.voice_processing.speech_to_text import SpeechToTextProcessor
from hailo_apps.python.core.gen_ai_utils.voice_processing.text_to_speech import TextToSpeechProcessor
from hailo_apps.python.core.gen_ai_utils.llm_utils import (
    agent_utils,
    context_manager,
    message_formatter,
    streaming,
    tool_discovery,
    tool_execution,
    tool_parsing,
    tool_selection,
)
from hailo_apps.python.core.common.defines import (
    SHARED_VDEVICE_GROUP_ID,
    AGENT_APP,
    WHISPER_CHAT_APP,
    HAILO10H_ARCH,
)
from hailo_apps.python.core.common.core import (
    get_standalone_parser,
    handle_list_models_flag,
    resolve_hef_path,
)
from hailo_apps.python.core.common.hailo_logger import get_logger, init_logging, level_from_args

try:
    from . import config, system_prompt
except ImportError:
    # Add the script's directory to sys.path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    import config
    import system_prompt


logger = get_logger(__name__)


class VoiceAgentApp:
    """
    Voice-enabled chat agent application.

    Combines speech-to-text, LLM inference, and text-to-speech
    for interactive voice-based tool calling.
    """

    def __init__(self, llm_hef_path, selected_tool, whisper_hef_path=None, debug=False, no_tts=False):
        """
        Initialize the voice agent application.

        Args:
            llm_hef_path: Path to the LLM HEF model file (Qwen2.5-Coder).
            selected_tool: The selected tool configuration dict.
            whisper_hef_path: Optional path to the Whisper HEF model file.
                             If None, uses the default from whisper_chat app.
            debug: Enable debug mode.
            no_tts: Disable text-to-speech.
        """
        self.debug = debug
        self.no_tts = no_tts
        self.selected_tool = selected_tool
        self.selected_tool_name = selected_tool.get("name", "")

        # Initialize tools lookup
        self.tools_lookup = {self.selected_tool_name: selected_tool}

        print("Initializing AI components...")

        # Initialize Hailo VDevice and Models
        try:
            params = VDevice.create_params()
            params.group_id = SHARED_VDEVICE_GROUP_ID  # Support multiple processes using the same VDevice
            self.vdevice = VDevice(params)
        except Exception as e:
            logger.error("Failed to create VDevice: %s", e)
            raise

        # LLM (Qwen2.5-Coder-1.5B-Instruct)
        try:
            self.llm = LLM(self.vdevice, str(llm_hef_path))
        except Exception as e:
            logger.error("Failed to initialize LLM: %s", e)
            self.vdevice.release()
            raise

        # S2T (Whisper-Base)
        try:
            self.s2t = SpeechToTextProcessor(self.vdevice, hef_path=whisper_hef_path)
        except Exception as e:
            logger.error("Failed to initialize Speech-to-Text: %s", e)
            self.llm.release()
            self.vdevice.release()
            raise

        # TTS
        self.tts = None
        if not no_tts:
            try:
                self.tts = TextToSpeechProcessor()
            except Exception as e:
                logger.warning("Failed to initialize TTS: %s", e)
                print("Continuing without TTS support.")

        # Initialize Context
        self._init_context()

        print("✅ AI components ready!")

    def _init_context(self):
        """Initialize the LLM context with system prompt."""
        system_text = system_prompt.create_system_prompt([self.selected_tool])
        logger.debug("System prompt: %d chars", len(system_text))

        cache_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        try:
            context_loaded = context_manager.load_context_from_cache(self.llm, self.selected_tool_name, cache_dir, logger)
        except Exception as e:
            logger.warning("Failed to load context cache: %s", e)
            context_loaded = False

        if not context_loaded:
            logger.info("Initializing context...")
            try:
                prompt = [message_formatter.messages_system(system_text)]
                context_manager.add_to_context(self.llm, prompt, logger)
                context_manager.save_context_to_cache(self.llm, self.selected_tool_name, cache_dir, logger)
            except Exception as e:
                logger.error("Failed to initialize system context: %s", e)
        else:
            logger.debug("Using cached context")

        self.system_text = system_text

    def on_processing_start(self):
        """Callback when processing starts - interrupt any ongoing TTS."""
        if self.tts:
            try:
                self.tts.interrupt()
            except Exception as e:
                logger.debug("TTS interrupt failed: %s", e)

    def on_audio_ready(self, audio):
        """
        Callback when audio recording is ready.

        Args:
            audio: The recorded audio data.
        """
        # 1. Transcribe
        try:
            user_text = self.s2t.transcribe(audio)
        except Exception as e:
            logger.error("Transcription failed: %s", e)
            return

        if not user_text:
            print("No speech detected.")
            return

        print(f"\nYou: {user_text}")

        # 2. Process with LLM and Tools
        self.process_interaction(user_text)

    def on_clear_context(self):
        """Callback to clear the LLM context."""
        if self.llm:
            try:
                self.llm.clear_context()
                print("Context cleared.")

                # Try to reload cached context after clearing
                cache_dir = Path(os.path.dirname(os.path.abspath(__file__)))
                context_reloaded = context_manager.load_context_from_cache(
                    self.llm, self.selected_tool_name, cache_dir, logger
                )
                if context_reloaded:
                    logger.debug("Context reloaded")
                else:
                    logger.debug("No cache, will reinit")
            except Exception as e:
                logger.error("Failed to clear context: %s", e)

    def process_interaction(self, user_text):
        """
        Process a user interaction with the LLM.

        Args:
            user_text: The transcribed user text.
        """
        # Check if we need to trim context based on actual token usage
        if context_manager.is_context_full(self.llm, context_threshold=config.CONTEXT_THRESHOLD, logger_instance=logger):
            logger.info("Context full, clearing...")
            self.llm.clear_context()
            cache_dir = Path(os.path.dirname(os.path.abspath(__file__)))

            if context_manager.load_context_from_cache(self.llm, self.selected_tool_name, cache_dir, logger):
                logger.debug("Context restored")
            else:
                logger.warning("Context cleared but failed to restore from cache")

        prompt = [message_formatter.messages_user(user_text)]
        logger.debug("User message: %s", json.dumps(prompt, ensure_ascii=False))

        # Prepare for streaming response
        current_gen_id = None
        # Using a mutable container to track state inside inner function
        state = {
            'sentence_buffer': "",
            'first_chunk_sent': False
        }

        if self.tts:
            # Clear any pending speech to respond to new input immediately
            self.tts.clear_interruption()
            current_gen_id = self.tts.get_current_gen_id()

        def tts_callback(chunk: str):
            if self.tts:
                logger.debug("[TTS] Received chunk: %r", chunk[:50])
                state['sentence_buffer'] += chunk
                # Chunk speech
                old_buffer_len = len(state['sentence_buffer'])
                state['sentence_buffer'] = self.tts.chunk_and_queue(
                    state['sentence_buffer'], current_gen_id, not state['first_chunk_sent']
                )
                if len(state['sentence_buffer']) < old_buffer_len:
                    logger.debug("[TTS] Queued text, queue size: %d", self.tts.speech_queue.qsize())

                if not state['first_chunk_sent'] and not self.tts.speech_queue.empty():
                    state['first_chunk_sent'] = True
                    logger.debug("[TTS] First chunk queued for playback")

        try:
            # Use generate() for streaming output with on-the-fly filtering and TTS callback
            is_debug = logger.isEnabledFor(logging.DEBUG)
            raw_response = streaming.generate_and_stream_response(
                llm=self.llm,
                prompt=prompt,
                temperature=config.TEMPERATURE,
                prefix="Assistant: ",
                debug_mode=is_debug,
                token_callback=tts_callback
            )
        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            logger.debug("Traceback: %s", traceback.format_exc())
            return

        # Flush remaining speech
        if self.tts and state['sentence_buffer'].strip():
            remaining_text = state['sentence_buffer'].strip()
            logger.debug("[TTS] Flushing remaining buffer: %s", remaining_text[:100])
            self.tts.queue_text(remaining_text, current_gen_id)
            logger.debug("[TTS] Final queue size: %d", self.tts.speech_queue.qsize())

        # Check for tool calls
        tool_call = tool_parsing.parse_function_call(raw_response)
        if tool_call:
            self.handle_tool_call(tool_call)

    def handle_tool_call(self, tool_call):
        """
        Handle a tool call from the LLM.

        Args:
            tool_call: The parsed tool call dict.
        """
        # Execute tool
        result = tool_execution.execute_tool_call(tool_call, self.tools_lookup)
        tool_execution.print_tool_result(result)

        if self.tts:
            if result.get("ok"):
                res_str = str(result.get("result", ""))
                logger.debug("[TTS] Queuing tool result: %s", res_str[:100])
                self.tts.queue_text(res_str)
                logger.debug("[TTS] Queue size after queuing: %d", self.tts.speech_queue.qsize())
            else:
                error_msg = "There was an error executing the tool."
                logger.debug("[TTS] Queuing error message")
                self.tts.queue_text(error_msg)

        agent_utils.update_context_with_tool_result(self.llm, result, logger)

    def close(self):
        """Clean up resources."""
        if self.tts:
            try:
                self.tts.stop()
            except Exception:
                pass

        # Cleanup shared resources (LLM, VDevice, Tool)
        tool_module = self.selected_tool.get("module")
        agent_utils.cleanup_resources(
            getattr(self, 'llm', None),
            getattr(self, 'vdevice', None),
            tool_module,
            logger
        )


def main():
    """Main entry point for the voice chat agent."""
    # Parse arguments first to get log level
    parser = get_standalone_parser()
    parser.description = 'Voice-enabled AI Tool Agent'
    parser.add_argument('--no-tts', action='store_true', help='Disable TTS')

    # Handle --list-models flag before full initialization
    handle_list_models_flag(parser, AGENT_APP)

    args = parser.parse_args()

    # Initialize logging from CLI args
    init_logging(level=level_from_args(args))

    # Validate configuration
    try:
        config.validate_config()
    except ValueError as e:
        logger.error("Configuration Error: %s", e)
        return

    # Resolve LLM HEF path (Qwen2.5-Coder-1.5B-Instruct) with auto-download
    llm_hef_path = resolve_hef_path(
        hef_path=args.hef_path if args.hef_path is not None else "Qwen2.5-Coder-1.5B-Instruct",
        app_name=AGENT_APP,
        arch=HAILO10H_ARCH
    )
    if llm_hef_path is None:
        logger.error("Failed to resolve HEF path for LLM model. Exiting.")
        return

    # Resolve Whisper HEF path (Whisper-Base) with auto-download
    whisper_hef_path = resolve_hef_path(
        hef_path="Whisper-Base",
        app_name=WHISPER_CHAT_APP,
        arch=HAILO10H_ARCH
    )
    if whisper_hef_path is None:
        logger.error("Failed to resolve HEF path for Whisper model. Exiting.")
        return

    logger.info("Using LLM HEF: %s", llm_hef_path)
    logger.info("Using Whisper HEF: %s", whisper_hef_path)

    # Tool Selection
    try:
        modules = tool_discovery.discover_tool_modules(tool_dir=Path(__file__).parent)
        all_tools = tool_discovery.collect_tools(modules)
    except Exception as e:
        logger.error("Failed to discover tools: %s", e)
        logger.debug(traceback.format_exc())
        return

    if not all_tools:
        logger.error("No tools found.")
        return

    tool_thread, tool_result = tool_selection.start_tool_selection_thread(all_tools)
    selected_tool = tool_selection.get_tool_selection_result(tool_thread, tool_result)

    if not selected_tool:
        return

    tool_execution.initialize_tool_if_needed(selected_tool)

    # Start App
    try:
        app = VoiceAgentApp(
            llm_hef_path=llm_hef_path,
            selected_tool=selected_tool,
            whisper_hef_path=whisper_hef_path,
            debug=args.debug,
            no_tts=args.no_tts
        )
    except Exception:
        # Error already logged in __init__
        return

    interaction = VoiceInteractionManager(
        title="Voice-Enabled Tool Agent",
        on_audio_ready=app.on_audio_ready,
        on_processing_start=app.on_processing_start,
        on_clear_context=app.on_clear_context,
        on_shutdown=app.close,
        debug=args.debug
    )

    interaction.run()


if __name__ == "__main__":
    main()
