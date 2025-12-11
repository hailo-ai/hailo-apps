"""
Unified Agent Application.

Combines text and voice modes into a single agent with:
- Tool calling via Hailo LLM
- YAML-based configuration
- Context state management
- Voice input/output (optional)

Usage:
    # Text mode (default)
    python -m hailo_apps.python.standalone_apps.agent_tools_example.agent

    # Voice mode
    python -m hailo_apps.python.standalone_apps.agent_tools_example.agent --voice

    # Load specific context state
    python -m hailo_apps.python.standalone_apps.agent_tools_example.agent --state optimized_v2

    # Debug mode
    python -m hailo_apps.python.standalone_apps.agent_tools_example.agent --debug
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Hailo imports
from hailo_platform import VDevice
from hailo_platform.genai import LLM

# Core imports
from hailo_apps.python.core.common.core import (
    get_standalone_parser,
    handle_list_models_flag,
    resolve_hef_path,
)
from hailo_apps.python.core.common.defines import (
    AGENT_APP,
    HAILO10H_ARCH,
    SHARED_VDEVICE_GROUP_ID,
    WHISPER_CHAT_APP,
)
from hailo_apps.python.core.common.hailo_logger import (
    get_logger,
    init_logging,
    level_from_args,
)

# LLM utilities
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

# Local imports
try:
    from . import config, system_prompt
    from .state_manager import StateManager
    from .yaml_config import load_yaml_config, ToolYamlConfig
except ImportError:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    import config
    import system_prompt
    from state_manager import StateManager
    from yaml_config import load_yaml_config, ToolYamlConfig

logger = get_logger(__name__)


@dataclass
class AgentResponse:
    """
    Response from agent query processing.

    Attributes:
        tool_called: Whether a tool was called.
        tool_name: Name of the tool called (if any).
        tool_result: Result from tool execution (if any).
        text: Text response from LLM.
        raw_response: Raw LLM response before processing.
    """

    tool_called: bool = False
    tool_name: str = ""
    tool_result: Optional[Dict[str, Any]] = None
    text: str = ""
    raw_response: str = ""


class AgentApp:
    """
    Unified agent supporting text and voice modes.

    Combines the functionality of chat_agent and voice_chat_agent
    into a single application with mode switching.
    """

    def __init__(
        self,
        llm_hef_path: Path,
        selected_tool: Dict[str, Any],
        voice_enabled: bool = False,
        no_tts: bool = False,
        state_name: str = "default",
        whisper_hef_path: Optional[Path] = None,
        debug: bool = False,
    ):
        """
        Initialize the agent application.

        Args:
            llm_hef_path: Path to the LLM HEF model file.
            selected_tool: The selected tool configuration dict.
            voice_enabled: Enable voice mode.
            no_tts: Disable text-to-speech in voice mode.
            state_name: Name of context state to load.
            whisper_hef_path: Path to Whisper HEF (voice mode only).
            debug: Enable debug mode.
        """
        self.debug = debug
        self.voice_enabled = voice_enabled
        self.no_tts = no_tts
        self.selected_tool = selected_tool
        self.selected_tool_name = selected_tool.get("name", "")

        # Tool lookup for execution
        self.tools_lookup = {self.selected_tool_name: selected_tool}

        # Paths
        self.cache_dir = Path(os.path.dirname(os.path.abspath(__file__)))

        # State management
        self.state_manager = StateManager(
            tool_name=self.selected_tool_name,
            contexts_dir=self._get_contexts_dir(),
        )

        # Load YAML config if available
        self.yaml_config: Optional[ToolYamlConfig] = None
        config_path = selected_tool.get("config_path")
        if config_path and Path(config_path).exists():
            self.yaml_config = load_yaml_config(Path(config_path))
            if self.yaml_config:
                logger.info("Loaded YAML config: %s", config_path)

        print("Initializing AI components...")

        # Initialize Hailo VDevice
        try:
            if voice_enabled:
                # Shared VDevice for multiple models
                params = VDevice.create_params()
                params.group_id = SHARED_VDEVICE_GROUP_ID
                self.vdevice = VDevice(params)
            else:
                self.vdevice = VDevice()
        except Exception as e:
            logger.error("Failed to create VDevice: %s", e)
            raise

        # Initialize LLM
        try:
            self.llm = LLM(self.vdevice, str(llm_hef_path))
        except Exception as e:
            logger.error("Failed to initialize LLM: %s", e)
            self.vdevice.release()
            raise

        # Voice components (optional)
        self.s2t = None
        self.tts = None
        if voice_enabled:
            self._init_voice_components(whisper_hef_path)

        # Initialize context
        self._init_context(state_name)

        print("AI components ready!")

    def _get_contexts_dir(self) -> Path:
        """
        Get the contexts directory for the selected tool.

        Returns:
            Path to contexts directory.
        """
        # Check if tool has a package-based structure
        tool_module = self.selected_tool.get("module")
        if tool_module:
            module_path = getattr(tool_module, "__file__", None)
            if module_path:
                tool_dir = Path(module_path).parent
                contexts_dir = tool_dir / "contexts"
                if contexts_dir.exists() or (tool_dir / "config.yaml").exists():
                    return contexts_dir

        # Fallback: use cache_dir
        return self.cache_dir

    def _init_voice_components(self, whisper_hef_path: Optional[Path]) -> None:
        """
        Initialize voice input/output components.

        Args:
            whisper_hef_path: Path to Whisper HEF model.
        """
        # Import voice components
        try:
            from hailo_apps.python.core.gen_ai_utils.voice_processing.speech_to_text import (
                SpeechToTextProcessor,
            )
            from hailo_apps.python.core.gen_ai_utils.voice_processing.text_to_speech import (
                TextToSpeechProcessor,
                PiperModelNotFoundError,
            )
            from hailo_apps.python.core.gen_ai_utils.voice_processing.audio_diagnostics import (
                AudioDiagnostics,
            )
        except ImportError as e:
            logger.error("Voice processing modules not available: %s", e)
            raise

        # Speech-to-Text
        try:
            self.s2t = SpeechToTextProcessor(self.vdevice, hef_path=whisper_hef_path)
        except Exception as e:
            logger.error("Failed to initialize Speech-to-Text: %s", e)
            self.llm.release()
            self.vdevice.release()
            raise

        # Text-to-Speech
        if not self.no_tts:
            try:
                _, output_device_id = AudioDiagnostics.auto_detect_devices()
                self.tts = TextToSpeechProcessor(device_id=output_device_id)
            except PiperModelNotFoundError:
                logger.warning("Piper TTS model not found, continuing without TTS")
                self.tts = None
            except Exception as e:
                logger.warning("Failed to initialize TTS: %s", e)
                self.tts = None

    def _init_context(self, state_name: str) -> None:
        """
        Initialize the LLM context.

        Tries to load from saved state first, falls back to fresh initialization.

        Args:
            state_name: Name of state to load.
        """
        # Build system prompt (with YAML config if available)
        self.system_text = system_prompt.create_system_prompt(
            [self.selected_tool],
            yaml_config=self.yaml_config,
        )
        logger.debug("System prompt: %d chars", len(self.system_text))

        # Try to load saved state
        if self.state_manager.load_state(state_name, self.llm):
            logger.info("Loaded state: %s", state_name)
            return

        # Try legacy cache format
        try:
            if context_manager.load_context_from_cache(
                self.llm, self.selected_tool_name, self.cache_dir, logger
            ):
                logger.info("Loaded legacy cache: %s", self.selected_tool_name)
                return
        except Exception as e:
            logger.debug("Legacy cache load failed: %s", e)

        # Fresh initialization
        logger.info("Initializing fresh context")
        try:
            prompt = [message_formatter.messages_system(self.system_text)]
            context_manager.add_to_context(self.llm, prompt, logger)

            # Add few-shot examples if available
            if self.yaml_config and self.yaml_config.few_shot_examples:
                logger.info("Adding %d few-shot examples to context", len(self.yaml_config.few_shot_examples))
                system_prompt.add_few_shot_examples_to_context(
                    self.llm,
                    self.yaml_config.few_shot_examples,
                    logger,
                )

            # Save initial state
            yaml_dict = self.yaml_config.raw_config if self.yaml_config else {}
            self.state_manager.save_state("default", self.llm, yaml_dict)
        except Exception as e:
            logger.error("Failed to initialize context: %s", e)
            print(f"[Error] Failed to initialize AI context: {e}")

    def run(self) -> None:
        """
        Main entry point - runs text or voice loop.
        """
        if self.voice_enabled:
            self._run_voice_loop()
        else:
            self._run_text_loop()

    def _run_text_loop(self) -> None:
        """
        Run the text-based chat loop.
        """
        print(f"\nChat started. Type '/exit' to quit. Use '/clear' to reset context.")
        print(f"Tool in use: {self.selected_tool_name}\n")

        try:
            while True:
                print("You: ", end="", flush=True)
                try:
                    user_text = sys.stdin.readline().strip()
                except KeyboardInterrupt:
                    print("\nInterrupted. Type '/exit' to quit properly.")
                    continue

                if not user_text:
                    continue

                # Handle commands
                if user_text.lower() in {"/exit", ":q", "quit", "exit"}:
                    print("Bye.")
                    break

                if user_text.lower() == "/clear":
                    self._handle_clear_context()
                    continue

                if user_text.lower() == "/context":
                    context_manager.print_context_usage(
                        self.llm, show_always=True, logger_instance=logger
                    )
                    continue

                if user_text.lower() == "/states":
                    self._show_states()
                    continue

                # Process query
                self.process_query(user_text)

        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.close()

    def _run_voice_loop(self) -> None:
        """
        Run the voice-based interaction loop.
        """
        try:
            from hailo_apps.python.core.gen_ai_utils.voice_processing.interaction import (
                VoiceInteractionManager,
            )
        except ImportError as e:
            logger.error("Voice processing modules not available: %s", e)
            return

        interaction = VoiceInteractionManager(
            title="Voice-Enabled Tool Agent",
            on_audio_ready=self._on_voice_input,
            on_processing_start=self._on_processing_start,
            on_clear_context=self._handle_clear_context,
            on_shutdown=self.close,
            debug=self.debug,
        )

        interaction.run()

    def _on_voice_input(self, audio) -> None:
        """
        Callback when voice audio is ready.

        Args:
            audio: Recorded audio data.
        """
        if not self.s2t:
            return

        try:
            user_text = self.s2t.transcribe(audio)
        except Exception as e:
            logger.error("Transcription failed: %s", e)
            return

        if not user_text:
            print("No speech detected.")
            return

        print(f"\nYou: {user_text}")
        self.process_query(user_text, stream_to_tts=True)

    def _on_processing_start(self) -> None:
        """Callback when processing starts - interrupt any ongoing TTS."""
        if self.tts:
            try:
                self.tts.interrupt()
            except Exception as e:
                logger.debug("TTS interrupt failed: %s", e)

    def _handle_clear_context(self) -> None:
        """Handle context clear request."""
        try:
            self.llm.clear_context()
            print("[Info] Context cleared.")

            # Reload from state
            if self.state_manager.reload_state(self.llm):
                logger.debug("Context restored from state")
            elif context_manager.load_context_from_cache(
                self.llm, self.selected_tool_name, self.cache_dir, logger
            ):
                logger.debug("Context restored from legacy cache")
            else:
                logger.warning("Could not restore context")
        except Exception as e:
            print(f"[Error] Failed to clear context: {e}")

    def _show_states(self) -> None:
        """Show available context states."""
        states = self.state_manager.list_states()
        if not states:
            print("[Info] No saved states found.")
            return

        print("\n[Info] Available states:")
        for state in states:
            current = " (current)" if state.state_name == self.state_manager.current_state else ""
            print(f"  - {state.state_name}: {state.context_tokens} tokens{current}")
        print()

    def process_query(
        self,
        user_text: str,
        stream_to_tts: bool = False,
    ) -> AgentResponse:
        """
        Process a user query.

        Args:
            user_text: The user's query text.
            stream_to_tts: Stream response to TTS (voice mode).

        Returns:
            AgentResponse with results.
        """
        # Check context capacity
        if context_manager.is_context_full(
            self.llm, context_threshold=config.CONTEXT_THRESHOLD, logger_instance=logger
        ):
            logger.info("Context full, reloading state...")
            self.state_manager.reload_state(self.llm)

        prompt = [message_formatter.messages_user(user_text)]
        logger.debug("User message: %s", json.dumps(prompt, ensure_ascii=False))

        # Setup TTS callback if needed
        tts_callback = None
        tts_state = {"sentence_buffer": "", "gen_id": None}

        if stream_to_tts and self.tts:
            self.tts.clear_interruption()
            tts_state["gen_id"] = self.tts.get_current_gen_id()
            tts_callback = self._create_tts_callback(tts_state)

        # Generate response
        try:
            is_debug = logger.isEnabledFor(logging.DEBUG)
            raw_response = streaming.generate_and_stream_response(
                llm=self.llm,
                prompt=prompt,
                temperature=config.TEMPERATURE,
                seed=config.SEED,
                max_tokens=config.MAX_GENERATED_TOKENS,
                prefix="Assistant: ",
                debug_mode=is_debug,
                token_callback=tts_callback,
            )
            logger.debug("Raw response length: %d, content: %s", len(raw_response), raw_response[:200])
        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            logger.debug("Traceback: %s", traceback.format_exc())
            return AgentResponse(text=f"Error: {e}", raw_response="")

        # Flush remaining TTS buffer
        if self.tts and tts_state["sentence_buffer"].strip():
            self.tts.queue_text(tts_state["sentence_buffer"].strip(), tts_state["gen_id"])

        # Check for tool calls
        tool_call = tool_parsing.parse_function_call(raw_response)
        if tool_call is None:
            logger.debug("Direct response (no tool)")
            return AgentResponse(
                tool_called=False,
                text=raw_response,
                raw_response=raw_response,
            )

        # Execute tool
        result = tool_execution.execute_tool_call(tool_call, self.tools_lookup)
        tool_execution.print_tool_result(result)

        # TTS for tool result
        if self.tts:
            if result.get("ok"):
                self.tts.queue_text(str(result.get("result", "")))
            else:
                self.tts.queue_text("There was an error executing the tool.")

        # Update context with tool result
        agent_utils.update_context_with_tool_result(self.llm, result, logger)

        return AgentResponse(
            tool_called=True,
            tool_name=tool_call.get("name", ""),
            tool_result=result,
            raw_response=raw_response,
        )

    def _create_tts_callback(self, state: Dict[str, Any]) -> Callable[[str], None]:
        """
        Create a TTS callback function.

        Args:
            state: Mutable state dict for the callback.

        Returns:
            Callback function.
        """

        def tts_callback(chunk: str) -> None:
            if not self.tts:
                return

            state["sentence_buffer"] += chunk
            old_len = len(state["sentence_buffer"])
            state["sentence_buffer"] = self.tts.chunk_and_queue(
                state["sentence_buffer"],
                state["gen_id"],
                state.get("first_chunk_sent", False) is False,
            )
            if len(state["sentence_buffer"]) < old_len:
                state["first_chunk_sent"] = True

        return tts_callback

    def close(self) -> None:
        """Clean up resources."""
        if self.tts:
            try:
                self.tts.stop()
            except Exception:
                pass

        tool_module = self.selected_tool.get("module")
        agent_utils.cleanup_resources(
            getattr(self, "llm", None),
            getattr(self, "vdevice", None),
            tool_module,
            logger,
        )


def create_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for the unified agent.

    Returns:
        Configured ArgumentParser.
    """
    parser = get_standalone_parser()
    parser.description = "Unified AI Tool Agent (Text + Voice)"

    parser.add_argument(
        "--voice",
        action="store_true",
        help="Enable voice mode (speech input/output)",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable text-to-speech in voice mode",
    )
    parser.add_argument(
        "--state",
        type=str,
        default="default",
        help="Context state to load (default: 'default')",
    )
    parser.add_argument(
        "--tool",
        "-t",
        type=str,
        default=None,
        help="Tool to use (skips interactive selection). Use tool name (e.g., 'math', 'weather').",
    )

    return parser


def main() -> None:
    """Main entry point for the unified agent."""
    parser = create_parser()

    # Handle --list-models flag
    handle_list_models_flag(parser, AGENT_APP)

    args = parser.parse_args()

    # Initialize logging
    init_logging(level=level_from_args(args))

    # Validate configuration
    try:
        config.validate_config()
    except ValueError as e:
        logger.error("Configuration Error: %s", e)
        sys.exit(1)

    # Resolve LLM HEF path
    llm_hef_path = resolve_hef_path(
        hef_path=args.hef_path if args.hef_path else "Qwen2.5-Coder-1.5B-Instruct",
        app_name=AGENT_APP,
        arch=HAILO10H_ARCH,
    )
    if llm_hef_path is None:
        logger.error("Failed to resolve HEF path for LLM model.")
        sys.exit(1)

    # Resolve Whisper HEF path (voice mode)
    whisper_hef_path = None
    if args.voice:
        whisper_hef_path = resolve_hef_path(
            hef_path="Whisper-Base",
            app_name=WHISPER_CHAT_APP,
            arch=HAILO10H_ARCH,
        )
        if whisper_hef_path is None:
            logger.error("Failed to resolve HEF path for Whisper model.")
            sys.exit(1)

    logger.info("Using LLM HEF: %s", llm_hef_path)
    if whisper_hef_path:
        logger.info("Using Whisper HEF: %s", whisper_hef_path)

    # Discover tools
    try:
        modules = tool_discovery.discover_tool_modules(tool_dir=Path(__file__).parent)
        all_tools = tool_discovery.collect_tools(modules)
    except Exception as e:
        logger.error("Failed to discover tools: %s", e)
        logger.debug(traceback.format_exc())
        sys.exit(1)

    if not all_tools:
        logger.error("No tools found. Add tool modules or packages.")
        sys.exit(1)

    # Tool selection
    selected_tool = None

    if args.tool:
        # Find tool by name
        tool_name = args.tool.lower().strip()
        for tool in all_tools:
            if tool.get("name", "").lower() == tool_name:
                selected_tool = tool
                break
        if not selected_tool:
            available = ", ".join(t.get("name", "") for t in all_tools)
            logger.error("Tool '%s' not found. Available: %s", args.tool, available)
            sys.exit(1)
    else:
        # Interactive selection
        tool_thread, tool_result = tool_selection.start_tool_selection_thread(all_tools)
        selected_tool = tool_selection.get_tool_selection_result(tool_thread, tool_result)

    if not selected_tool:
        sys.exit(0)

    # Initialize tool
    tool_execution.initialize_tool_if_needed(selected_tool)

    # Start agent
    try:
        app = AgentApp(
            llm_hef_path=llm_hef_path,
            selected_tool=selected_tool,
            voice_enabled=args.voice,
            no_tts=args.no_tts,
            state_name=args.state,
            whisper_hef_path=whisper_hef_path,
            debug=args.debug,
        )
        app.run()
    except Exception as e:
        logger.error("Agent failed: %s", e)
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

