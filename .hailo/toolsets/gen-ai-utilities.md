# Toolset: Gen AI Utilities Reference

> API reference for shared gen AI utilities: LLM tools, voice processing, and agent support.

## LLM Utilities (`gen_ai_utils/llm_utils/`)

### streaming.py — Token Streaming
```python
from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils.streaming import (
    StreamingTextFilter,          # Stateful filter: strips <text>/<tool_call>/<tool_response> tags from a token stream
    generate_and_stream_response, # Drive llm.generate(), stream to stdout, return raw response string
    clean_response,               # Strip special tokens / XML wrapper tags from a full response string
)

# generate_and_stream_response(llm, prompt, temperature=0.1, seed=42, max_tokens=200,
#                               prefix="Assistant: ", token_callback=None, abort_callback=None,
#                               show_raw_stream=True) -> str (raw response, for tool-call parsing)
raw_response = generate_and_stream_response(llm, prompt=[{"role": "user", "content": "Hi"}])
```

### tool_parsing.py — Parse Tool Calls from LLM Output
```python
from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils.tool_parsing import (
    parse_function_call,    # Extract {"name": ..., "arguments": {...}} from <tool_call>...</tool_call>
    validate_and_fix_call,  # Validate a parsed call dict, coercing stringified JSON arguments
)

call = parse_function_call(raw_response)   # -> dict | None
if call:
    call = validate_and_fix_call(call)     # -> dict | None
```

### tool_execution.py — Execute Parsed Tool Calls
```python
from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils.tool_execution import (
    initialize_tool_if_needed,  # Call tool_module.initialize_tool() if present
    execute_tool_call,          # (tool_call: dict, tools_lookup: dict) -> result dict {"ok": bool, ...}
    print_tool_result,          # Pretty-print a result dict to the user
)
```

### tool_discovery.py — Auto-Discover Tools from `tools/` Subdirectory
```python
from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils.tool_discovery import (
    discover_tool_modules,  # (tool_dir: Path | None) -> list[ModuleType], scans <tool_dir>/tools/
    collect_tools,          # (modules: list[ModuleType]) -> list[dict], extracts name/description/schema/runner
)

modules = discover_tool_modules(Path(__file__).parent)
tools = collect_tools(modules)
```

### tool_selection.py — Interactive Tool Selection (Background Thread)
```python
from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils.tool_selection import (
    wait_for_tool_selection,       # (all_tools: list[dict]) -> dict | None — starts thread + blocks for result
    start_tool_selection_thread,   # (all_tools) -> (Thread, result_dict) — non-blocking variant
    get_tool_selection_result,     # (thread, result_dict) -> dict | None — join + extract result
)

selected = wait_for_tool_selection(tools)
```

### context_manager.py — Conversation Context (module-level functions, no class)
```python
from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils.context_manager import (
    is_context_full,          # (llm, context_threshold=0.95) -> bool
    print_context_usage,      # (llm, show_always=False) -> None
    add_to_context,           # (llm, prompt: list) -> bool — appends messages via a 1-token generate()
    save_context_to_cache,    # (llm, tool_name, cache_dir: Path) -> bool
    load_context_from_cache,  # (llm, tool_name, cache_dir: Path) -> bool
)
```

### message_formatter.py — Format Messages for LLM
```python
from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils.message_formatter import (
    messages_system,    # (text: str) -> {"role": "system", "content": text}
    messages_user,      # (text: str) -> {"role": "user", "content": text}
    messages_assistant,  # (text: str) -> {"role": "assistant", "content": text}
    messages_tool,       # (text: str) -> {"role": "tool", "content": text}
)
```

### agent_utils.py — Agent Loop Helpers
```python
from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils.agent_utils import (
    update_context_with_tool_result,  # (llm, result: dict) -> None — wraps result in <tool_response> and adds to context
    cleanup_resources,                 # (llm, vdevice, tool_module=None) -> None — releases LLM/VDevice, calls tool_module.cleanup_tool()
)
```
There is no `run_agent_loop()` — compose the agent loop yourself from `generate_and_stream_response`,
`parse_function_call`, `execute_tool_call`, and `update_context_with_tool_result`.

### terminal_ui.py — Terminal UI Components
```python
from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils.terminal_ui import TerminalUI

TerminalUI.show_banner(title="My App", controls={"SPACE": "record", "Q": "quit"})
ch = TerminalUI.get_char()  # blocking single-character read (handles Windows/POSIX + non-TTY)
```

---

## Voice Processing (`gen_ai_utils/voice_processing/`)

### speech_to_text.py
```python
from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.speech_to_text import SpeechToTextProcessor

stt = SpeechToTextProcessor(vdevice, hef_path)  # hef_path=None auto-resolves the Whisper-Base HEF
text = stt.transcribe(audio_numpy_array, language="en", timeout_ms=15000)
# No release()/close() method — SpeechToTextProcessor wraps hailo_platform.genai.Speech2Text directly.
```

### text_to_speech.py
```python
from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.text_to_speech import TextToSpeechProcessor

tts = TextToSpeechProcessor()          # onnx_path defaults to the bundled Piper model
tts.queue_text("Hello, the scene is quiet and peaceful.")  # async: chunks are queued and spoken by a worker thread
tts.wait_for_completion(timeout=30.0)  # block until the queue drains
tts.interrupt()                        # stop current speech + clear queue (increments generation id)
tts.stop()                             # shut down the background worker
```
Uses **Piper TTS** (CPU-based, lightweight). Models stored in `local_resources/piper_models/`.
There is no `speak()` method — use `queue_text()`.

### audio_recorder.py
```python
from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.audio_recorder import AudioRecorder

recorder = AudioRecorder(device_id=None, debug=False)  # device_id=None auto-selects preferred mic
recorder.start()                 # begin capturing in the background
# ... wait / do other work ...
audio_data = recorder.stop()     # -> np.ndarray of everything captured since start()
recorder.close()                 # release the underlying sounddevice stream
```
There is no `sample_rate` constructor arg and no `record(duration=...)` method.

### audio_player.py
```python
from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.audio_player import AudioPlayer

player = AudioPlayer(device_id=None)
player.play(audio_numpy_array, block=False)  # queues audio; no sample_rate kwarg (always TARGET_SR internally)
player.stop()    # clears the queue
player.close()   # shuts down the persistent playback stream
```

### vad.py — Voice Activity Detection
```python
from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.vad import VoiceActivityDetector, add_vad_args

# sample_rate and chunk_size are required (no defaults); webrtcvad needs 8/16/32/48kHz.
vad = VoiceActivityDetector(sample_rate=16000, chunk_size=1024, aggressiveness=3, energy_threshold=0.05)
is_speech, energy = vad.process(audio_chunk)   # NOT is_speech(chunk) — returns a tuple

# CLI args helper
add_vad_args(parser)  # Adds --vad, --vad-aggressiveness, --vad-energy-threshold
```
The class is `VoiceActivityDetector`, not `VAD`. The method is `process()`, not `is_speech()`.

### interaction.py — High-Level Voice Interaction (callback-driven, not request/response)
```python
from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.interaction import VoiceInteractionManager

# Constructor takes callbacks, not vdevice/whisper_hef_path — STT/LLM calls happen inside your own
# on_audio_ready callback, which VoiceInteractionManager invokes with the recorded np.ndarray.
vim = VoiceInteractionManager(
    title="My Voice App",
    on_audio_ready=lambda audio: handle_audio(audio),   # called with recorded np.ndarray
    on_clear_context=lambda: clear_context(),
    on_shutdown=lambda: shutdown(),
    vad_enabled=True,
    vad_aggressiveness=3,
    tts=tts,  # optional TextToSpeechProcessor, used to auto-inhibit VAD while speaking
)
vim.run()      # blocking loop: SPACE to record, Q to quit, C to clear context (or hands-free via VAD)
vim.close()    # release recorder/VAD resources
```
There is no `listen()`/`speak()`/`cleanup()` — it's an event loop (`run()`) driven by your callbacks,
not a per-turn `listen()`/`speak()` API.

### audio_diagnostics.py / audio_troubleshoot.py
```python
from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.audio_diagnostics import (
    check_voice_dependencies,  # () -> None, exits(1) with install instructions if sounddevice/numpy missing
    AudioDiagnostics,          # class: list_audio_devices(), get_preferred_devices(), test_microphone(), test_speaker(), ...
)

# audio_troubleshoot.py is a CLI tool (argparse-based), not a library with a zero-arg entry point:
#   python3 -m hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.audio_troubleshoot --diagnose
# Its run_diagnostics(args)/run_device_selection(args)/run_auto_configure(args) all require the
# parsed argparse.Namespace — don't call them directly with no arguments.
```

---

## Agent Tools Framework (`agent_tools_example/tools/`)

### Tool Base Class
```python
from hailo_apps.python.gen_ai_apps.agent_tools_example.tools.base import (
    BaseTool,       # Abstract base class for tools
    ToolResult,     # Standardized result: ToolResult.success(data) / ToolResult.failure(msg)
    ToolConfig,     # YAML-loaded tool configuration dataclass
)
```

### Tool Structure
Each tool lives in its own directory:
```
tools/{tool_name}/
├── __init__.py
├── tool.py         # Implements BaseTool or has module-level: name, description, schema, TOOLS_SCHEMA, run
└── config.yaml     # Tool configuration (optional)
```

### YAML Configuration
```python
from hailo_apps.python.gen_ai_apps.agent_tools_example.yaml_config import (
    load_yaml_config,   # (config_path: Path) -> ToolYamlConfig | None
    find_tool_config,   # (tool_dir: Path) -> Path | None
    ToolYamlConfig,     # Parsed config object
)
```

### State Management
```python
from hailo_apps.python.gen_ai_apps.agent_tools_example.state_manager import StateManager

sm = StateManager(tool_name="my_tool", contexts_dir=Path("./contexts"))
sm.save_state("default", llm, yaml_config=cfg_dict)   # llm is required (saves its context)
sm.load_state("default", llm)                          # llm is required (loads context into it)
```


