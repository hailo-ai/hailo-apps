# Agent Tools Infrastructure Update - Specification

> **Version**: 1.0
> **Date**: 2025-12-11
> **Status**: Planning

---

## 1. Overview

This specification describes a comprehensive update to the agent tools infrastructure:

- **Unified Agent**: Merge `chat_agent.py` and `voice_chat_agent.py` into single `agent.py`
- **YAML Configuration**: Per-tool YAML configs for prompts, few-shot examples, test cases
- **Context State Management**: Save/load LLM context states with YAML snapshots for reproducibility
- **Testing Framework**: Benchmarks, accuracy metrics, interaction collection
- **Transfer Learning**: Use strong models to optimize prompts for on-device model

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACE                                │
│                    (Text CLI or Voice Interaction)                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          agent.py (Unified)                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ Text Mode   │  │ Voice Mode  │  │ State Mgr   │  │ Tool Loader │    │
│  │ (stdin)     │  │ (S2T/TTS)   │  │ (save/load) │  │ (discovery) │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
          ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
          │ tools/math/ │  │tools/weather│  │tools/rgb_led│  ...
          │  tool.py    │  │  tool.py    │  │  tool.py    │
          │  config.yaml│  │  config.yaml│  │  config.yaml│
          │  contexts/  │  │  contexts/  │  │  hardware.py│
          └─────────────┘  └─────────────┘  └─────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    gen_ai_utils (Shared Infrastructure)                 │
│  ┌──────────────────────────────┐  ┌──────────────────────────────┐    │
│  │        llm_utils/            │  │     voice_processing/        │    │
│  │  • message_formatter.py      │  │  • interaction.py            │    │
│  │  • context_manager.py        │  │  • speech_to_text.py         │    │
│  │  • streaming.py              │  │  • text_to_speech.py         │    │
│  │  • tool_discovery.py         │  │  • audio_recorder.py         │    │
│  │  • tool_execution.py         │  │  • audio_player.py           │    │
│  └──────────────────────────────┘  └──────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Infrastructure File References

### 3.1 LLM Utilities (`gen_ai_utils/llm_utils/`)

| File                                                                        | Key Functions                                                                                   | Purpose                                   |
| --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ----------------------------------------- |
| [`message_formatter.py`](../../gen_ai_utils/llm_utils/message_formatter.py) | `messages_system()`, `messages_user()`, `messages_assistant()`, `messages_tool()`               | Format messages for Hailo LLM API         |
| [`context_manager.py`](../../gen_ai_utils/llm_utils/context_manager.py)     | `save_context_to_cache()`, `load_context_from_cache()`, `add_to_context()`, `is_context_full()` | Manage LLM context window and caching     |
| [`streaming.py`](../../gen_ai_utils/llm_utils/streaming.py)                 | `StreamingTextFilter`, `generate_and_stream_response()`                                         | Stream LLM output, filter XML tags        |
| [`tool_discovery.py`](../../gen_ai_utils/llm_utils/tool_discovery.py)       | `discover_tool_modules()`, `collect_tools()`                                                    | Auto-discover tool modules                |
| [`tool_execution.py`](../../gen_ai_utils/llm_utils/tool_execution.py)       | `execute_tool_call()`, `initialize_tool_if_needed()`                                            | Execute tool calls from LLM               |
| [`tool_parsing.py`](../../gen_ai_utils/llm_utils/tool_parsing.py)           | `parse_function_call()`                                                                         | Parse `<tool_call>` XML from LLM response |
| [`tool_selection.py`](../../gen_ai_utils/llm_utils/tool_selection.py)       | `start_tool_selection_thread()`, `get_tool_selection_result()`                                  | Interactive tool selection UI             |
| [`agent_utils.py`](../../gen_ai_utils/llm_utils/agent_utils.py)             | `update_context_with_tool_result()`, `cleanup_resources()`                                      | Agent helper utilities                    |
| [`terminal_ui.py`](../../gen_ai_utils/llm_utils/terminal_ui.py)             | `TerminalUI.show_banner()`, `TerminalUI.get_char()`                                             | Terminal UI helpers                       |

### 3.2 Voice Processing (`gen_ai_utils/voice_processing/`)

| File                                                                               | Key Classes               | Purpose                                                               |
| ---------------------------------------------------------------------------------- | ------------------------- | --------------------------------------------------------------------- |
| [`interaction.py`](../../gen_ai_utils/voice_processing/interaction.py)             | `VoiceInteractionManager` | Callback-based voice interaction loop (SPACE=record, Q=quit, C=clear) |
| [`speech_to_text.py`](../../gen_ai_utils/voice_processing/speech_to_text.py)       | `SpeechToTextProcessor`   | Hailo Whisper model wrapper                                           |
| [`text_to_speech.py`](../../gen_ai_utils/voice_processing/text_to_speech.py)       | `TextToSpeechProcessor`   | Piper TTS with streaming playback                                     |
| [`audio_recorder.py`](../../gen_ai_utils/voice_processing/audio_recorder.py)       | `AudioRecorder`           | Microphone recording with auto-detection                              |
| [`audio_player.py`](../../gen_ai_utils/voice_processing/audio_player.py)           | `AudioPlayer`             | Cross-platform audio playback                                         |
| [`audio_diagnostics.py`](../../gen_ai_utils/voice_processing/audio_diagnostics.py) | `AudioDiagnostics`        | Device enumeration and troubleshooting                                |

### 3.3 Current Agent Files (to be refactored)

| File                                                | Purpose               | Target                             |
| --------------------------------------------------- | --------------------- | ---------------------------------- |
| [`chat_agent.py`](../chat_agent.py)                 | Text-only agent       | → `agent.py`                       |
| [`voice_chat_agent.py`](../voice_chat_agent.py)     | Voice-enabled agent   | → `agent.py`                       |
| [`config.py`](../config.py)                         | Python configuration  | → `agent_config.yaml`              |
| [`system_prompt.py`](../system_prompt.py)           | System prompt builder | → Updated to use YAML              |
| [`tool_math.py`](../tool_math.py)                   | Math tool             | → `tools/math/tool.py`             |
| [`tool_weather.py`](../tool_weather.py)             | Weather tool          | → `tools/weather/tool.py`          |
| [`tool_rgb_led.py`](../tool_rgb_led.py)             | LED tool              | → `tools/rgb_led/tool.py`          |
| [`tool_servo.py`](../tool_servo.py)                 | Servo tool            | → `tools/servo/tool.py`            |
| [`tool_elevator.py`](../tool_elevator.py)           | Elevator tool         | → `tools/elevator/tool.py`         |
| [`hardware_interface.py`](../hardware_interface.py) | LED/Servo hardware    | → Split into `tools/*/hardware.py` |
| [`elevator_interface.py`](../elevator_interface.py) | Elevator interface    | → `tools/elevator/interface.py`    |
| [`weather_api_utils.py`](../weather_api_utils.py)   | Weather API client    | → `tools/weather/api.py`           |

---

## 4. Unified Agent Application

### 4.1 CLI Interface

```bash
# Text mode (default)
python -m hailo_apps.python.gen_ai_apps.agent_tools_example.agent

# Voice mode
python -m hailo_apps.python.gen_ai_apps.agent_tools_example.agent --voice

# Voice without TTS
python -m hailo_apps.python.gen_ai_apps.agent_tools_example.agent --voice --no-tts

# Load specific context state
python -m hailo_apps.python.gen_ai_apps.agent_tools_example.agent --tool math --state optimized_v2

# Skip cached states and rebuild context from scratch
python -m hailo_apps.python.gen_ai_apps.agent_tools_example.agent --no-cache

# Debug mode
python -m hailo_apps.python.gen_ai_apps.agent_tools_example.agent --debug
```

### 4.2 AgentApp Architecture

```python
class AgentApp:
    """Unified agent supporting text and voice modes."""

    def __init__(
        self,
        tool_name: str,
        state_name: str = "default",
        voice_enabled: bool = False,
        no_tts: bool = False,
    ):
        # Core components
        self.tool = self._load_tool(tool_name)
        self.yaml_config = self._load_yaml_config(tool_name)
        self.state_manager = StateManager(tool_name)

        # LLM initialization
        self.vdevice = VDevice()
        self.llm = LLM(self.vdevice, hef_path)

        # Load or initialize context
        if not self.state_manager.load_state(state_name, self.llm):
            self._initialize_context()
            self.state_manager.save_state("default", self.llm, self.yaml_config)

        # Voice components (optional)
        if voice_enabled:
            self.s2t = SpeechToTextProcessor(self.vdevice)
            self.tts = None if no_tts else TextToSpeechProcessor()

    def run(self):
        """Main entry point - text or voice loop."""
        if self.voice_enabled:
            VoiceInteractionManager(
                title="Tool Agent",
                on_audio_ready=self._on_voice_input,
                on_clear_context=self._on_clear,
                on_shutdown=self.close,
            ).run()
        else:
            self._text_loop()

    def process_query(self, user_text: str) -> AgentResponse:
        """Core query processing - shared by text and voice."""
        # 1. Reload state for fresh context (each query starts fresh)
            self.state_manager.reload_state(self.llm)

        # 2. Add user message to context
        prompt = [message_formatter.messages_user(user_text)]

        # 3. Generate response with streaming
        raw_response = streaming.generate_and_stream_response(
            llm=self.llm,
            prompt=prompt,
            token_callback=self._on_token if self.tts else None,
        )

        # 4. Parse and execute tool call if present
        tool_call = tool_parsing.parse_function_call(raw_response)
        if tool_call:
            result = tool_execution.execute_tool_call(tool_call, self.tools_lookup)
            # Note: Context is not updated with tool results (fresh mode)
            return AgentResponse(tool_called=True, result=result)

        return AgentResponse(tool_called=False, text=raw_response)
```

---

## 5. Per-Tool Directory Structure

Each tool is a self-contained Python package:

```
tools/
├── __init__.py                 # Tool discovery (scans subdirectories)
├── base.py                     # BaseTool ABC (optional)
│
├── math/                       # ─── Math Tool Package ───
│   ├── __init__.py             # Re-exports: name, schema, run, TOOLS_SCHEMA
│   ├── tool.py                 # Implementation (from tool_math.py)
│   ├── config.yaml             # Prompts, few-shot examples, test cases
│   └── contexts/               # Saved context states
│       ├── default.state       # Binary LLM context
│       ├── default.yaml        # YAML config snapshot
│       └── default.meta.json   # Performance metrics
│
├── weather/                    # ─── Weather Tool Package ───
│   ├── __init__.py
│   ├── tool.py                 # (from tool_weather.py)
│   ├── api.py                  # (from weather_api_utils.py)
│   ├── config.yaml
│   └── contexts/
│
├── rgb_led/                    # ─── RGB LED Tool Package ───
│   ├── __init__.py
│   ├── tool.py                 # (from tool_rgb_led.py)
│   ├── hardware.py             # Real hardware driver
│   ├── simulator.py            # Browser-based simulator
│   ├── config.yaml
│   └── contexts/
│
├── servo/                      # ─── Servo Tool Package ───
│   ├── __init__.py
│   ├── tool.py                 # (from tool_servo.py)
│   ├── hardware.py             # PWM hardware driver
│   ├── simulator.py            # Browser simulator
│   ├── config.yaml
│   └── contexts/
│
├── elevator/                   # ─── Elevator Tool Package ───
│   ├── __init__.py
│   ├── tool.py                 # (from tool_elevator.py)
│   ├── interface.py            # (from elevator_interface.py)
│   ├── config.yaml
│   └── contexts/
│
└── _template/                  # ─── Template for new tools ───
    ├── __init__.py
    ├── tool.py                 # Skeleton implementation
    ├── config.yaml             # Example YAML structure
    └── README.md               # How to create a new tool
```

---

## 6. YAML Configuration Format

### 6.1 Tool Configuration (`tools/*/config.yaml`)

```yaml
# tools/math/config.yaml
version: "1.0"
tool_name: "math"  # Must match tool.py name

# ═══════════════════════════════════════════════════════════════
# PERSONA - Defines the assistant's role and behavior
# ═══════════════════════════════════════════════════════════════
persona:
  role: "Mathematical Calculator Assistant"
  style: "Concise, accurate, tool-focused"
  constraints:
    - "NEVER calculate math directly - ALWAYS use the math tool"
    - "ALWAYS call the tool before responding with a number"
    - "For greetings and non-math queries, respond directly without tools"

# ═══════════════════════════════════════════════════════════════
# CAPABILITIES - What this agent can do
# ═══════════════════════════════════════════════════════════════
capabilities:
  - "Evaluate mathematical expressions"
  - "Support operators: +, -, *, /, **, //, %"
  - "Handle parentheses and operator precedence"
  - "Process negative numbers"

# ═══════════════════════════════════════════════════════════════
# OUTPUT FORMAT - Response formatting rules
# ═══════════════════════════════════════════════════════════════
output_format:
  tool_call: "XML-wrapped JSON: <tool_call>{...}</tool_call>"
  direct_response: "Brief, factual answers"
  after_tool: "State the result concisely"

# ═══════════════════════════════════════════════════════════════
# TOOL INSTRUCTIONS - Detailed LLM instructions (replaces Python description)
# ═══════════════════════════════════════════════════════════════
tool_instructions: |
  CRITICAL RULE: You MUST use this tool for ALL arithmetic operations.
  The function name is 'math' (use this exact name in tool calls).

  Pass any mathematical expression as a string in the 'expression' parameter.

  Examples of valid expressions:
  - Simple: '5 + 3', '10 / 2', '7 * 8'
  - Complex: '2 - 3 * (2 + 3) / 2'
  - With negatives: '-5 + 3 * -2'
  - Powers: '2 ** 10'

# ═══════════════════════════════════════════════════════════════
# FEW-SHOT EXAMPLES - Added to context for priming
# ═══════════════════════════════════════════════════════════════
few_shot_examples:
  - user: "What is 5 plus 3?"
    tool_call:
      name: "math"
      arguments:
        expression: "5 + 3"

  - user: "Calculate 2 times 3 plus 4"
    tool_call:
      name: "math"
      arguments:
        expression: "2 * 3 + 4"

  - user: "Hello, how are you?"

# ═══════════════════════════════════════════════════════════════
# TEST CASES - For benchmarking (not added to context)
# ═══════════════════════════════════════════════════════════════
test_cases:
  - id: "math_001"
    input: "what is 10 divided by 2"
    expected:
      tool_called: true
      tool_name: "math"
      result_contains: "5"
    tags: ["basic", "division"]
    difficulty: "easy"

  - id: "math_002"
    input: "calculate (10 + 5) * 2 - 8 / 4"
    expected:
      tool_called: true
      tool_name: "math"
      result_contains: "28"
    tags: ["complex", "precedence"]
    difficulty: "medium"

  - id: "math_003"
    input: "hello, how are you?"
    expected:
      tool_called: false
    tags: ["greeting", "no-tool"]
    difficulty: "easy"

# ═══════════════════════════════════════════════════════════════
# METADATA
# ═══════════════════════════════════════════════════════════════
metadata:
  author: "hailo-team"
  created: "2025-12-11"
  description: "Math tool configuration with optimized prompts"
```

### 6.2 Base Agent Configuration (`agent_config.yaml`)

```yaml
# agent_config.yaml
version: "1.0"

llm:
  default_model: "Qwen2.5-Coder-1.5B-Instruct"
  temperature: 0.1
  max_tokens: 200
  seed: 42

context:
  threshold: 0.95           # Clear at 95% capacity
  # Note: Currently all queries reload state (fresh mode). Continue mode not yet implemented.

voice:
  whisper_model: "Whisper-Base"
  enable_tts: true
  tts_model: "en_US-amy-low"

hardware:
  mode: "simulator"         # "real" or "simulator"
  led_spi_bus: 0
  servo_pwm_channel: 0
```

---

## 7. Context State Management

### 7.1 How Context Building Works

```
┌─────────────────────────────────────────────────────────────────┐
│                      SYSTEM PROMPT                              │
│  Built from YAML: persona + capabilities + output_format +     │
│  tool_instructions + tool schema (from Python)                  │
├─────────────────────────────────────────────────────────────────┤
│  message_formatter.messages_system(system_text)                 │
│  context_manager.add_to_context(llm, [system_msg])              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    FEW-SHOT EXAMPLES                            │
│  From YAML: few_shot_examples[] converted to messages           │
├─────────────────────────────────────────────────────────────────┤
│  For each example:                                              │
│    message_formatter.messages_user(example.user)                │
│    message_formatter.messages_assistant(tool_call_xml)          │
│    # Note: tool_response and final_response are NOT added here  │
│    # They will be used when continue mode is implemented        │
│  context_manager.add_to_context(llm, [msg])                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    SAVE STATE                                   │
│  Binary context + YAML snapshot + metrics                       │
├─────────────────────────────────────────────────────────────────┤
│  state_manager.save_state("default", llm, yaml_config)          │
│    → contexts/default.state      (binary LLM context)           │
│    → contexts/default.yaml       (YAML config snapshot)         │
│    → contexts/default.meta.json  (metrics: tokens, timestamp)   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    RUNTIME                                      │
│  Load state, process queries (fresh context each query)        │
│  Note: Continue mode not yet implemented                       │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Continue Mode (Future Implementation)

**Status**: Not yet implemented. Currently, all queries reload the saved context state (fresh mode).

**What will be needed for continue mode**:

1. **Few-shot examples with full conversation**:
   - Add `tool_response` and `final_response` fields to YAML config `few_shot_examples` entries
   - Update `prepare_few_shot_examples_messages()` to include `tool_response` and `final_response` when continue mode is enabled
   - This will show the model complete conversation examples including tool results
   - Example YAML format:
     ```yaml
     few_shot_examples:
       - user: "What is 5 plus 3?"
         tool_call:
           name: "math"
           arguments:
             expression: "5 + 3"
         tool_response: '{"ok": true, "result": "5 + 3 = 8"}'
         final_response: "8"
     ```

2. **Context accumulation**:
   - In `process_query()`, add user messages, tool calls, tool results, and assistant responses to context
   - Use `agent_utils.update_context_with_tool_result()` to add tool results
   - Add assistant responses to context after generation

3. **Context management**:
   - Check context capacity before each query
   - When context is full, reload state (reset to initial context)
   - Optionally save conversation snapshots periodically

4. **CLI flag**:
   - Add `--continue` flag to enable conversation history
   - Default behavior remains fresh mode (reload state each query)

5. **State management**:
   - Optionally save conversation states with different names
   - Allow loading conversation snapshots

**Benefits of continue mode**:
- Multi-turn conversations with context awareness
- Follow-up questions ("What was the result again?")
- Complex workflows requiring multiple tool calls
- Better user experience for interactive sessions

**Current behavior (fresh mode)**:
- Each query starts from the saved context state
- No conversation history between queries
- Consistent, reproducible behavior
- Lower context usage

### 7.3 StateManager API

```python
class StateManager:
    """Manages context states for a tool."""

    def __init__(self, tool_name: str, contexts_dir: Path = None):
        self.tool_name = tool_name
        self.contexts_dir = contexts_dir or Path(f"tools/{tool_name}/contexts")

    def save_state(
        self,
        state_name: str,
        llm: LLM,
        yaml_config: dict,
        metrics: dict = None,
    ) -> bool:
        """
        Save LLM context with YAML snapshot and optional metrics.

        Creates:
          - {state_name}.state      - Binary LLM context
          - {state_name}.yaml       - YAML config snapshot (for reproducibility)
          - {state_name}.meta.json  - Metadata and performance metrics
        """
        ...

    def load_state(self, state_name: str, llm: LLM) -> bool:
        """Load a saved context state into the LLM."""
        ...

    def list_states(self) -> List[StateInfo]:
        """List all saved states with their metrics."""
        ...

    def get_best_state(self, metric: str = "accuracy") -> str:
        """Get the state name with best performance on given metric."""
        ...

    def delete_state(self, state_name: str) -> bool:
        """Delete a saved state."""
        ...
```

### 7.3 State Metadata Format

```json
{
  "state_name": "optimized_v2",
  "created": "2025-12-11T14:30:00Z",
  "yaml_hash": "sha256:abc123...",
  "context_tokens": 1847,
  "few_shot_count": 3,
  "parent_state": "default",
  "performance": {
    "tool_call_accuracy": 0.95,
    "e2e_accuracy": 0.92,
    "avg_latency_ms": 142,
    "p95_latency_ms": 187,
    "no_tool_precision": 0.88,
    "test_cases_passed": 19,
    "test_cases_total": 20
  },
  "notes": "Optimized by GPT-4 transfer learning, iteration 3"
}
```

---

## 8. Testing and Evaluation Framework

### 8.1 Test Harness

```python
class AgentTestHarness:
    """Programmatic interface for testing agents."""

    def __init__(self, tool_name: str, state_name: str = "default"):
        self.agent = AgentApp(
            tool_name=tool_name,
            state_name=state_name,
            voice_enabled=False,
            headless=True,  # No UI output
        )

    def send_query(self, text: str) -> AgentResponse:
        """Send a query and get structured response."""
        return self.agent.process_query(text)

    def run_test_case(self, test_case: dict) -> TestResult:
        """Run a single test case and check expectations."""
        response = self.send_query(test_case["input"])
        return self._evaluate(response, test_case["expected"])

    def run_benchmark(self, yaml_path: str) -> BenchmarkResult:
        """Run full benchmark from YAML test cases."""
        ...

    def reset_context(self):
        """Reset to initial state (for isolated tests)."""
        self.agent.state_manager.reload_state(self.agent.llm)
```

### 8.2 Evaluation Metrics

| Metric                  | Description                                         | Target  |
| ----------------------- | --------------------------------------------------- | ------- |
| **Tool Call Accuracy**  | % of queries where correct tool was called          | > 95%   |
| **Argument Accuracy**   | % of tool calls with correct arguments              | > 90%   |
| **End-to-End Accuracy** | % of queries with correct final result              | > 90%   |
| **No-Tool Precision**   | % of greetings/chitchat correctly NOT calling tools | > 90%   |
| **Latency P50**         | Median response time                                | < 200ms |
| **Latency P95**         | 95th percentile response time                       | < 500ms |

### 8.3 State Scoring

```python
def score_state(metrics: dict) -> float:
    """Calculate weighted score for state ranking."""
    return (
        0.40 * metrics.get("tool_call_accuracy", 0) +
        0.30 * metrics.get("e2e_accuracy", 0) +
        0.20 * (1 - min(metrics.get("p95_latency_ms", 500) / 500, 1)) +
        0.10 * metrics.get("no_tool_precision", 0)
    )
```

---

## 9. Transfer Learning Pipeline

### 9.1 Workflow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Base YAML   │───▶│ Build State │───▶│ Run Tests   │
│ Config      │    │ (Context)   │    │ (Benchmark) │
└─────────────┘    └─────────────┘    └──────┬──────┘
                                             │
                   ┌─────────────────────────┘
                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Collect     │───▶│ Send to     │───▶│ Get Prompt  │
│ Failures    │    │ Strong LLM  │    │ Suggestions │
└─────────────┘    └─────────────┘    └──────┬──────┘
                                             │
                   ┌─────────────────────────┘
                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Update YAML │───▶│ Rebuild     │───▶│ Re-evaluate │
│ Config      │    │ State       │    │             │
└─────────────┘    └─────────────┘    └──────┬──────┘
                                             │
                        ┌────────────────────┴────────────────────┐
                        ▼                                         ▼
               ┌─────────────┐                           ┌─────────────┐
               │ Improved?   │                           │ Not Better? │
               │ Save State  │                           │ Iterate     │
               └─────────────┘                           └─────────────┘
```

### 9.2 Prompt Optimizer

```python
class PromptOptimizer:
    """Use strong model to optimize prompts for on-device model."""

    def __init__(self, api_endpoint: str, api_key: str = None):
        """Initialize with API (OpenAI, Anthropic, or local)."""
        self.client = self._init_client(api_endpoint, api_key)

    def analyze_failures(self, failures: List[TestFailure]) -> PromptSuggestions:
        """
        Send failure cases to strong model, get improvement suggestions.

        Returns suggestions for:
        - Tool instructions improvements
        - Additional few-shot examples
        - Constraint modifications
        """
        ...

    def optimize_yaml(
        self,
        yaml_path: str,
        suggestions: PromptSuggestions,
        output_path: str = None,
    ) -> str:
        """Apply suggestions to YAML, return new config path."""
        ...
```

### 9.3 Interaction Collector

```python
class InteractionCollector:
    """Capture user interactions for test set creation."""

    def __init__(self, output_dir: str = "optimization/collected/"):
        self.output_dir = Path(output_dir)
        self.session_id = datetime.now().isoformat()

    def record(
        self,
        user_input: str,
        tool_call: dict = None,
        tool_result: dict = None,
        final_response: str = None,
        user_feedback: str = None,  # "correct", "wrong", "partial"
    ):
        """Record an interaction with optional user feedback."""
        ...

    def export_to_test_cases(
        self,
        output_path: str,
        filter_feedback: List[str] = None,
    ) -> int:
        """Export collected interactions as YAML test cases."""
        ...
```

---

## 10. Implementation Phases

### Phase 1: Foundation (Week 1)
- [ ] Create per-tool directory structure
- [ ] Implement YAML config loader
- [ ] Update `tool_discovery.py` to scan `tools/*/`
- [ ] Implement `StateManager` with YAML snapshots
- [ ] Merge agents into unified `agent.py`

### Phase 2: Testing Framework (Week 2)
- [ ] Build `AgentTestHarness`
- [ ] Create benchmark runner
- [ ] Implement evaluation metrics
- [ ] Migrate existing tools to per-tool directories

### Phase 3: Transfer Learning (Week 3)
- [ ] Build `PromptOptimizer` with API integration
- [ ] Implement `InteractionCollector`
- [ ] Create evolution/selection pipeline
- [ ] Document optimization workflow

### Phase 4: Documentation (Week 4)
- [ ] Write `TUTORIAL.md` - user guide
- [ ] Create `VIDEO_SCRIPT.md` - YouTube video outline
- [ ] Update `README.md` and `AGENTS.md`
- [ ] Add inline documentation and examples

---

## 11. Complete File Structure

```
agent_tools_example/
│
├── agent.py                        # Unified agent (text + voice)
├── agent_config.yaml               # Base configuration
├── state_manager.py                # State save/load management
│
├── tools/
│   ├── __init__.py                 # Tool discovery
│   ├── base.py                     # BaseTool ABC
│   │
│   ├── math/
│   │   ├── __init__.py
│   │   ├── tool.py
│   │   ├── config.yaml
│   │   └── contexts/
│   │
│   ├── weather/
│   │   ├── __init__.py
│   │   ├── tool.py
│   │   ├── api.py
│   │   ├── config.yaml
│   │   └── contexts/
│   │
│   ├── rgb_led/
│   │   ├── __init__.py
│   │   ├── tool.py
│   │   ├── hardware.py
│   │   ├── simulator.py
│   │   ├── config.yaml
│   │   └── contexts/
│   │
│   ├── servo/
│   │   ├── __init__.py
│   │   ├── tool.py
│   │   ├── hardware.py
│   │   ├── simulator.py
│   │   ├── config.yaml
│   │   └── contexts/
│   │
│   ├── elevator/
│   │   ├── __init__.py
│   │   ├── tool.py
│   │   ├── interface.py
│   │   ├── config.yaml
│   │   └── contexts/
│   │
│   └── _template/
│       ├── __init__.py
│       ├── tool.py
│       ├── config.yaml
│       └── README.md
│
├── optimization/
│   ├── __init__.py
│   ├── prompt_optimizer.py
│   ├── interaction_collector.py
│   └── collected/
│
├── doc/
│   ├── SPEC.md                     # This document
│   ├── TUTORIAL.md                 # User tutorial
│   └── VIDEO_SCRIPT.md             # YouTube video outline
│
├── README.md
├── AGENTS.md
└── __init__.py
```

---

## Appendix A: Message Format Reference

The Hailo LLM API expects messages in this format (see `message_formatter.py`):

```python
# System message
{"role": "system", "content": [{"type": "text", "text": "..."}]}

# User message
{"role": "user", "content": [{"type": "text", "text": "..."}]}

# Assistant message
{"role": "assistant", "content": [{"type": "text", "text": "..."}]}

# Tool message
{"role": "tool", "content": [{"type": "text", "text": "..."}]}
```

---

## Appendix B: Qwen 2.5 Tool Calling Format

Tool definitions and calls follow Qwen 2.5 Coder format:

```xml
<!-- Tool definitions in system prompt -->
<tools>
[{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]
</tools>

<!-- Tool call from assistant -->
<tool_call>
{"name": "math", "arguments": {"expression": "5 + 3"}}
</tool_call>

<!-- Tool response -->
<tool_response>
{"ok": true, "result": "5 + 3 = 8"}
</tool_response>
```

---

*End of Specification*

