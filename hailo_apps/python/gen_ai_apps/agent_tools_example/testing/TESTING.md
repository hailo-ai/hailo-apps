# Agent Testing Documentation

This document describes how to test the agent functionality, including debug flag, tool calls, and response handling.

## Test Files

### 1. `test_debug_flag.py`
Tests debug flag functionality:
- Debug mode affects streaming output (shows raw tokens)
- Debug logging is enabled when debug flag is set
- Debug flag propagation through the system

### 2. `test_comprehensive.py`
Comprehensive test suite covering:
- Tool call execution
- Debug flag functionality
- Tool call parsing with various formats
- Error detection and reporting

### 3. `test_agent_fix.py`
Quick validation test for basic agent functionality.

## Running Tests

### Unit Tests (No Hardware Required)

```bash
# Test debug flag functionality
python test_debug_flag.py

# Test tool parsing
python -c "from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils import tool_parsing; \
  result = tool_parsing.parse_function_call('<tool_call>{\"name\":\"math\",\"arguments\":{\"expression\":\"5+3\"}}</tool_call>'); \
  print(result)"

# Test comprehensive suite (requires hardware for full execution)
python test_comprehensive.py
```

### Integration Tests (Requires Hardware)

```bash
# Test with test harness
python -m hailo_apps.python.gen_ai_apps.agent_tools_example.testing.runner math

# Test with debug flag
python -m hailo_apps.python.gen_ai_apps.agent_tools_example.agent --tool math --debug

# Test voice mode with debug
python -m hailo_apps.python.gen_ai_apps.agent_tools_example.agent --tool math --voice --debug
```

## Debug Flag Testing

### What the Debug Flag Does

1. **Streaming Output**: In debug mode, all tokens are shown without filtering (including XML tags)
2. **Logging Level**: Sets logging to DEBUG level
3. **TTS Callback**: Adds debug logging for TTS operations
4. **Tool Execution**: Logs detailed information about tool calls and results

### Testing Debug Flag

```python
from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils.streaming import StreamingTextFilter

# Test without debug
filter_no_debug = StreamingTextFilter(debug_mode=False)
result = filter_no_debug.process_token("<tool_call>test</tool_call>")
# Result: '' (filtered out)

# Test with debug
filter_debug = StreamingTextFilter(debug_mode=True)
result = filter_debug.process_token("<tool_call>test</tool_call>")
# Result: '<tool_call>test</tool_call>' (passed through)
```

## Tool Call Testing

### Expected Behavior

1. **Math Query**: "What is 5 plus 3?"
   - Should call `math` tool
   - Should execute with expression "5 + 3"
   - Should return result containing "8"

2. **Non-Math Query**: "Hello, how are you?"
   - Should NOT call any tool
   - Should return direct text response

### Testing Tool Calls

```python
from hailo_apps.python.gen_ai_apps.agent_tools_example.testing.harness import AgentTestHarness

with AgentTestHarness(tool_name="math", state_name="default", headless=True) as harness:
    # Test math query
    response = harness.send_query("What is 5 plus 3?")
    assert response.tool_called == True
    assert response.tool_name == "math"
    assert response.tool_result.get("ok") == True

    # Test non-math query
    harness.reset_context()
    response = harness.send_query("Hello")
    assert response.tool_called == False
```

## Tool Parsing Testing

The parser handles multiple formats:

1. **Standard XML Format**:
   ```
   <tool_call>
   {"name":"math","arguments":{"expression":"5+3"}}
   </tool_call>
   ```

2. **Compact Format**:
   ```
   <tool_call>{"name":"math","arguments":{"expression":"5+3"}}</tool_call>
   ```

3. **Message Format (Edge Case)**:
   ```
   [{'type': 'text', 'text': '<tool_call>{"name":"math","arguments":{"expression":"2+2"}}</tool_call>'}]
   ```

### Testing Parsing

```python
from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils import tool_parsing

# Standard format
result = tool_parsing.parse_function_call(
    '<tool_call>{"name":"math","arguments":{"expression":"5+3"}}</tool_call>'
)
assert result == {"name": "math", "arguments": {"expression": "5+3"}}

# Message format
result = tool_parsing.parse_function_call(
    "[{'type': 'text', 'text': '<tool_call>{\"name\":\"math\",\"arguments\":{\"expression\":\"2+2\"}}</tool_call>'}]"
)
assert result == {"name": "math", "arguments": {"expression": "2+2"}}
```

## Error Detection

The system detects and logs:

1. **Tool Call XML Found But Not Parsed**: Warning logged with raw response
2. **Tool Execution Failed**: Error logged with failure reason
3. **Empty Response**: Warning logged if LLM generates empty response
4. **Context Issues**: Warnings for context management problems

## Validation Checklist

Before considering code complete, verify:

- [ ] Debug flag properly enables debug mode
- [ ] Debug mode shows raw tokens (no filtering)
- [ ] Tool calls are properly parsed from responses
- [ ] Tool calls are executed correctly
- [ ] Tool results are added to context
- [ ] Non-tool queries return direct responses
- [ ] Error cases are properly detected and logged
- [ ] Test harness uses same code path as main agent
- [ ] All gen_ai_utils functions are used correctly

## Known Issues and Workarounds

1. **Message Format Response**: Sometimes LLM returns stringified Python format. The parser now handles this automatically.

2. **Hardware Required**: Full integration tests require Hailo hardware. Unit tests can run without hardware.

3. **Context State**: Tests should reset context between queries to ensure isolation.

## Continuous Testing

Run these tests regularly:

```bash
# Quick validation (no hardware)
python test_debug_flag.py
python -c "from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils import tool_parsing; \
  print('Tool parsing works:', tool_parsing.parse_function_call('<tool_call>{\"name\":\"test\"}</tool_call>') is not None)"

# Full suite (requires hardware)
python test_comprehensive.py
python test_agent_fix.py
```


