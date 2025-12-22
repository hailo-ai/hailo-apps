# Hailo LLM Tools - Interactive Chat Agent

Interactive CLI chat agent that uses Hailo LLM models with function calling capabilities. The agent automatically discovers tools and allows the LLM to call them during conversations.

## Overview

The tools application provides an interactive CLI chat agent that uses Hailo LLM models with function calling capabilities. The system automatically discovers tools from modules named `tool_*.py` and allows the LLM to call them during conversations.

### Architecture

The system follows a simple tool discovery and execution pattern:

1. **Tool Discovery**: Tools are automatically discovered from modules following the naming pattern `tool_*.py`
2. **Tool Call Flow**: User input → LLM (with tools) → Tool execution (if needed) → Final response
3. **Context Management**: Uses token-based context management (clears at 80% capacity) for optimal performance

## Quick Start

### Basic Usage

```bash
# Text mode (default)
python -m hailo_apps.python.gen_ai_apps.agent_tools_example.agent

# Voice mode
python -m hailo_apps.python.gen_ai_apps.agent_tools_example.agent --voice
```

### With Debug Logging

To enable debug logging, edit `config.py` and set:
```python
DEFAULT_LOG_LEVEL = "DEBUG"
```

## Interactive Commands

Once in the chat, you can use these commands:

| Command    | Description                |
| ---------- | -------------------------- |
| `/exit`    | Exit the chat              |
| `/clear`   | Clear conversation context |
| `/context` | Show context token usage   |

## Available Tools

The system automatically discovers tools from modules named `tool_*.py`. Current tools include:

### Math Tool
Perform basic arithmetic operations: addition, subtraction, multiplication, and division. Multi step calculations are not supported. This is a simple tool for basic calculations.

### Weather Tool
Get current weather and rain forecasts for any location worldwide using the Open-Meteo API (no API key required).

**Features:**
- Current weather conditions
- Precipitation data
- Timezone-aware responses

**Usage:** The tool uses a `future_days` parameter (0=today, 1=tomorrow, etc.) instead of absolute dates, making it easier for the LLM to use.

### RGB LED Tool
Control RGB LED: turn on/off, change color by name, adjust intensity (0-100%).

**Hardware Support:**
- **Real Hardware**: Uses `rpi5-ws2812` library for Raspberry Pi 5 SPI control
- **Simulator**: Flask-based web interface showing LED state in real-time
- **Configuration**: Set `HARDWARE_MODE` in `config.py` to "real" or "simulator"

**Features:**
- Color control by name (e.g., "red", "blue", "green")
- Intensity adjustment (0-100%)
- On/off control

For hardware installation instructions, see [RGB LED Hardware Setup](#rgb-led-hardware-setup).

### Servo Tool
Control servo: move to absolute angle or by relative angle (-90 to 90 degrees).

**Hardware Support:**
- **Real Hardware**: Uses `rpi-hardware-pwm` library for hardware PWM control on Raspberry Pi
- **Simulator**: Flask-based web interface with visual servo arm display
- **Configuration**: Set `HARDWARE_MODE` in `config.py` to "real" or "simulator"

**Features:**
- Absolute positioning (-90° to 90°)
- Relative movement
- Automatic angle clamping

For hardware installation instructions, see [Servo Hardware Setup](#servo-hardware-setup).

### Hardware Configuration

Edit `config.py` to customize hardware settings

**Notes**:
- **SPI**: Uses the MOSI pin (GPIO 10) automatically - no pin configuration needed. The SPI bus and device numbers correspond to `/dev/spidev0.0` by default.
- **Servo**: Default configuration uses logical channel 0, which maps to GPIO 18 (physical pin 12). Ensure the servo is properly powered with an external power supply for reliable operation.

## Example Session

```
Available tools:
  1. math: Perform basic arithmetic operations: addition, subtraction, multiplication, and division.
  2. weather: Get current weather and rain forecasts (supports future days) using the Open-Meteo API.

Select a tool by number (or 'q' to quit): 1

Selected tool: math
Loading model...

Chat started. Type '/exit' to quit. Use '/clear' to reset context. Type '/context' to show stats.
Tool in use: math

You: what is 5 times 3?
Assistant: 15.0

You: calculate 314 divided by 3
Assistant: 104.66666666666667

You: /exit
Bye.
```

## Tips for Using Tools

To encourage the LLM to use a specific tool, it's helpful to mention the tool explicitly in your request. For example:

- ✅ **Better**: "Turn on the lights using the LED tool"
- ✅ **Better**: "Use the LED tool to set the color to red"
- ❌ **Less reliable**: "Turn on the lights" (LLM might not realize it should use the tool)

The LLM is more likely to call a tool when you:
1. Mention the tool name explicitly (e.g., "LED tool", "math tool", "weather tool")
2. Use action words that match the tool's purpose (e.g., "calculate", "get weather", "turn on")
3. Provide clear parameters (e.g., "set LED to red at 50% brightness")

## Creating New Tools

Tools are automatically discovered - just create a new file following the pattern:

### Step 1: Copy Template

```bash
cp tool_TEMPLATE.py tool_mytool.py
```

### Step 2: Implement Tool Interface

Each tool must expose:

1. **`name: str`** - Unique tool identifier
2. **`description: str`** - Clear instructions for the LLM on when/how to use (this is critical!)
3. **`schema: dict`** - JSON schema following OpenAI function calling format
4. **`TOOLS_SCHEMA: list[dict]`** - List containing function definition
5. **`run(input: dict) -> dict`** - Tool execution function

### Tool Description Best Practices

The `description` field is where tool-specific instructions belong. Be explicit and clear:

✅ **Good Example:**
```python
description: str = (
    "CRITICAL: You MUST use this tool for ALL arithmetic operations. "
    "NEVER calculate math directly - ALWAYS call this tool. "
    "The function name is 'math' (use this exact name in tool calls). "
    "Supported operations: add (+), sub (-), mul (*), div (/). "
    "The 'op' parameter specifies which operation: 'add', 'sub', 'mul', or 'div'."
)
```

### Schema Best Practices

- Follow OpenAI function calling format
- **DO NOT use**: `default`, `minimum`, `maximum`, `minItems`, `maxItems`, `additionalProperties`
- Include clear parameter descriptions
- Specify required vs optional parameters using `required` array
- Use appropriate types (`string`, `number`, `array`, `object`)

### Tool Return Format

The `run()` function must return a dictionary with:

```python
{
    "ok": bool,      # Success status
    "result": Any,    # Success result (if ok=True)
    "error": str     # Error message (if ok=False)
}
```

### Step 3: Test

The tool will be automatically discovered when you run the agent. No code changes needed in the agent!

## Troubleshooting & Common Issues

### Quick Troubleshooting Guide

| Symptom                               | Possible Cause                       | Solution                                                           |
| ------------------------------------- | ------------------------------------ | ------------------------------------------------------------------ |
| **Tool not found**                    | File name doesn't start with `tool_` | Rename file to `tool_myname.py`                                    |
| **Model chats but doesn't call tool** | Description too vague                | Add "CRITICAL: You MUST use this tool..." to description           |
| **"Invalid JSON" error**              | Model output malformed               | Check if tool args use single quotes (fixed automatically usually) |
| **Context full warnings**             | Long conversation                    | Use `/clear` command to reset context                              |
| **Hardware tool fails**               | Permissions or wiring                | Check `sudo` permissions and wiring (GPIO 10/18)                   |

### Common Issues

#### 1. Model Doesn't Call Tools
* **Cause**: Tool descriptions may be unclear or too vague.
* **Fix**: Use explicit imperative language in the `description` field.
  * ❌ "This tool calculates numbers."
  * ✅ "CRITICAL: You MUST use this tool for ANY calculation. NEVER calculate mentally."

#### 2. Parsing Errors
* **Cause**: Model outputting invalid JSON (e.g., single quotes, trailing commas).
* **Fix**: The agent has built-in robust parsing, but you can improve reliability by adding format examples to the system prompt or tool description.

#### 3. Tool Execution Fails
* **Cause**: Exceptions in the `run()` function.
* **Fix**: Ensure your `run()` function handles exceptions and returns `{"ok": False, "error": "..."}` instead of crashing.

#### 4. Wrong Function Name
* **Cause**: Model hallucinates a function name.
* **Fix**: Add explicit instruction: "The function name is 'my_tool_name' (use this exact name)."

### Context Management

The agent uses token-based context management (clears at 80% capacity) instead of message counting. This provides:
- Accurate tracking based on actual token usage
- Maximized context utilization
- Automatic adaptation to different model capacities

Use `/context` command to view current token usage.

### Context Caching

The agent automatically caches the LLM context after initializing the system prompt with tool definitions. This significantly reduces startup time when using the same tool again.

**How it works:**
- On first run with a tool, the system prompt (including tool definitions) is initialized and saved to a cache file
- On subsequent runs with the same tool, the cached context is loaded instantly
- Cache files are tool-specific, so each tool has its own cached context

**Cache file location:**
- Cache files are stored in the `agent_tools_example` directory
- File naming format: `context_{tool_name}.cache`
- Example: `context_math.cache` for the math tool

**When context is cached:**
- After first initialization of a tool's system prompt
- Context is loaded on tool selection
- Context is reloaded after using the `/clear` command (if cache exists)

**How to force re-initialization:**
If you modify the system prompt or tool definitions and want to force re-initialization:
1. Delete the corresponding cache file: `rm context_{tool_name}.cache`
2. Restart the agent

The cache will be automatically regenerated on the next run.

## Customizing LLM Behavior

Users can influence the LLM's behavior by modifying two key areas: the general **System Prompt** and the specific **Tool Descriptions**.

### 1. System Prompt

The system prompt provides the LLM with high-level instructions on how to behave, how to format tool calls, and when (or when not) to use tools in general.

-   **Location**: The system prompt is constructed in the `create_system_prompt` function within `system_prompt.py`.

#### Good Practices for Editing the System Prompt:
-   **Keep it General**: The system prompt should contain rules that apply to *all* tools. Avoid mentioning specific tool names or functionalities here.
-   **Focus on Formatting**: Use it to enforce consistent output formats (e.g., XML tags, JSON standards).
-   **Set High-Level Rules**: Define broad rules, such as when *not* to call a tool (e.g., for greetings or small talk).
-   **Use Clear, Imperative Language**: Use strong, direct language like "ALWAYS," "NEVER," and "MUST" to make instructions unambiguous.

### 2. Tool Descriptions

Tool descriptions are the most important factor in determining whether the LLM will use a tool correctly. Each tool's `description` string tells the LLM its purpose, parameters, and when it should be called.

-   **Location**: Each `tool_*.py` file (e.g., `tool_math.py`) has a `description` variable that is passed to the LLM.

#### Good Practices for Editing Tool Descriptions:
-   **Be Explicit and Critical**: Start with "CRITICAL:" to get the model's attention. Use strong language like "You MUST use this tool for..."
-   **Define the "When"**: Clearly state the conditions under which the tool should be used. List keywords, user intents, and example phrases. For instance, the `tool_rgb_led.py` description includes a comprehensive list: "LED, light, lights, turn on, turn off, change color...".
-   **Specify the Function Name**: Explicitly tell the model the function name to use in the tool call, e.g., "The function name is 'math' (use this exact name in tool calls)."
-   **Explain Parameters Clearly**: Describe what each parameter does and provide examples of valid values (e.g., for `color`, list valid color names).
-   **Give Concrete Examples**: Provide a few examples of user requests and the corresponding tool call parameters. The `tool_servo.py` description does this well: `'move servo to 45 degrees' → mode='absolute', angle=45`.

By carefully crafting the system prompt and, more importantly, the individual tool descriptions, you can significantly improve the accuracy and reliability of the agent's tool-calling capabilities.

## Hardware Installation Guides

### RGB LED Hardware Setup

To use the RGB LED tool with real hardware on a Raspberry Pi 5:

1. **Enable SPI** (required): The LED control uses the Serial Peripheral Interface (SPI) port. Enable SPI via:
   ```bash
   sudo raspi-config
   ```
   Navigate to `Interfacing Options` > `SPI` and select `Yes` to enable. Then reboot:
   ```bash
   sudo reboot
   ```

2. **Wiring**: Connect the LED strip's data input (DIN) to the Raspberry Pi's MOSI pin:
   - **GPIO 10** (pin 19 on the header) - This is the SPI MOSI pin
   - Ensure a common ground between the Raspberry Pi and the LED strip
   - Power the LED strip according to its specifications

3. **Install the library**:
   ```bash
   pip install rpi5-ws2812
   ```

**Troubleshooting:**
- **Hardware mode failures**: If `HARDWARE_MODE='real'` is set and hardware initialization fails, the application will exit with an error. Make sure:
  - SPI is enabled via `sudo raspi-config` (Interfacing Options > SPI > Enable) and the system has been rebooted
  - Required libraries are installed (`rpi5-ws2812`)
  - The LED strip data line is connected to GPIO 10 (SPI MOSI pin)
  - SPI device is accessible (check with `ls /dev/spidev*`)
- **SPI not enabled**: If you see initialization errors, verify SPI is enabled:
  ```bash
  ls /dev/spidev*
  ```
  You should see `/dev/spidev0.0` (and possibly `/dev/spidev0.1`). If not, enable SPI via `sudo raspi-config` and reboot.
- **Permission errors**: If you see permission errors accessing SPI, you may need to add your user to the `spi` group:
  ```bash
  sudo usermod -a -G spi $USER
  ```
  Then log out and log back in (or reboot).

### Servo Hardware Setup

To use the servo tool with real hardware on a Raspberry Pi:

1. **Install the library**:
   ```bash
   pip install rpi-hardware-pwm
   ```

2. **Enable Hardware PWM**:
   Edit `/boot/firmware/config.txt`:
   ```bash
   sudo nano /boot/firmware/config.txt
   ```

   **Disable onboard audio** (may be required on some models): On older Raspberry Pi models, the analog audio may conflict with hardware PWM channels. If you encounter issues with PWM initialization, try disabling audio by commenting out this line (if present):
   ```
   # dtparam=audio=on
   ```
   **Note**: On Raspberry Pi 5 and some newer models, PWM and audio can typically coexist, so this step may not be necessary. Try enabling PWM first; only disable audio if you encounter conflicts.

   **Add the PWM overlay**: Add the following line at the **bottom** of the config file:
   ```
   dtoverlay=pwm-2chan
   ```
   This enables hardware PWM channels:
   - **Logical Channel 0** → GPIO 18 (PWM0_CHAN2)
   - **Logical Channel 1** → GPIO 19 (PWM0_CHAN1)

   **Important**: Place the `dtoverlay` line at the bottom of the config file to ensure proper loading.

   Save the file and reboot:
   ```bash
   sudo reboot
   ```

   After rebooting, verify PWM is enabled:
   ```bash
   ls /sys/class/pwm/
   ```
   You should see `pwmchip0` and `pwmchip1` listed. This confirms hardware PWM is enabled.

   **Verify PWM pin configuration**: Test that GPIO 18 is configured for PWM:
   ```bash
   pinctrl get 18
   ```
   Expected successful output (indicating PWM function is active):
   ```
   18: a3 pd | lo // PIN12/GPIO18 = PWM0_CHAN2
   ```
   If the overlay loaded correctly, the output should show a function other than `input` or `output` (like `PWM0_CHAN2` in the example above).

3. **Wiring**: Connect the servo motor to the Raspberry Pi:
   - **Control Signal (Orange/Yellow wire)**:
     - **Logical Channel 0**: Connect to **GPIO 18** (physical pin 12)
     - **Logical Channel 1**: Connect to **GPIO 19** (physical pin 35)
   - **Power (Red wire)**: Connect to **5V** (pin 2 or 4) - **IMPORTANT**: Use external power supply for high current servos
   - **Ground (Brown/Black wire)**: Connect to **GND** (pin 6, 9, 14, 20, 25, 30, 34, or 39)

   **⚠️ Power Warning**: Standard servos can draw significant current (often 1-2A or more). Do NOT power the servo directly from the Raspberry Pi's 5V pin without an external power supply, as this can damage the Pi or cause instability. Use one of these approaches:
   - **Recommended**: Use a separate 5V power supply for the servo, with common ground shared between the Pi and servo power supply
   - **For small servos only**: If using a very small, low-current servo (< 500mA), you may power from Pi's 5V, but monitor for stability issues

4. **Servo Specifications**:
   - Works with standard PWM servos (e.g., SG90, MG90S, etc.)
   - Default angle range: -90° to +90° (configurable in `config.py`)
   - Control signal: 50Hz hardware PWM (standard servo frequency)
   - Uses hardware PWM for precise, jitter-free control

**Troubleshooting:**
- **Hardware mode failures**: If `HARDWARE_MODE='real'` is set and hardware initialization fails, the application will exit with an error. Make sure:
  - Hardware PWM is enabled in `/boot/firmware/config.txt` (see step 1)
  - The system has been rebooted after enabling PWM
  - Required library is installed (`rpi-hardware-pwm`)
  - The servo control signal is connected to the correct GPIO pin:
    - Logical channel 0 → GPIO 18 (physical pin 12)
    - Logical channel 1 → GPIO 19 (physical pin 35)
  - The servo is properly powered (external power supply recommended)
  - Verify PWM configuration with `pinctrl get 18` (should show PWM0_CHAN2)
- **Servo not moving**: Check:
  - Wiring connections (signal, power, ground)
  - Power supply voltage (should be 5V for most servos)
  - Power supply current capacity (servos need adequate current)
  - Verify PWM is enabled: `ls /sys/class/pwm/` should show `pwmchip0` and `pwmchip1`
  - Verify servo works with a simple test script
- **Jittery or unstable movement**: This often indicates:
  - Insufficient power supply current
  - Poor ground connection
  - Electrical noise - try adding a capacitor (100-1000µF) across servo power and ground
  - Servo may be damaged or incompatible
  - Hardware PWM should eliminate jitter - if jitter persists, check power supply and wiring

## References

- **Qwen 2.5 Coder Tool Calling** - Colab notebook with hands-on walkthrough ([link](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_Coder_(1.5B)-Tool_Calling.ipynb))
- **OpenAI Function Calling Guide** - Official reference on defining functions ([link](https://platform.openai.com/docs/guides/function-calling?api-mode=chat#defining-functions))
