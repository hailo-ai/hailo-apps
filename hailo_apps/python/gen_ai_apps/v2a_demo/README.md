Hailo Voice-to-Action Demo
==========================

A voice-controlled AI assistant powered by a Hailo-10H AI accelerator. Speak a command, and the system will identify the right tool, extract parameters, execute it, and speak the results back — all using on-device AI inference.

Requirements
------------
- Hailo-10H host device
- Microphone and speaker
- Python 3.10+
- HailoRT Python package (`hailo-platform`) installed per [Hailo documentation](https://hailo.ai/developer-zone/)
- Dependencies from `requirements.txt`

Supported Models
----------------

This example uses the following models:

- Wake Word Detection: OpenWakeWord (ONNX, CPU)
- Stage 1 — STT: Whisper-Base (HEF, Hailo)
- Stage 2 — Tool Selection: all-MiniLM-L6-v2 (HEF, Hailo)
- Stage 3 — LLM: Qwen2.5-Coder-1.5B-Instruct (HEF, Hailo)
- Stage 5 — TTS: Piper TTS (ONNX, CPU)

## Linux Installation

### Existing Shared Installation

If you already installed the repository, activate its environment and install the demo dependencies and required assets:

```bash
source setup_env.sh
cd hailo_apps/python/gen_ai_apps/v2a_demo
python -m pip install -r requirements.txt
bash download_resources.sh
python main.py
```

Run these commands from the repository root. For a separate environment, follow the standalone instructions below.

### Standalone Installation

To avoid compatibility issues, it's recommended to use a clean virtual environment.

0. Install PCIe driver and PyHailoRT
    - Download and install the PCIe driver and PyHailoRT from the Hailo website.
    - To install the PyHailoRT wheel:
    ```shell script
    pip install hailort-X.X.X-cpXX-cpXX-linux_x86_64.whl
    ```

1. Clone the repository:
    ```shell script
    git clone https://github.com/hailo-ai/hailo-apps.git
    cd hailo-apps/hailo_apps/python/gen_ai_apps/v2a_demo
    ```

2. Install dependencies:
    ```shell script
    pip install -r requirements.txt
    ```

3. Download required wake-word, TTS, and tool-selection assets before running:
    ```bash
    bash download_resources.sh
    ```

4. (Optional) Set API key for the weather tool:
    ```shell script
    export OPENWEATHER_API_KEY="your_key_here"
    ```

5. (Optional) Pre-download HEF resources:
    ```shell script
    python -m hailo_apps.installation.download_resources --group v2a_demo
    ```

### Run
After completing installation, run from the application folder:
```shell script
python main.py
```

## Windows Installation

To avoid compatibility issues, it's recommended to use a clean virtual environment.

0. Install HailoRT (MSI) + PyHailoRT
    1. Download and install the **HailoRT Windows MSI** from the Hailo website.
    2. During installation, make sure **PyHailoRT** is selected.
    3. Create and activate a virtual environment:
    ```powershell
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```
    4. Install the PyHailoRT wheel:
    ```powershell
    pip install "C:\Program Files\HailoRT\python\hailort-*.whl"
    ```

1. Clone the repository:
    ```powershell
    git clone https://github.com/hailo-ai/hailo-apps.git
    cd hailo-apps\hailo_apps\python\gen_ai_apps\v2a_demo
    ```

2. Install dependencies:
    ```powershell
    pip install -r requirements.txt
    ```
3. Download required wake-word, TTS, and tool-selection assets before running:
    ```powershell
    .\download_resources.ps1
    ```

4. (Optional) Set API key for the weather tool:
    ```powershell
    $env:OPENWEATHER_API_KEY="your_key_here"
    ```

5. (Optional) Pre-download HEF resources:
    ```powershell
    python -m hailo_apps.installation.download_resources --group v2a_demo
    ```

### Run
```powershell
python .\main.py
```

Available Tools
---------------

| Tool | Description | Example Command |
|------|-------------|-----------------|
| `get_weather` | Weather forecast for a city | "Hey Hailo, what's the weather in London?" |
| `control_led` | Control the board LED (Raspberry Pi only) | "Hey Hailo, blink the LED 5 times" |
| `get_travel_time` | Travel time between locations | "Hey Hailo, how long to drive from London to Manchester?" |
| `data_storage` | Store/retrieve personal info | "Hey Hailo, remember John's phone is 123-456" |
| `system_check` | CPU, RAM, disk, temperature report | "Hey Hailo, run a system check" |
| `explain_tools` | List available capabilities | "Hey Hailo, what can you do?" |

### Travel configuration

Set your addresses before launching the app to use "home" and "work":

```bash
export HOME_ADDRESS="Alexanderplatz, Berlin, Germany"
export WORK_ADDRESS="Berlin Hauptbahnhof, Berlin, Germany"
```

In Windows PowerShell:

```powershell
$env:HOME_ADDRESS = "Alexanderplatz, Berlin, Germany"
$env:WORK_ADDRESS = "Berlin Hauptbahnhof, Berlin, Germany"
```

Replace the examples with your addresses. Variables last for the current shell
session; restart the app after changes. Addresses cannot be saved by voice.

"Here" or an omitted starting place uses approximate IP location, which may be
inaccurate with a VPN. Travel requests require internet access.

Arguments
---------

- `--wake-word-model`: [optional] Path to wake-word model. Default: `resources/hey_hailo.onnx`.
- `--audio-input-path`: [optional] Process a pre-recorded audio file once and exit.
- `--audio-output-path`: [optional] Save generated TTS output to a file.
- `--audio-device`: [optional] Microphone device index.
- `--debug`: [optional] Enable debug logs.

For more information:
```shell script
python main.py -h
```

Example
-------

**Standard mode**
```shell script
python main.py
```

**Select microphone device**
```shell script
python main.py --audio-device 2
```

**Use a different wake-word model**
```shell script
python main.py --wake-word-model resources/hey_hailo.onnx
```

**Process pre-recorded input**
```shell script
python main.py --audio-input-path sample.wav
```

**Save TTS output**
```shell script
python main.py --audio-output-path reply.wav
```

Adding a New Tool
-----------------

Adding a tool requires one new file and one line in the registry.

**Step 1.** Create `tools/your_tool.py`:

```python
"""Description of your tool."""

TOOL_PROMPT = (
    "Extract parameters from the user's request as a JSON object.\n"
    "Parameters:\n"
    '- "param1" (required): Description.\n'
    "\n"
    "Examples:\n"
    '"Do something with X" -> {"param1": "X"}\n'
    "\n"
    "Output ONLY the JSON object, nothing else."
)

# Natural language descriptions of what this tool does.
# The tool selector compares the user's speech against these to decide
# which tool to invoke. Write 7-10 diverse phrasings of the same intent.
TOOL_DESCRIPTIONS = [
    "Do something with a given input",
    "Perform an action on X",
    "Can you do something for me?",
    "I want to do something with X",
    "Handle requests to do something",
    "Process a user's request to do X",
    "Execute an action based on user input",
]

def your_tool(param1: str) -> str:
    """Execute the tool. Parameter names must match the JSON keys in TOOL_PROMPT.
    Returns a string that will be spoken back to the user."""
    return f"Done with {param1}."
```

**Step 2.** Register in `tools/__init__.py` — add the import and one line to `_REGISTRY`:

```python
from tools import ..., your_tool

_REGISTRY = {
    ...
    "your_tool": (your_tool, "your_tool"),
}
```

For tools that take no parameters (like `system_check`), set `TOOL_PROMPT` to output `{}` and add the tool name to `NO_PARAM_TOOLS` in `tools/__init__.py`.

Pipeline Architecture
---------------------

```text
Microphone
    |
    v
[Wake Word Detection]      OpenWakeWord (ONNX, CPU)
    |
    v
[Stage 1 — STT]            Whisper-Base (HEF, Hailo)
    |  text
    v
[Stage 2 — Tool Selection] all-MiniLM-L6-v2 (HEF, Hailo)
    |  tool name
    v
[Stage 3 — LLM]            Qwen2.5-Coder-1.5B-Instruct (HEF, Hailo)
    |  JSON params
    v
[Stage 4 — Tool Execution] Python function call
    |  response text
    v
[Stage 5 — TTS]            Piper TTS (ONNX, CPU)
    |
    v
Speaker
```

Additional Notes
----------------

- This demo targets Hailo-10H systems.
- `OPENWEATHER_API_KEY` is required only for weather requests.
- If `--audio-input-path` is not provided, the app runs in continuous listening mode.

Disclaimer
----------
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.
