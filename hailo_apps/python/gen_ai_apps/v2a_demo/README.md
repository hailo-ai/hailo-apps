Hailo Voice-to-Action (V2A) Demo
================================

A voice-controlled AI assistant powered by a Hailo-10H AI accelerator. Say **"Hey Hailo"** and a command, and the system will identify the right tool, extract its parameters, execute it, and speak the result back — all with on-device AI inference, no cloud LLM.

```text
"Hey Hailo, how long does it take to drive from home to the airport?"
  -> get_travel_time(origin="home", destination="the airport", mode="driving")
  -> "Driving from home to the airport takes about 19 minutes, covering 17 kilometers."
```

Requirements
------------
- Hailo-10H host: Raspberry Pi 5 with the Raspberry Pi AI HAT+ 2, or an x86_64 Linux / Windows machine with a Hailo-10H M.2 module
- HailoRT and PyHailoRT. Tested with HailoRT 5.1.1 (the Raspberry Pi `apt` packages) and 5.4.0
- Python 3.10+
- Microphone and speaker (see [Microphone and speaker setup](#microphone-and-speaker-setup))
- Internet connection for the weather and travel tools, and to download models on the first run

Supported Models
----------------

This example uses the following models:

- Wake Word Detection: OpenWakeWord "Hey Hailo" (ONNX, CPU)
- Stage 1 — STT: Whisper-Base (HEF, Hailo)
- Stage 2 — Tool Selection: all-MiniLM-L6-v2 (HEF, Hailo)
- Stage 3 — LLM: Qwen2.5-Coder-1.5B-Instruct (HEF, Hailo)
- Stage 5 — TTS: Piper TTS (ONNX, CPU)

## Raspberry Pi 5 Installation

1. Set up the AI HAT+ 2 and install HailoRT by following the [Raspberry Pi AI HAT+ / HAT+ 2 Guide](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html#ai-hat-plus):
    ```shell script
    sudo apt update
    sudo apt install hailo-h10-all libportaudio2
    sudo reboot
    ```

2. Clone the repository and create a virtual environment that can see the system PyHailoRT package:
    ```shell script
    git clone https://github.com/hailo-ai/hailo-apps.git
    cd hailo-apps/hailo_apps/python/gen_ai_apps/v2a_demo
    python3 -m venv --system-site-packages venv
    source venv/bin/activate
    ```
    If you installed a newer HailoRT from the [Hailo Developer Zone](https://hailo.ai/developer-zone/) instead of the `apt` packages, also install its PyHailoRT wheel into the venv: `pip install hailort-X.X.X-cpXX-cpXX-linux_aarch64.whl`.

3. Install dependencies:
    ```shell script
    pip install -r requirements.txt
    pip install --no-deps "openwakeword==0.6.0"
    ```
    `--no-deps` skips openwakeword's `tflite-runtime` dependency, which has no wheels for Python 3.12+ (the app uses the ONNX models). pip then warns that `tflite-runtime` is not installed; this is expected.

4. Download the models and the wake-word, TTS and tool-selection assets:
    ```shell script
    bash download_resources.sh
    ```

5. Complete the [Raspberry Pi setup](#raspberry-pi-setup) (microphone sample rate, LED access).

6. Run:
    ```shell script
    python main.py
    ```

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

2. Install dependencies (`sudo apt install libportaudio2` first if it is missing):
    ```shell script
    pip install -r requirements.txt
    pip install --no-deps "openwakeword==0.6.0"
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
    pip install --no-deps "openwakeword==0.6.0"
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

Microphone and speaker setup
----------------------------

The app records **16 kHz mono** audio and plays the spoken reply at 22.05 kHz through the default output device. List the audio devices and pick the microphone with `--audio-device`:

```shell script
python -m sounddevice
python main.py --audio-device 2
```

Many USB microphones and speakers support only 44.1/48 kHz. On Linux the app then fails at start-up with `Invalid sample rate [PaErrorCode -9997]`, or when it speaks the reply. Fix it by letting ALSA resample. Find the card names with `arecord -l` and `aplay -l` (the word after `card N:`, e.g. `Mic`), then create `~/.asoundrc`:

```text
pcm.!default {
    type asym
    capture.pcm "plughw:CARD=Mic,DEV=0"       # your microphone
    playback.pcm "plughw:CARD=Speaker,DEV=0"  # your speaker
}
```

Run `python main.py` without `--audio-device` so the app uses this default device. On Raspberry Pi OS `~/.asoundrc` can be removed by the desktop after a reboot; put the same configuration in `/etc/asound.conf` (with `sudo`) to keep it.

For reliable wake-word detection, keep the microphone within about 1 m of the speaker. In loud rooms (e.g. conference halls) use a close-talk or headset microphone.

Raspberry Pi setup
------------------

- **Microphone sample rate**: see [Microphone and speaker setup](#microphone-and-speaker-setup).
- **LED tool**: the board LED (`/sys/class/leds/ACT`) is writable only by root, so `control_led` replies "I couldn't control the LED". Give the `gpio` group access once (persists across reboots; your user must be in `gpio`, as the default user is):
    ```shell script
    echo 'SUBSYSTEM=="leds", ACTION=="add", RUN+="/bin/chgrp -R gpio /sys%p", RUN+="/bin/chmod -R g+w /sys%p"' | sudo tee /etc/udev/rules.d/99-v2a-leds.rules
    sudo udevadm control --reload && sudo udevadm trigger --subsystem-match=leds --action=add
    ```

Configuration
-------------

| Environment variable | Used by | Description |
|---|---|---|
| `OPENWEATHER_API_KEY` | `get_weather` | [OpenWeatherMap](https://openweathermap.org/api) API key. Without it, weather requests reply "Weather service is not configured". |
| `HOME_ADDRESS`, `WORK_ADDRESS`, `CURRENT_ADDRESS` | `get_travel_time` | What "home", "work" and "here" mean. See [Travel configuration](#travel-configuration). |

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

"Here" or an omitted starting place uses approximate IP location, which can be tens of
kilometers off (or wrong with a VPN). Set `CURRENT_ADDRESS` the same way to give the
device location. Places such as "the airport" or "the supermarket" resolve to the closest
match near the starting point. Travel requests require internet access.

Arguments
---------

- `--wake-word-model`: [optional] Path to wake-word model. Default: `resources/hey_hailo_v3.onnx`.
- `--audio-input-path`: [optional] Process a pre-recorded audio file once and exit. The file must contain the wake word followed by the command.
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

Troubleshooting
---------------

| Symptom | Cause and fix |
|---|---|
| `Invalid sample rate [PaErrorCode -9997]` | The microphone or speaker does not support 16 kHz / 22.05 kHz. See [Microphone and speaker setup](#microphone-and-speaker-setup). |
| `HAILO_OUT_OF_PHYSICAL_DEVICES` | Another process is using the Hailo-10H. Stop it (only one app can use the device at a time). |
| "I couldn't control the LED" | Missing permission to write the LED; see [Raspberry Pi setup](#raspberry-pi-setup). |
| The wake word is not detected | Speak closer to the microphone and check the input level with `python -m sounddevice`. The detection threshold is `WAKE_WORD_THRESHOLD` in `listener.py`. |
| `--audio-input-path` logs "No wake word / speech found in the input file" | The recording must start with "Hey Hailo", then the command. |

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

Tool embeddings are cached in `resources/tool_embeddings_cache.npz` and recomputed automatically on the next start when `TOOL_DESCRIPTIONS` change.

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
