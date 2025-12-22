# Voice Processing Module

This module provides components for building voice-enabled applications using Hailo's AI platform.
It handles audio recording, speech-to-text, text-to-speech, and interaction management with robust cross-platform support.

## Features

- **Cross-platform audio I/O** - Works on Linux (x86 and ARM/Raspberry Pi) with automatic device detection
- **Speech-to-text** - Integration with Hailo's Whisper model for real-time transcription
- **Text-to-speech** - Piper TTS integration with streaming playback and interruption support
- **Device management** - Automatic device selection with preference persistence
- **Audio diagnostics** - Built-in tools for troubleshooting and device testing
- **Interaction management** - High-level API for building voice-enabled applications

## Prerequisites

- **Hardware**: Hailo AI accelerator device (H10 or compatible)
- **Python**: Python 3.10 or higher
- **Hailo Platform SDK**: Must be installed and configured
- **System dependencies**: PortAudio development libraries (for sounddevice)

### Installing GenAI Dependencies

The voice processing module requires additional Python packages. Install them using:

```bash
# From the repository root directory
pip install -e ".[gen-ai]"
```

This will install:
- `sounddevice==0.5.1` - For audio input/output (microphone recording and playback)
- `piper-tts` - For text-to-speech synthesis
- `numpy` - For audio data processing

If you encounter audio issues, you may need to install system dependencies:

```bash
sudo apt-get install portaudio19-dev
```

**For complete installation instructions, see:** [GenAI Applications Installation Guide](../../README.md#installation)

## Components

- **`VoiceInteractionManager`**: Manages the interaction loop, recording, user input, and callbacks. Provides a high-level API for building voice-enabled applications.
- **`AudioRecorder`**: Handles microphone recording using `sounddevice` with automatic device detection and preference support. Supports resampling and debug audio saving.
- **`SpeechToTextProcessor`**: Wraps Hailo's Speech2Text API (Whisper model) for audio transcription with language support.
- **`TextToSpeechProcessor`**: Handles speech synthesis using Piper TTS with streaming playback, interruption support, and text cleaning for optimal quality.
- **`AudioPlayer`**: Cross-platform audio playback using `sounddevice` with persistent streaming and automatic device selection.
- **`AudioDiagnostics`**: Tools for device enumeration, testing, troubleshooting, and preference management. Provides platform-specific guidance.

## Files

- `interaction.py` - VoiceInteractionManager class for managing voice interactions
- `audio_recorder.py` - AudioRecorder class for microphone recording
- `speech_to_text.py` - SpeechToTextProcessor class for audio transcription
- `text_to_speech.py` - TextToSpeechProcessor class for speech synthesis
- `audio_player.py` - AudioPlayer class for audio playback
- `audio_diagnostics.py` - AudioDiagnostics class and troubleshooting utilities
- `audio_troubleshoot.py` - Command-line troubleshooting tool entry point

### Piper TTS Model Installation

The Piper TTS model files must be installed in: `local_resources/piper_models/`

#### Quick Installation

```bash
# 1. Navigate to the repository root
cd /path/to/hailo-apps

# 2. Navigate to the piper_models directory
cd local_resources/piper_models

# 3. Download the default voice model (en_US-amy-low)
python3 -m piper.download_voices en_US-amy-low
```

This will download two files (~65MB total):
- `en_US-amy-low.onnx` - The neural network model
- `en_US-amy-low.onnx.json` - Model configuration

#### Verify Installation

```bash
ls -lh local_resources/piper_models/
```

You should see:
```
en_US-amy-low.onnx
en_US-amy-low.onnx.json
```

### Using Alternative Voice Models

Piper supports many voice models in different languages and styles. To use a different voice:

1. **Browse available voices:**
   - Visit: https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_PYTHON.md
   - Or list voices: `python3 -m piper.download_voices --list`

2. **Download your chosen voice:**
   ```bash
   cd local_resources/piper_models
   python3 -m piper.download_voices <voice-name>
   ```

3. **Update the configuration:**
   Edit `hailo_apps/python/core/common/defines.py` and change the `TTS_MODEL_NAME` constant:
   ```python
   TTS_MODEL_NAME = "your-chosen-voice-name"
   ```

   The model files will be automatically resolved from `local_resources/piper_models/` based on this setting.

### Common Voice Options

| Voice Name          | Language     | Gender | Size   | Quality    |
| ------------------- | ------------ | ------ | ------ | ---------- |
| `en_US-amy-low`     | English (US) | Female | ~65MB  | Low (fast) |
| `en_US-amy-medium`  | English (US) | Female | ~100MB | Medium     |
| `en_GB-alan-low`    | English (UK) | Male   | ~65MB  | Low (fast) |
| `en_GB-alan-medium` | English (UK) | Male   | ~100MB | Medium     |

## Audio Configuration

### Device Selection

The module automatically selects the best available input and output devices. On systems with multiple audio devices (especially Raspberry Pi), you can manually select and save your preferred devices:

```bash
# Interactive device selection
hailo-audio-troubleshoot --select-devices

# Command-line selection
hailo-audio-troubleshoot --select-devices --input-device 2 --output-device 3
```

**How it works:**
- Preferences are saved to `local_resources/audio_device_preferences.json`
- `AudioRecorder`, `AudioPlayer`, and `TextToSpeechProcessor` automatically use saved preferences when `device_id=None`
- Falls back to auto-detection if saved devices are unavailable
- Preferences persist across reboots and sessions

### Troubleshooting Tool

For audio issues, use the built-in troubleshooting tool:

```bash
hailo-audio-troubleshoot
```

This tool lists devices, tests hardware, and provides platform-specific troubleshooting tips.

## Usage

### Voice Interaction Manager

The `VoiceInteractionManager` is the high-level entry point that handles the complete interaction loop.

```python
from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.interaction import VoiceInteractionManager
import numpy as np

def on_audio_ready(audio_data: np.ndarray):
    # Process audio here (e.g., transcribe, send to LLM)
    print(f"Audio received: {len(audio_data)} samples")

def on_processing_start():
    # Called when recording starts (e.g., stop TTS)
    print("Recording started")

def on_clear_context():
    # Called when user presses 'C' to clear context
    print("Context cleared")

def on_shutdown():
    # Called when user quits (cleanup)
    print("Shutting down")

def on_abort():
    # Called when user presses 'X' to abort generation
    print("Aborting")

manager = VoiceInteractionManager(
    title="My Voice App",
    on_audio_ready=on_audio_ready,
    on_processing_start=on_processing_start,
    on_clear_context=on_clear_context,
    on_shutdown=on_shutdown,
    on_abort=on_abort,
    debug=False
)

# Start the interaction loop (blocks until user quits)
manager.run()
```

**Interactive Controls:**
- `SPACE` - Start/stop recording
- `Q` - Quit the application
- `C` - Clear conversation context
- `X` - Abort current generation/speech

### Individual Components

#### Speech-to-Text

```python
from hailo_platform import VDevice
from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.speech_to_text import SpeechToTextProcessor
import numpy as np

vdevice = VDevice()
s2t = SpeechToTextProcessor(vdevice)  # Uses default Whisper model

# Or specify a custom HEF path
# s2t = SpeechToTextProcessor(vdevice, hef_path="/path/to/model.hef")

# Transcribe audio (numpy array: float32, mono, 16kHz)
text = s2t.transcribe(audio_data, language="en", timeout_ms=15000)
print(f"Transcribed: {text}")
```

The `SpeechToTextProcessor`:
- Automatically resolves the default Whisper model from the whisper_chat app configuration
- Returns empty string if no speech is detected
- Supports language specification (defaults to "en")
- Includes timeout protection (default 15 seconds)

#### Text-to-Speech

```python
from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.text_to_speech import TextToSpeechProcessor

tts = TextToSpeechProcessor(device_id=None)  # Uses saved preferences or auto-detects

# Queue text for synthesis and playback
tts.queue_text("Hello, this is a test.")

# For streaming responses, use chunk_and_queue with generation IDs
gen_id = tts.get_current_gen_id()
remaining = tts.chunk_and_queue("First sentence. Second sentence.", gen_id, is_first_chunk=True)

# Interrupt current speech
tts.interrupt()

# Cleanup
tts.stop()
```

The `TextToSpeechProcessor`:
- Automatically cleans text (removes markdown, special characters) for better synthesis quality
- Supports streaming with chunk-based queuing for low-latency responses
- Handles interruption gracefully with generation ID tracking
- Uses background threads for non-blocking synthesis and playback

#### Audio Recording

```python
from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.audio_recorder import AudioRecorder

recorder = AudioRecorder(device_id=None, debug=True)  # Uses saved preferences or auto-detects
recorder.start()
# ... user speaks ...
audio_data = recorder.stop()  # Returns numpy array (float32, mono, 16kHz)
recorder.close()
```

The `AudioRecorder` automatically:
- Selects the best available input device (or uses saved preferences)
- Handles device sample rate differences with automatic resampling
- Converts audio to the format expected by speech-to-text models (float32, mono, 16kHz)
- Optionally saves debug WAV files when `debug=True`

#### Audio Playback

```python
from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.audio_player import AudioPlayer

player = AudioPlayer(device_id=None)  # Uses saved preferences or auto-detects
player.play(audio_data)  # Can accept numpy array or WAV file path
player.stop()  # Stop current playback
player.close()  # Cleanup
```

#### Audio Diagnostics

```python
from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.audio_diagnostics import AudioDiagnostics

# List all audio devices
devices = AudioDiagnostics.list_devices()

# Get preferred devices (from saved preferences or auto-detected)
input_id, output_id = AudioDiagnostics.get_preferred_devices()

# Test a device
result = AudioDiagnostics.test_device(device_id=0, device_type='input')
```

## Troubleshooting

### Audio Device Issues

If you encounter audio device problems:

1. **Run the troubleshooting tool:**
   ```bash
   hailo-audio-troubleshoot
   ```

2. **Select and save preferred devices:**
   ```bash
   hailo-audio-troubleshoot --select-devices
   ```

3. **Test your devices:**
   The troubleshooting tool includes device testing capabilities to verify microphone and speaker functionality.

### Missing Dependencies

If you see import errors for `sounddevice` or `piper-tts`:
```bash
pip install -e ".[gen-ai]"
```

### Piper TTS Model Not Found

If you see an error about missing Piper TTS model files:
1. Follow the [Piper TTS Model Installation](#piper-tts-model-installation) instructions above
2. Ensure model files are in `local_resources/piper_models/`
3. Verify the model name matches `TTS_MODEL_NAME` in `hailo_apps/python/core/common/defines.py`

### No Audio Input/Output Detected

- Run `hailo-audio-troubleshoot` to list available devices
- Check system audio settings (ensure microphone/speakers are not muted)
- On Linux, verify ALSA/PulseAudio is working: `aplay -l` and `arecord -l`

## Platform Specifics

### Raspberry Pi

- **USB Audio**: Strongly recommended over built-in audio for better quality and reliability
- **Device Profiles**: Use `hailo-audio-troubleshoot --configure` to set USB device profiles (Pro Audio or Duplex)
- **Permissions**: Ensure your user is in the `audio` group: `sudo usermod -a -G audio $USER`
- See troubleshooting tool for Raspberry Pi-specific setup instructions

### x86 / Desktop

- Works out-of-the-box with standard ALSA/PulseAudio setups
- Ensure microphone is not muted in system settings
- USB audio devices work seamlessly with automatic detection

## API Reference

For detailed API documentation, see the docstrings in each module:
- `interaction.py` - VoiceInteractionManager class
- `audio_recorder.py` - AudioRecorder class
- `speech_to_text.py` - SpeechToTextProcessor class
- `text_to_speech.py` - TextToSpeechProcessor class
- `audio_player.py` - AudioPlayer class
- `audio_diagnostics.py` - AudioDiagnostics class and utility functions

## Additional Resources

- [GenAI Applications README](../../README.md) - Overview of GenAI applications and installation
- [Voice Assistant README](../../voice_assistant/README.md) - Example application using this module
- [Agent Tools Example README](../../agent_tools_example/README.md) - Another example using voice processing
