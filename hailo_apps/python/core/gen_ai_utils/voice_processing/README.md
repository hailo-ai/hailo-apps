# Voice Processing Module

This module provides components for building voice-enabled applications using Hailo's AI platform.
It handles audio recording, speech-to-text, text-to-speech, and interaction management with robust cross-platform support.

## Components

- **`VoiceInteractionManager`**: Manages the interaction loop, recording, user input, and callbacks.
- **`AudioRecorder`**: Handles microphone recording using `sounddevice` with auto-detection.
- **`SpeechToTextProcessor`**: Wraps Hailo's Speech2Text API (Whisper model).
- **`TextToSpeechProcessor`**: Handles speech synthesis using Piper TTS and playback via `AudioPlayer`.
- **`AudioPlayer`**: Cross-platform audio playback using `sounddevice`.
- **`AudioDiagnostics`**: Tools for device enumeration, testing, and troubleshooting.

## Installation

### Prerequisites

The module relies on `sounddevice` (PortAudio wrapper) for audio I/O and Piper TTS for speech synthesis.

### Piper TTS Model Installation

The Piper TTS model files must be installed in: `local_resources/piper_models/`

#### Quick Installation

```bash
# 1. Navigate to the repository root
cd /path/to/hailo-apps-infra

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
   Edit `hailo_apps/python/core/common/defines.py`:
   ```python
   TTS_MODEL_NAME = "your-chosen-voice-name"
   ```

### Common Voice Options

| Voice Name          | Language     | Gender | Size   | Quality    |
| ------------------- | ------------ | ------ | ------ | ---------- |
| `en_US-amy-low`     | English (US) | Female | ~65MB  | Low (fast) |
| `en_US-amy-medium`  | English (US) | Female | ~100MB | Medium     |
| `en_GB-alan-low`    | English (UK) | Male   | ~65MB  | Low (fast) |
| `en_GB-alan-medium` | English (UK) | Male   | ~100MB | Medium     |

## Audio Configuration & Troubleshooting

### Audio Troubleshooting Tool

If you experience issues with recording or playback, use the built-in troubleshooting tool:

```bash
hailo-audio-troubleshoot
```

This tool will:
1. List all available audio input/output devices.
2. Auto-detect the best devices for voice interaction.
3. Allow you to interactively test your microphone and speakers.
4. Provide platform-specific troubleshooting tips (Raspberry Pi, etc.).

### Auto-Detection

The module automatically selects the best available input and output devices based on:
- Capability (supports 16kHz sample rate).
- System default status.
- Device name (prefers physical hardware over virtual sinks).

You can override this by passing specific `device_id`s to the components, but auto-detection works best for most users.

## Usage

### Voice Interaction Manager

The `VoiceInteractionManager` is the high-level entry point.

```python
from hailo_apps.python.core.gen_ai_utils.voice_processing.interaction import VoiceInteractionManager

def on_audio_ready(audio_data):
    # Process audio here (e.g., transcribe)
    print("Audio received")

manager = VoiceInteractionManager(
    title="My Voice App",
    on_audio_ready=on_audio_ready
)

# Start the interaction loop
manager.run()
```

### Individual Components

#### Speech-to-Text

```python
from hailo_platform import VDevice
from hailo_apps.python.core.gen_ai_utils.voice_processing.speech_to_text import SpeechToTextProcessor

vdevice = VDevice()
s2t = SpeechToTextProcessor(vdevice)

# Transcribe audio
text = s2t.transcribe(audio_data, language="en")
print(f"Transcribed: {text}")
```

#### Text-to-Speech

```python
from hailo_apps.python.core.gen_ai_utils.voice_processing.text_to_speech import TextToSpeechProcessor

# Initialize (auto-detects output device)
tts = TextToSpeechProcessor()

# Queue text for speech
tts.queue_text("Hello, this is a test.")

# Interrupt ongoing speech
tts.interrupt()

# Cleanup
tts.stop()
```

#### Audio Recording

```python
from hailo_apps.python.core.gen_ai_utils.voice_processing.audio_recorder import AudioRecorder

# Initialize (auto-detects input device)
recorder = AudioRecorder(debug=True)

# Start recording
recorder.start()

# ... user speaks ...

# Stop and get audio
audio_data = recorder.stop()

# Cleanup
recorder.close()
```

## Platform Specifics

### Raspberry Pi

- **USB Audio**: Strongly recommended over built-in audio.
- **PulseAudio/PipeWire**: The module works with both.
- **Pro Audio Profile**: If using a USB headset, ensure the "Pro Audio" or "Duplex" profile is selected in your system's Volume Control settings to allow simultaneous recording and playback.

### x86 / Desktop

- Works out-of-the-box with standard ALSA/PulseAudio setups.
- Ensure your microphone is not muted in the system settings.

## Troubleshooting

### "No audio input device detected"

Run `hailo-audio-troubleshoot` to see if your microphone is recognized. Check connections and permissions.

### "PIPER TTS MODEL NOT FOUND"

Follow the installation instructions to download the Piper voice model.

### Poor Audio Quality

1. Run `hailo-audio-troubleshoot` and test the microphone.
2. Ensure you are using a USB microphone or headset.
3. Check background noise levels.

## API Reference

See the docstrings in each module for detailed API info:
- `audio_diagnostics.py`
- `audio_player.py`
- `audio_recorder.py`
- `interaction.py`
- `speech_to_text.py`
- `text_to_speech.py`
