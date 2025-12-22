# Hailo Voice Assistant Interactive Application

An interactive voice-controlled AI assistant using Hailo's Speech-to-Text and Large Language Model for real-time audio processing and conversational AI.

## Prerequisites

### Required: Piper TTS Model Installation

Before running the voice assistant, you must install the Piper TTS voice model.

**For installation instructions, see:**
- [Voice Processing Module Documentation](../gen_ai_utils/voice_processing/README.md)

The application will check for the model on startup and display an error with instructions if it's missing.

The voice assistant uses the shared voice processing module from `hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing`.

## Audio Setup

**High-quality microphone input is crucial for optimal speech recognition performance.**

For audio device configuration and troubleshooting, use the built-in tool:

```bash
hailo-audio-troubleshoot
```

This tool will:
- List all available audio devices
- Test your microphone and speakers
- Help select and save preferred devices
- Provide platform-specific setup instructions (Raspberry Pi, etc.)

**For detailed audio setup instructions, see:** [Voice Processing Module Documentation](../gen_ai_utils/voice_processing/README.md)

**Tips for best results:**
- Use a USB microphone for better quality than built-in mics
- Position microphone 15-30cm from your mouth
- Minimize background noise during recording
- Speak clearly at a normal volume

## Features

- **Real-time speech processing** with Hailo AI acceleration
- **Interactive voice mode** - press Space to start/stop recording
- **Streaming text-to-speech** - responsive audio playback with interruption support
- **Context management** - maintains conversation history with clear option
- **Debug logging** - detailed logging for troubleshooting (use `--debug` or `--log-level DEBUG`)
- **Low-resource mode** - optional TTS disable for reduced system load

## Requirements

- Hailo AI processor and SDK
- Python 3.10+
- sounddevice (for audio I/O)
- NumPy
- Piper TTS (for voice synthesis)
- Hailo Platform libraries

## Files

- `voice_assistant.py` - Main application with terminal-based voice interface

## Usage

1. Run the application:
   ```bash
   python -m hailo_apps.python.gen_ai_apps.voice_assistant.voice_assistant
   ```

2. Optional flags:
   ```bash
   python -m hailo_apps.python.gen_ai_apps.voice_assistant.voice_assistant --debug      # Enable debug logging
   python -m hailo_apps.python.gen_ai_apps.voice_assistant.voice_assistant --no-tts     # Disable text-to-speech
   python -m hailo_apps.python.gen_ai_apps.voice_assistant.voice_assistant --log-level DEBUG  # Set log level
   ```

3. **Interactive controls**:
   - Press `Space` to start/stop recording
   - Press `Q` to quit the application
   - Press `C` to clear conversation context
   - Press `X` to abort generation and speech
   - Speak naturally during recording

## How it works

The application uses a threaded architecture to handle:
- Real-time audio capture and processing via `VoiceInteractionManager`
- Hailo Speech-to-Text transcription
- Large Language Model inference for responses
- Streaming text-to-speech synthesis with interruption support
- Non-blocking user input handling

The voice assistant can engage in natural conversations, answer questions, and provide assistance while maintaining context
