# Hailo Simple Whisper Chat Example

A simple example application demonstrating audio transcription using Hailo's Whisper speech-to-text model.

## Features

- Audio file loading and processing
- Speech-to-text transcription
- Segment-based output
- Default audio file: `audio.wav` (in same directory)
- Auto-downloads model on first run

## Prerequisites

- Hailo AI accelerator device (H10 or compatible)
- Python 3.10+
- Hailo Platform SDK

## Installation

Before running this example, ensure GenAI dependencies are installed:

```bash
# From the repository root directory
pip install -e ".[gen-ai]"
```

This will install required packages including:
- NumPy

**For complete installation instructions, see:** [GenAI Applications Installation Guide](../README.md#installation)

## Files

- `simple_whisper_chat.py` - Main example script demonstrating Whisper transcription functionality
- `audio.wav` - Sample audio file for testing

## Usage

Run the example with the default audio file:

```bash
python -m hailo_apps.python.gen_ai_apps.simple_whisper_chat.simple_whisper_chat
```

Or specify a custom audio file:

```bash
python -m hailo_apps.python.gen_ai_apps.simple_whisper_chat.simple_whisper_chat --audio /path/to/your/audio.wav
```

### Command Line Options

- `--hef-path PATH` - Specify a custom path to the HEF model file
- `--list-models` - List available models for this application
- `--audio PATH` - Path to audio file (default: audio.wav in same directory)

### Example Output

The example will:
1. Initialize the Hailo device and load the Whisper model
2. Load the audio file (default: audio.wav in the same directory)
3. Process the audio and generate transcription segments
4. Display the complete transcription
5. Clean up resources

## Model Requirements

The example uses the `WHISPER_MODEL_NAME_H10` model which is automatically downloaded on first run. No manual download required.

**Note:** Models are downloaded automatically when you run the example for the first time.

## Audio File Format

The example supports WAV files with:
- Sample rate: Any (will be processed by the model)
- Channels: Mono or stereo (will be processed)
- Sample width: 16-bit PCM

## Troubleshooting

### Model not found error
- Ensure Hailo models are properly installed
- Check model paths in the resource directory
- The model will be downloaded automatically on first run

### Device initialization failed
- Verify Hailo device is connected and recognized
- Check device permissions

### Audio file not found
- Verify the audio file exists at the specified path
- If using default, ensure `audio.wav` is in the same directory as the script
- Check file permissions

### Error reading WAV file
- Ensure the file is a valid WAV format
- Check that the file is not corrupted
- Verify the file is readable

### Import errors
- Ensure Hailo Platform SDK is properly installed
- Verify Python environment has all required packages (NumPy)

## How it works

The example demonstrates basic Whisper transcription:
1. Creates a VDevice for Hailo hardware access
2. Initializes a Speech2Text instance with the Whisper model
3. Loads the audio file using Python's wave module
4. Converts audio to the format expected by the model (float32, normalized)
5. Generates transcription segments using the Whisper model
6. Combines segments into a complete transcription
7. Cleans up resources properly

This is a simplified example. For more advanced features like real-time audio recording, streaming transcription, and voice interaction, see the full [Voice Assistant](../voice_assistant/) application.

