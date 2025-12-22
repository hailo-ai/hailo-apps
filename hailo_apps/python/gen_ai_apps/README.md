# Hailo GenAI Applications

This directory contains Generative AI applications and utilities for the Hailo AI platform. These applications leverage Hailo's AI acceleration hardware to provide real-time LLM (Large Language Model), VLM (Vision Language Model), and speech processing capabilities.

## Overview

The GenAI applications package includes:

- **Full Applications**: Fully functioning interactive applications
- **Simple Examples**: Basic demonstration scripts for common GenAI tasks
- **Shared Utilities**: Reusable components for building custom GenAI applications

**Note:** These applications require the Hailo10 device. Due to the size of models and additional requirements, it is not installed by default. See [installation instructions](#installation).

## Applications

### 🤖 [Agent Tools Example](agent_tools_example/README.md)

An interactive CLI chat agent that uses Hailo LLM models with function calling capabilities. The agent automatically discovers tools and allows the LLM to call them during conversations.

**Features:**
- Automatic tool discovery and execution
- Text and voice interaction modes
- Context management with token-based tracking
- Multiple built-in tools (math, weather, RGB LED, servo, elevator)

**Documentation:** See [agent_tools_example/README.md](agent_tools_example/README.md) for detailed usage, tool creation guide, and hardware setup instructions.

**Additional Resources:**
- [AGENTS.md](agent_tools_example/AGENTS.md) - Detailed developer documentation and architecture guide
- [SPEC.md](agent_tools_example/doc/SPEC.md) - Technical specification
- [TESTING.md](agent_tools_example/testing/TESTING.md) - Testing framework documentation

### 👁️ [VLM Chat](vlm_chat/README.md)

An interactive computer vision application using Hailo's Vision Language Model (VLM) for real-time image analysis and question answering.

**Features:**
- Real-time video processing with Hailo AI acceleration
- Interactive Q&A mode for captured frames
- Single window display (continuous video feed that freezes during Q&A mode)
- Custom prompt support

**Documentation:** See [vlm_chat/README.md](vlm_chat/README.md) for usage instructions and configuration options.

### 🎤 [Voice Assistant](voice_assistant/README.md)

An interactive voice-controlled AI assistant using Hailo's Speech-to-Text and Large Language Model for real-time audio processing and conversational AI.

**Features:**
- Real-time speech processing with Hailo AI acceleration
- Interactive voice mode (press Space to start/stop recording)
- Streaming text-to-speech with interruption support
- Context management for conversation history
- Debug logging for troubleshooting

**Documentation:** See [voice_assistant/README.md](voice_assistant/README.md) for usage instructions, microphone setup, and troubleshooting.

### 📚 Simple Examples

Basic example applications demonstrating the use of Hailo's GenAI platform for different AI tasks:

#### 💬 [Simple LLM Chat](simple_llm_chat/README.md)

A simple example demonstrating text-based conversation with Hailo's Large Language Model (LLM).

**Features:**
- Simple text prompt processing
- Configurable temperature and token limits
- System message for assistant behavior definition
- Auto-downloads model on first run

**Documentation:** See [simple_llm_chat/README.md](simple_llm_chat/README.md) for usage instructions.

#### 👁️ [Simple VLM Chat](simple_vlm_chat/README.md)

A simple example demonstrating image analysis and description using Hailo's Vision Language Model (VLM).

**Features:**
- Image loading and preprocessing
- Visual question answering
- Image description generation
- Uses example image from doc/images/ directory
- Auto-downloads model on first run

**Documentation:** See [simple_vlm_chat/README.md](simple_vlm_chat/README.md) for usage instructions.

#### 🎤 [Simple Whisper Chat](simple_whisper_chat/README.md)

A simple example demonstrating audio transcription using Hailo's Whisper speech-to-text model.

**Features:**
- Audio file loading and processing
- Speech-to-text transcription
- Segment-based output
- Default audio file included
- Auto-downloads model on first run

**Documentation:** See [simple_whisper_chat/README.md](simple_whisper_chat/README.md) for usage instructions.

### 🔌 [Hailo Ollama](hailo_ollama/README.md)

Integration guide for using Hailo-Ollama with Open WebUI for an interactive AI chat interface.

**Features:**
- Ollama-compatible REST API
- Integration with Open WebUI
- Model management and deployment
- Web-based chat interface

**Documentation:** See [hailo_ollama/README.md](hailo_ollama/README.md) for installation and setup instructions.

## Shared Utilities

Reusable components for building custom GenAI applications:

### LLM Utilities (`gen_ai_utils/llm_utils/`)

Provides helpers for building LLM agents:
- **Context Management**: Token-based context window tracking and caching
- **Message Formatting**: Helper functions to format messages for the LLM
- **Streaming**: Utilities for streaming LLM responses and filtering special tokens
- **Tool Discovery & Execution**: Automatic tool discovery and execution framework
- **Terminal UI**: Terminal UI helpers for interactive applications

### Voice Processing (`gen_ai_utils/voice_processing/`)

Components for building voice-enabled applications:
- **VoiceInteractionManager**: Manages the interaction loop, recording, and UI
- **AudioRecorder**: Handles microphone recording with auto-detection
- **SpeechToTextProcessor**: Wraps Hailo's Speech2Text API (Whisper)
- **TextToSpeechProcessor**: Handles speech synthesis using Piper TTS
- **AudioPlayer**: Cross-platform audio playback
- **AudioDiagnostics**: Tools for device enumeration and troubleshooting

**Documentation:** See [gen_ai_utils/README.md](gen_ai_utils/README.md) for detailed API documentation and usage examples.

## Installation

### Prerequisites

- **Hardware**: Hailo AI GenAI accelerator device (H10 or compatible)
- **Python**: Python 3.10 or higher
- **Hailo Platform SDK**: Must be installed and configured

### Step 1: Install GenAI Dependencies

The GenAI applications require additional Python packages that are not installed by default. Install them using:

```bash
# From the repository root directory
pip install -e ".[gen-ai]"
```

This will install:
- `piper-tts` - For text-to-speech synthesis
- `sounddevice==0.5.1` - For audio input/output (microphone recording and playback)

If you encounter audio issues, you may need to install system dependencies:
```bash
sudo apt-get install portaudio19-dev
```

### Step 2: Voice support (Optional)

If you plan to use voice features (text-to-speech, speech-to-text), you'll need to install the Piper TTS model.
We also provide tools to configure and validate your audio devices.

See [Voice Processing Module Documentation](gen_ai_utils/voice_processing/README.md) for installation instructions.

### Step 3: Model Download

**Important:** GenAI models are **not downloaded by default** due to their large size. They are downloaded **on-demand** when you run an application for the first time.

#### Automatic Download (On-Demand)

When you run a GenAI application for the first time, the required models will be automatically downloaded if they are not already present. This ensures you only download models you actually use.

#### Manual Download (Optional)

If you prefer to download models in advance, you can use the resource downloader:

```bash
# Download models for a specific GenAI app
hailo-download-resources --group vlm_chat --arch hailo10h
hailo-download-resources --group llm_chat --arch hailo10h
hailo-download-resources --group whisper_chat --arch hailo10h

# Download all GenAI models (requires Hailo-10H hardware)
hailo-download-resources --all --include-gen-ai --arch hailo10h

# List available GenAI models
hailo-download-resources --list-models --arch hailo10h
```

**Note:** GenAI applications are only available on Hailo-10H hardware. Make sure to specify `--arch hailo10h` when downloading models.

## Quick Start

### Running an Application

```bash
# Agent Tools Example (text mode)
python -m hailo_apps.python.gen_ai_apps.agent_tools_example.agent

# Agent Tools Example (voice mode)
python -m hailo_apps.python.gen_ai_apps.agent_tools_example.agent --voice

# VLM Chat
python -m hailo_apps.python.gen_ai_apps.vlm_chat.vlm_chat --input usb

# Voice Assistant
python -m hailo_apps.python.gen_ai_apps.voice_assistant.voice_assistant

# Simple Examples
python -m hailo_apps.python.gen_ai_apps.simple_llm_chat.simple_llm_chat
python -m hailo_apps.python.gen_ai_apps.simple_vlm_chat.simple_vlm_chat
python -m hailo_apps.python.gen_ai_apps.simple_whisper_chat.simple_whisper_chat --audio /path/to/audio.wav
```

### Using Shared Utilities

For detailed usage examples and API documentation, see:
- **[Voice Processing Examples](gen_ai_utils/voice_processing/README.md#usage)** - Complete examples for `VoiceInteractionManager`, audio recording, speech-to-text, and text-to-speech
- **[LLM Utilities Examples](gen_ai_utils/llm_utils/README.md#usage-examples)** - Complete examples for context management, streaming, tool discovery, and agent building

**Quick Start Examples:**

```python
# Voice Processing - See detailed examples in voice_processing/README.md
from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.interaction import VoiceInteractionManager

manager = VoiceInteractionManager(
    title="My Voice App",
    on_audio_ready=lambda audio: print(f"Audio: {len(audio)} samples"),
    on_shutdown=lambda: print("Shutting down")
)
manager.run()
```

```python
# LLM Utilities - See detailed examples in llm_utils/README.md
from hailo_platform import VDevice
from hailo_platform.genai import LLM
from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils import context_manager, streaming, message_formatter

vdevice = VDevice()
llm = LLM(vdevice=vdevice)

messages = [
    message_formatter.messages_system("You are a helpful assistant."),
    message_formatter.messages_user("Tell me a joke.")
]

response = streaming.generate_and_stream_response(llm=llm, prompt=messages, max_tokens=200)
vdevice.release()
```

## Architecture

The GenAI applications follow a modular architecture:

```
gen_ai_apps/
├── agent_tools_example/     # Full agent application with tools
├── vlm_chat/                # Vision Language Model application
├── voice_assistant/         # Voice assistant application
├── simple_llm_chat/         # Simple LLM chat example
├── simple_vlm_chat/         # Simple VLM chat example
├── simple_whisper_chat/     # Simple Whisper chat example
├── hailo_ollama/            # Hailo Ollama integration guide
└── gen_ai_utils/            # Shared utilities
    ├── llm_utils/           # LLM interaction utilities
    └── voice_processing/    # Voice processing components
```

Applications use the shared utilities from `gen_ai_utils/` to avoid code duplication and ensure consistency.

## Troubleshooting

### Missing Dependencies

If you see import errors for `piper-tts` or `sounddevice`:
```bash
pip install -e ".[gen-ai]"
```

This installs all dependencies needed for GenAI voice features.

### Model Not Found

If an application reports a missing model:
1. The model will be downloaded automatically on first run
2. Or download manually: `hailo-download-resources --group <app_name> --arch hailo10h`

### Audio Issues

For microphone or audio playback problems, run: `hailo-audio-troubleshoot`

See [Voice Processing Module Documentation](gen_ai_utils/voice_processing/README.md) for detailed troubleshooting.

### Hardware Compatibility

GenAI applications require **Hailo-10H hardware**. They are not available on Hailo-8 or Hailo-8L devices.

## Additional Resources

- [Agent Tools Example README](agent_tools_example/README.md) - Complete guide for the agent application
- [Voice Processing README](gen_ai_utils/voice_processing/README.md) - Voice processing module documentation
- [GenAI Utils README](gen_ai_utils/README.md) - Shared utilities documentation
- [Simple LLM Chat README](simple_llm_chat/README.md) - Simple LLM chat example guide
- [Simple VLM Chat README](simple_vlm_chat/README.md) - Simple VLM chat example guide
- [Simple Whisper Chat README](simple_whisper_chat/README.md) - Simple Whisper chat example guide
- [Hailo Ollama README](hailo_ollama/README.md) - Hailo Ollama integration guide

## License

These applications are part of the Hailo Apps Infrastructure and follow the project's license terms.

