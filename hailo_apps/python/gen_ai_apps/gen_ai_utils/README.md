# Hailo GenAI Utilities

This package provides shared utilities for Generative AI applications on Hailo platforms. It encapsulates common functionality for voice processing, LLM interaction, and context management.

## Architecture

The package is organized into the following modules:

- **`voice_processing`**: Handles audio input/output and speech processing.
- **`llm_utils`**: Provides utilities for managing Large Language Model interactions and terminal UI.

## Modules

### Voice Processing (`voice_processing`)

Provides components for building voice-enabled applications.

- **`VoiceInteractionManager`**: Manages the interaction loop, recording, and UI.
- **`AudioRecorder`**: Handles microphone recording using sounddevice.
- **`SpeechToTextProcessor`**: Wraps Hailo's Speech2Text API (Whisper).
- **`TextToSpeechProcessor`**: Handles speech synthesis using Piper TTS.

#### Usage Example

For complete usage examples including all callbacks, audio recording, speech-to-text, and text-to-speech, see:
**[Voice Processing Module Documentation](voice_processing/README.md#usage)**

**Quick Example:**
```python
from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.interaction import VoiceInteractionManager

manager = VoiceInteractionManager(
    title="My App",
    on_audio_ready=lambda audio: print(f"Audio: {len(audio)} samples"),
    on_shutdown=lambda: print("Shutting down")
)
manager.run()
```

### LLM Utilities (`llm_utils`)

Provides helpers for building LLM agents.

- **`context_manager`**: Manages LLM context window, including tracking usage and caching.
- **`message_formatter`**: Helper functions to format messages for the LLM.
- **`streaming`**: Utilities for streaming LLM responses and filtering special tokens.
- **`terminal_ui`**: Terminal UI helpers for interactive applications.

#### Usage Example

For complete usage examples including chat loops, tool-based agents, streaming with TTS, and context caching, see:
**[LLM Utilities Module Documentation](llm_utils/README.md#usage-examples)**

**Quick Example:**
```python
from hailo_platform import VDevice
from hailo_platform.genai import LLM
from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils import context_manager, streaming, message_formatter

vdevice = VDevice()
llm = LLM(vdevice=vdevice)

messages = [
    message_formatter.messages_system("You are a helpful assistant."),
    message_formatter.messages_user("Tell me a joke.")
]

if context_manager.is_context_full(llm, context_threshold=0.95):
    llm.clear_context()

response = streaming.generate_and_stream_response(llm=llm, prompt=messages, max_tokens=200)
vdevice.release()
```

## Installation Requirements

### Voice Processing Module

If you plan to use voice processing features (TTS), you must install the Piper TTS model.

**See [Voice Processing Module Documentation](voice_processing/README.md) for installation instructions.**

## Integration Guide

To use these utilities in your standalone application:

1. Import the desired modules from `hailo_apps.python.gen_ai_apps.gen_ai_utils`.
2. Ensure `hailo_apps` is in your Python path.
3. If using voice processing, install the Piper TTS model (see above).
4. Follow the usage examples above.
