# Hailo Simple LLM Chat Example

A simple example application demonstrating text-based conversation with Hailo's Large Language Model (LLM).

## Features

- Simple text prompt processing
- Configurable temperature and token limits
- System message for assistant behavior definition
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

**For complete installation instructions, see:** [GenAI Applications Installation Guide](../README.md#installation)

## Files

- `simple_llm_chat.py` - Main example script demonstrating LLM chat functionality

## Usage

Run the example:

```bash
python -m hailo_apps.python.gen_ai_apps.simple_llm_chat.simple_llm_chat
```

### Command Line Options

- `--hef-path PATH` - Specify a custom path to the HEF model file
- `--list-models` - List available models for this application

### Example Output

The example will:
1. Initialize the Hailo device and load the LLM model
2. Send a simple prompt asking for a joke
3. Display the model's response
4. Clean up resources

## Model Requirements

The example uses the `LLM_MODEL_NAME_H10` model which is automatically downloaded on first run. No manual download required.

**Note:** Models are downloaded automatically when you run the example for the first time.

## Troubleshooting

### Model not found error
- Ensure Hailo models are properly installed
- Check model paths in the resource directory
- The model will be downloaded automatically on first run

### Device initialization failed
- Verify Hailo device is connected and recognized
- Check device permissions

### Import errors
- Ensure Hailo Platform SDK is properly installed
- Verify Python environment has all required packages

## How it works

The example demonstrates a basic LLM interaction:
1. Creates a VDevice for Hailo hardware access
2. Initializes an LLM instance with the model
3. Constructs a prompt with system and user messages
4. Generates a response using the LLM
5. Cleans up resources properly

This is a simplified example. For more advanced features like context management, streaming, and interactive chat, see the full [Voice Assistant](../voice_assistant/) application.

