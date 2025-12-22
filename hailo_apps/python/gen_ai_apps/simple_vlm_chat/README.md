# Hailo Simple VLM Chat Example

A simple example application demonstrating image analysis and description using Hailo's Vision Language Model (VLM).

## Features

- Image loading and preprocessing
- Visual question answering
- Image description generation
- Uses example image from doc/images/ directory
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
- OpenCV (opencv-python)
- NumPy

**For complete installation instructions, see:** [GenAI Applications Installation Guide](../README.md#installation)

## Files

- `simple_vlm_chat.py` - Main example script demonstrating VLM image analysis functionality

## Usage

Run the example:

```bash
python -m hailo_apps.python.gen_ai_apps.simple_vlm_chat.simple_vlm_chat
```

### Command Line Options

- `--hef-path PATH` - Specify a custom path to the HEF model file
- `--list-models` - List available models for this application

### Example Output

The example will:
1. Initialize the Hailo device and load the VLM model
2. Load an example image from the repository's doc/images/ directory
3. Ask the VLM a question about the image ("How many people in the image?")
4. Display the model's response
5. Clean up resources

## Model Requirements

The example uses the `VLM_MODEL_NAME_H10` model which is automatically downloaded on first run. No manual download required.

**Note:** Models are downloaded automatically when you run the example for the first time.

## Troubleshooting

### Model not found error
- Ensure Hailo models are properly installed
- Check model paths in the resource directory
- The model will be downloaded automatically on first run

### Device initialization failed
- Verify Hailo device is connected and recognized
- Check device permissions

### Image not found error
- Verify the example image exists at `doc/images/barcode-example.png` relative to the repository root
- The script uses a relative path from the repository root

### Import errors
- Ensure Hailo Platform SDK is properly installed
- Verify Python environment has all required packages (OpenCV, NumPy)

## How it works

The example demonstrates basic VLM image analysis:
1. Creates a VDevice for Hailo hardware access
2. Initializes a VLM instance with the model
3. Loads and preprocesses an image (resizes to 336x336, converts to RGB)
4. Constructs a prompt with system message and user question
5. Generates a response using the VLM with the image frame
6. Cleans up resources properly

This is a simplified example. For more advanced features like real-time video processing, interactive Q&A mode, and custom prompts, see the full [VLM Chat](../vlm_chat/) application.

