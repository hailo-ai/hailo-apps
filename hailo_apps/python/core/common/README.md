# Core Common Utilities

This directory contains shared utilities and infrastructure components used across all Hailo applications. These modules provide core functionality for logging, configuration, hardware detection, data processing, and more.

## Table of Contents

- [Core Infrastructure](#core-infrastructure)
- [Hardware & Device Utilities](#hardware--device-utilities)
- [Data Processing](#data-processing)
- [AI/ML Utilities](#aiml-utilities)
- [Database & Storage](#database--storage)
- [Communication & Integration](#communication--integration)
- [Configuration & Constants](#configuration--constants)

---

## Core Infrastructure

### `hailo_logger.py`
**Purpose**: Centralized logging infrastructure for all Hailo applications.

**Key Features**:
- Unified logging configuration with environment variable support (`HAILO_LOG_LEVEL`)
- Run ID tracking for log correlation across processes
- Smart formatter that adapts output format based on log level (verbose for DEBUG, concise for others)
- CLI argument integration (`--log-level`, `--debug`, `--log-file`)
- Automatic suppression of noisy internal loggers (installation modules)
- Support for both console and file logging

**Main Functions**:
- `init_logging()`: Configure root logger (called once at application startup)
- `get_logger(name)`: Get a named logger instance
- `add_logging_cli_args(parser)`: Add logging flags to argparse parser
- `level_from_args(args)`: Extract log level from parsed CLI arguments
- `get_run_id()`: Get stable run identifier for this process

**Usage Example**:
```python
from hailo_apps.python.core.common import init_logging, get_logger, add_logging_cli_args, level_from_args

parser = argparse.ArgumentParser()
add_logging_cli_args(parser)
args = parser.parse_args()
init_logging(level=level_from_args(args))

logger = get_logger(__name__)
logger.info("Application started")
```

---

### `core.py`
**Purpose**: Core application infrastructure including argument parsers, environment loading, and resource path resolution.

**Key Features**:
- Standardized CLI argument parsers for different application types:
  - `get_base_parser()`: Base parser with common flags
  - `get_pipeline_parser()`: Parser for GStreamer pipeline applications
  - `get_standalone_parser()`: Parser for standalone applications
  - `get_default_parser()`: **DEPRECATED** - Use `get_pipeline_parser()` or `get_standalone_parser()` instead
- Environment variable loading from `.env` files
- Resource path resolution (models, JSON configs, videos, photos)
- Model name resolution based on Hailo architecture
- FIFO drop queue for efficient data buffering

**Main Functions**:
- `load_environment(env_file, required_vars)`: Load and validate environment variables
- `resolve_hef_path(hef_path, app_name, arch)`: **Main method for resolving HEF model file paths** (see detailed docs below)
- `get_resource_path(pipeline_name, resource_type, model, ...)`: Resolve paths to non-model resources (JSON, videos, photos, SO files). Use `resolve_hef_path()` for HEF models.
- `get_model_name(pipeline_name, arch)`: Get model name for given architecture
- `get_base_parser()`: Create base argument parser
- `get_pipeline_parser()`: Create parser for pipeline apps
- `get_standalone_parser()`: Create parser for standalone apps
- `get_default_parser()`: **DEPRECATED** - Legacy function for backward compatibility. Use `get_pipeline_parser()` or `get_standalone_parser()` instead.
- `list_models_for_app(app_name, arch)`: List available models for an application
- `handle_list_models_flag(args, app_name)`: Handle --list-models CLI flag
- `FIFODropQueue`: Thread-safe queue with automatic overflow handling

**Usage Example**:
```python
from hailo_apps.python.core.common import (
    get_pipeline_parser,
    load_environment,
    get_resource_path,
    resolve_hef_path
)
from hailo_apps.python.core.common.defines import DETECTION_PIPELINE, RESOURCES_JSON_DIR_NAME

parser = get_pipeline_parser()  # Use get_pipeline_parser() or get_standalone_parser()
args = parser.parse_args()
load_environment()

# For HEF models, use resolve_hef_path()
model_path = resolve_hef_path(None, DETECTION_PIPELINE, "hailo8")

# For non-model resources (JSON, videos, photos, SO files), use get_resource_path()
json_path = get_resource_path(
    pipeline_name="detection",
    resource_type=RESOURCES_JSON_DIR_NAME,
    model="config.json"
)
```

**Note**: `get_default_parser()` is deprecated and will emit a `DeprecationWarning`. Use `get_pipeline_parser()` for GStreamer pipeline applications or `get_standalone_parser()` for standalone applications instead.

#### `resolve_hef_path()` - Main HEF Resolution Method

**Purpose**: This is the primary function to use when you need to locate a HEF model file. It provides intelligent path resolution with automatic model downloading capability.

**Resolution Order** (checked in sequence):
1. If `hef_path` is `None`, use the default model for the app/architecture
2. If `hef_path` is an absolute path that exists, use it directly
3. If `hef_path` exists in the resources folder, use it
4. If `hef_path` is a known model name (from config), automatically download it
5. If model is not found and not in available models list, return `None`

**Key Features**:
- Automatic model downloading for known models
- Integration with `config_manager` for default model detection
- Support for multiple input formats (absolute paths, relative paths, model names)
- Fallback to legacy `get_resource_path()` if config_manager unavailable
- Architecture-aware model selection

**Parameters**:
- `hef_path` (str | None): User-provided path or model name. Can be:
  - `None`: Uses default model for the app
  - Absolute path: `"/path/to/model.hef"` or `"/path/to/model"`
  - Model name: `"yolov8s"` (will be auto-downloaded if not found)
- `app_name` (str): Application name from resources config. Use pipeline constants from `defines.py` (e.g., `DETECTION_PIPELINE`, `FACE_RECOGNITION_PIPELINE`) or app names like `'detection'`, `'vlm_chat'`, etc.
- `arch` (str): Hailo architecture (`'hailo8'`, `'hailo8l'`, or `'hailo10h'`)

**Returns**: `Path` object pointing to the HEF file, or `None` if:
- Model not found and not in available models list
- Download failed
- Invalid architecture for the app

**Usage Examples**:
```python
from hailo_apps.python.core.common import resolve_hef_path, list_models_for_app
from hailo_apps.python.core.common.defines import DETECTION_PIPELINE

# List available models for an app
list_models_for_app("detection", "hailo8")  # Prints available models and exits

# Use default model (auto-detects from config)
hef = resolve_hef_path(None, DETECTION_PIPELINE, "hailo8")

# Use specific model name (auto-downloads if needed)
hef = resolve_hef_path("yolov8s", DETECTION_PIPELINE, "hailo8")

# Use absolute path
hef = resolve_hef_path("/custom/path/model.hef", DETECTION_PIPELINE, "hailo8")

# Check if model was found
if hef:
    print(f"Using model: {hef}")
else:
    print("Model not found")
```

**Integration**: This function integrates with the `config_manager` to:
- Determine default models for each application
- List available models for auto-download
- Validate model names against the resources configuration

**Note**: For non-model resources (JSON configs, videos, photos, SO files), continue using `get_resource_path()`. Only use `resolve_hef_path()` for HEF model files.

---

### `defines.py`
**Purpose**: Centralized constants and configuration definitions used throughout the codebase.

**Key Features**:
- Architecture constants (HAILO8, HAILO8L, HAILO10H)
- Default paths and directory names
- Environment variable key names
- Model and pipeline name constants
- Configuration validation constants
- Package name constants

**Main Constants**:
- Architecture: `HAILO8_ARCH`, `HAILO8L_ARCH`, `HAILO10H_ARCH`
- Paths: `RESOURCES_ROOT_PATH_DEFAULT`, `DEFAULT_DOTENV_PATH`, `DEFAULT_LOCAL_RESOURCES_PATH`
- Environment keys: `HAILO_ARCH_KEY`, `HAILO_LOG_LEVEL_KEY`, `TAPPAS_POSTPROC_PATH_KEY`
- Model names: `DETECTION_MODEL_NAME_H8`, `FACE_RECOGNITION_MODEL_NAME_H8L`, etc.
- Pipeline names: `DETECTION_PIPELINE`, `FACE_RECOGNITION_PIPELINE`, etc.

**Usage**: Import constants directly for use across modules:
```python
from hailo_apps.python.core.common.defines import HAILO8_ARCH, RESOURCES_ROOT_PATH_DEFAULT
```

---

## Hardware & Device Utilities

### `camera_utils.py`
**Purpose**: Camera device detection and enumeration utilities.

**Key Features**:
- USB camera detection using `udevadm`
- Raspberry Pi camera availability checking
- Video device enumeration and validation

**Main Functions**:
- `get_usb_video_devices()`: Scan `/dev` for USB-connected video devices with capture capability
- `is_rpi_camera_available()`: Check if Raspberry Pi camera is connected and responsive

**Usage Example**:
```python
from hailo_apps.python.core.common import get_usb_video_devices, is_rpi_camera_available

usb_cameras = get_usb_video_devices()  # Returns list like ['/dev/video0', '/dev/video2']
rpi_available = is_rpi_camera_available()  # Returns True/False
```

---

### `installation_utils.py`
**Purpose**: System architecture detection and installation-related utilities.

**Key Features**:
- Hailo architecture detection (H8, H8L, H10H)
- Host architecture detection (x86, ARM, Raspberry Pi)
- Package version detection (HailoRT, Tappas)
- Installation validation utilities

**Main Functions**:
- `detect_hailo_arch()`: Auto-detect connected Hailo device architecture
- `detect_host_arch()`: Detect host system architecture
- `detect_hailort_version()`: Get installed HailoRT version
- `detect_tappas_version()`: Get installed Tappas version
- `is_package_installed(pkg_name)`: Check if a package is installed

**Usage Example**:
```python
from hailo_apps.python.core.common import detect_hailo_arch, detect_host_arch

hailo_arch = detect_hailo_arch()  # Returns 'hailo8', 'hailo8l', 'hailo10h', or None
host_arch = detect_host_arch()    # Returns 'x86', 'rpi', 'arm', or 'unknown'
```

---

## Data Processing

### `buffer_utils.py`
**Purpose**: GStreamer buffer conversion utilities for extracting numpy arrays from video frames.

**Key Features**:
- Support for multiple video formats (RGB, NV12, YUYV)
- Efficient buffer mapping and unmapping
- Format-specific handlers for different pixel layouts

**Main Functions**:
- `get_numpy_from_buffer(buffer, format, width, height)`: Convert GStreamer buffer to numpy array
- `get_numpy_from_buffer_efficient(buffer, format, width, height)`: Optimized version with reduced overhead
- `get_caps_from_pad(pad)`: Extract format, width, and height from GStreamer pad capabilities

**Supported Formats**:
- `RGB`: Standard RGB format (height, width, 3)
- `NV12`: YUV 4:2:0 format (Y plane + interleaved UV plane)
- `YUYV`: YUV 4:2:2 format (height, width, 2)

**Usage Example**:
```python
from hailo_apps.python.core.common import get_numpy_from_buffer, get_caps_from_pad

format, width, height = get_caps_from_pad(pad)
frame = get_numpy_from_buffer(buffer, format, width, height)
# frame is now a numpy array ready for processing
```

---

### `toolbox.py`
**Purpose**: General-purpose utility functions for image processing, file I/O, and data manipulation.

**Key Features**:
- Image loading and validation
- Camera index validation
- JSON file loading with error handling
- Image extension checking
- Camera resolution mapping
- Batch processing utilities
- Preprocessing functions
- Visualization helpers

**Main Functions**:
- `load_json_file(path)`: Load and parse JSON files with comprehensive error handling
- `is_valid_camera_index(index)`: Check if a camera index can be opened
- `list_available_cameras(max_index)`: Enumerate all available camera indices
- `load_images_opencv(images_path)`: Load images from a directory using OpenCV
- `load_input_images(images_path)`: Load and validate input images
- `validate_images(images, batch_size)`: Validate image list for batch processing
- `preprocess_images(images, batch_size, ...)`: Preprocess images for model input
- `default_preprocess(image, model_w, model_h)`: Default image preprocessing function
- `get_labels(labels_path)`: Load class labels from a text file
- `FrameRateTracker`: Class for tracking and displaying FPS

**Note**: Many functions in this module are internal utilities used by pipeline applications. The functions listed above are the main public utilities.

**Usage Example**:
```python
from hailo_apps.python.core.common.toolbox import (
    load_json_file,
    is_valid_camera_index,
    load_images_opencv,
    get_labels
)

config = load_json_file("/path/to/config.json")
if is_valid_camera_index(0):
    # Use camera 0
    pass

images = load_images_opencv("/path/to/images")
labels = get_labels("/path/to/labels.txt")
```

---

## AI/ML Utilities

### `hailo_inference.py`
**Purpose**: High-level inference wrapper for Hailo HEF models with async support.

**Key Features**:
- Simplified interface for Hailo model inference
- Batch processing support
- Input/output format configuration
- Scheduler priority management
- Support for shared VDevice contexts

**Main Class**:
- `HailoInfer`: Wrapper around Hailo platform inference API

**Usage Example**:
```python
from hailo_apps.python.core.common.hailo_inference import HailoInfer

infer = HailoInfer(
    hef_path="/path/to/model.hef",
    batch_size=1,
    input_type="UINT8",
    output_type="FLOAT32",
    priority=0
)

# Perform inference (async with callback)
def inference_callback(bindings_list, **kwargs):
    # Process results from bindings_list
    for binding in bindings_list:
        outputs = binding.output()
        # Process outputs here

infer.run(input_batch=[preprocessed_frame], inference_callback_fn=inference_callback)
```

---

### `hef_utils.py`
**Purpose**: Utilities for extracting information from HEF (Hailo Executable Format) files.

**Key Features**:
- Input resolution extraction from HEF files
- Model metadata parsing
- Support for various tensor layouts (NHWC, NCHW)

**Main Functions**:
- `get_hef_input_size(hef_path)`: Extract input width and height from HEF file (returns tuple of int, int)
- `get_hef_input_shape(hef_path)`: Extract full input shape from HEF file (returns tuple, e.g., (1, 480, 640, 3))
- `get_hef_labels_json(hef_path)`: Extract labels JSON string from HEF file metadata
- Handles multiple tensor format layouts automatically (NHWC, NCHW)

**Usage Example**:
```python
from hailo_apps.python.core.common import (
    get_hef_input_size,
    get_hef_input_shape,
    get_hef_labels_json
)

# Get input dimensions
width, height = get_hef_input_size("/path/to/model.hef")
print(f"Model expects input size: {width}x{height}")

# Get full input shape
shape = get_hef_input_shape("/path/to/model.hef")
print(f"Model input shape: {shape}")  # e.g., (1, 480, 640, 3)

# Get labels JSON
labels_json = get_hef_labels_json("/path/to/model.hef")
```

---

## Database & Storage

### `db_handler.py`
**Purpose**: Database handler for face recognition and embedding storage using LanceDB.

**Key Features**:
- Vector database operations for embeddings
- Face recognition record management
- Sample storage and retrieval
- Similarity search using cosine distance
- Automatic threshold-based classification

**Main Class**:
- `DatabaseHandler`: Manages LanceDB operations for face recognition
- `Record`: Pydantic model for database records with embeddings

**Key Methods**:
- `create_record()`: Create a new face record with embeddings
- `insert_new_sample()`: Add a new sample to an existing record
- `search_record()`: Find similar faces using vector similarity
- `update_record_label()`: Update the label of a record
- `delete_record()`: Remove records from database
- `get_all_records()`: Retrieve all records
- `get_record_by_id()`: Get a record by global_id
- `get_record_by_label()`: Get records by label
- `clear_table()`: Clear all records from the table

**Usage Example**:
```python
from hailo_apps.python.core.common.db_handler import DatabaseHandler, Record

db = DatabaseHandler(
    db_name="faces",
    table_name="people",
    schema=Record,
    threshold=0.7,
    database_dir="/path/to/db",
    samples_dir="/path/to/samples"
)

# Create a face record
record = db.create_record(
    embedding=face_embedding_vector,
    sample="/path/to/image.jpg",
    timestamp=int(time.time()),
    label="Alice"
)

# Search for similar faces
results = db.search_record(query_embedding, top_k=5)
```

---

### `embedding_visualizer.py`
**Purpose**: Visualization utilities for embeddings using FiftyOne (Voxel51).

**Key Features**:
- Integration with FiftyOne for embedding visualization
- Dataset creation from database records
- Embedding similarity visualization
- Brain computation for clustering and similarity

**Main Functions**:
- `visualize_embeddings(db_handler)`: Create FiftyOne dataset from database and visualize embeddings

**Dependencies**: Requires `fiftyone` package to be installed.

**Usage Example**:
```python
from hailo_apps.python.core.common.embedding_visualizer import visualize_embeddings
from hailo_apps.python.core.common.db_handler import DatabaseHandler

db = DatabaseHandler(...)
visualize_embeddings(db)  # Opens FiftyOne app with visualization
```

---

## Communication & Integration

### `telegram_handler.py`
**Purpose**: Telegram bot integration for sending notifications and alerts.

**Key Features**:
- Send photo notifications via Telegram
- Rate limiting (1 notification per hour per person)
- Automatic image conversion from numpy arrays
- Configurable captions with confidence scores

**Main Class**:
- `TelegramHandler`: Manages Telegram bot communication

**Key Methods**:
- `send_notification(name, global_id, confidence, frame)`: Send notification with image
- `should_send_notification(global_id)`: Check if notification should be sent (rate limiting)

**Usage Example**:
```python
from hailo_apps.python.core.common.telegram_handler import TelegramHandler

handler = TelegramHandler(token="YOUR_BOT_TOKEN", chat_id="YOUR_CHAT_ID")

if handler.should_send_notification(person_id):
    handler.send_notification(
        name="Alice",
        global_id=person_id,
        confidence=0.95,
        frame=detected_frame
    )
```

---

## Module Organization

### Import Structure
All public utilities are exported through `__init__.py` for convenient importing:

```python
# Recommended: Import from common package (all exported functions)
from hailo_apps.python.core.common import (
    get_logger,
    init_logging,
    resolve_hef_path,
    get_resource_path,
    detect_hailo_arch,
    get_usb_video_devices,
    get_numpy_from_buffer,
    get_hef_input_size
)

# Or import specific modules (for classes and non-exported functions)
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.core.common.db_handler import DatabaseHandler
from hailo_apps.python.core.common.hailo_inference import HailoInfer
```

### Module Dependencies

**Core Dependencies** (used by most modules):
- `hailo_logger.py`: Used by all modules for logging
- `defines.py`: Constants used throughout

**External Dependencies**:
- `hailo_platform`: Required for `hailo_inference.py` and `hef_utils.py`
- `lancedb`: Required for `db_handler.py`
- `fiftyone`: Required for `embedding_visualizer.py`
- `telebot`: Required for `telegram_handler.py`
- `GStreamer` (gi.repository.Gst): Required for `buffer_utils.py`

---

## Best Practices

1. **Logging**: Always use `get_logger(__name__)` instead of `logging.getLogger(__name__)` to ensure consistent logging configuration.

2. **Resource Paths**:
   - For HEF models, use `resolve_hef_path()` which provides auto-download and better integration
   - For non-model resources (JSON, videos, photos, SO files), use `get_resource_path()`
   - Avoid hardcoding paths to ensure portability across installations

3. **Architecture Detection**: Use `detect_hailo_arch()` and `detect_host_arch()` for runtime detection rather than hardcoding values.

4. **Error Handling**: Most utility functions include comprehensive error handling and logging. Check return values and handle exceptions appropriately.

5. **Configuration**: Use environment variables and `.env` files for configuration. Load them using `load_environment()` at application startup.

---

## Testing

When adding new utilities to this directory:
- Follow PEP8 style guidelines
- Include type hints for all functions
- Add comprehensive docstrings (Google style)
- Write unit tests in the `tests/` directory
- Ensure backward compatibility when modifying existing APIs

---

## Version History

This directory contains stable, production-ready utilities. Breaking changes should be avoided, and deprecated functions should include migration guidance in their docstrings.

