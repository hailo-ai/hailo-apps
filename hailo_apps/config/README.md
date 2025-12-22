# Hailo Apps Configuration Module

This module provides centralized configuration management for the Hailo Apps Infrastructure project.

## Overview

The configuration system manages four YAML configuration files:

| File | Purpose | Location |
|------|---------|----------|
| `config.yaml` | Main installation/runtime settings | `hailo_apps/config/` |
| `resources_config.yaml` | Model, video, image resource definitions | `hailo_apps/config/` |
| `test_definition_config.yaml` | Test framework structure and suites | `hailo_apps/config/` |
| `test_control.yaml` | Test execution control settings | `tests/` |

## Quick Start

### Python API

```python
from hailo_apps.config import config_manager

# Get available applications
apps = config_manager.get_available_apps()
# ['clip', 'depth', 'detection', 'face_recognition', ...]

# Get models for an app and architecture
models = config_manager.get_default_models("detection", "hailo8")
# [ModelEntry(name='yolov8m', source='mz', url=None)]

# Get model names only
names = config_manager.get_model_names("detection", "hailo8", tier="all")
# ['yolov8m', 'yolov5m_wo_spp', 'yolov8s', ...]

# Get test suite configuration
suite = config_manager.get_test_suite("basic_show_fps")
# TestSuite(name='basic_show_fps', description='Basic test with FPS display', flags=('--show-fps',))

# Get control parameters
run_time = config_manager.get_control_parameter("default_run_time", default=24)
# 40
```

### CLI Interface

```bash
# Validate all configurations (dry run)
python -m hailo_apps.config.config_manager --dry-run

# List all available applications
python -m hailo_apps.config.config_manager --list-apps

# Show models for a specific app and architecture
python -m hailo_apps.config.config_manager --show-models detection hailo8

# Show configuration file paths
python -m hailo_apps.config.config_manager --show-paths
```

## API Reference

### Main Config (`config.yaml`)

```python
# Get complete config
config = config_manager.get_main_config()

# Get valid versions for validation
versions = config_manager.get_valid_versions("hailort")  # ["4.23.0", "5.1.1"]

# Get Model Zoo version for architecture
mz_version = config_manager.get_model_zoo_version_for_arch("hailo8")  # "v2.17.0"

# Get venv configuration
venv_config = config_manager.get_venv_config()
# {"name": "venv_hailo_apps", "use_system_site_packages": True}
```

### Resources Config (`resources_config.yaml`)

```python
# List all applications
apps = config_manager.get_available_apps()

# Get supported architectures for an app
archs = config_manager.get_supported_architectures("detection")
# ["hailo8", "hailo8l", "hailo10h"]

# Get models (returns ModelEntry dataclass)
default_models = config_manager.get_default_models("detection", "hailo8")
extra_models = config_manager.get_extra_models("detection", "hailo8")
all_models = config_manager.get_all_models("detection", "hailo8")

# Get just model names
names = config_manager.get_model_names("detection", "hailo8", tier="default")

# Get first default model name
name = config_manager.get_default_model_name("detection", "hailo8")

# Get specific model info
model = config_manager.get_model_info("detection", "hailo8", "yolov8m")
# ModelEntry(name='yolov8m', source='mz', url=None)

# Get shared resources
videos = config_manager.get_videos()    # ["example.mp4", ...]
images = config_manager.get_images()    # ["dog_bicycle.jpg", ...]

# Get all shared JSON files
json_files = config_manager.get_json_files()  # ["hailo_4_classes.json", "scrfd.json", ...]

# Check if app is Gen-AI
is_gen_ai = config_manager.is_gen_ai_app("vlm_chat")  # True
```

### Test Definition Config (`test_definition_config.yaml`)

```python
# Get app definition
app_def = config_manager.get_app_definition("detection")
# AppDefinition(name="Object Detection Pipeline", module="...", cli="hailo-detect", ...)

# List defined apps
apps = config_manager.get_defined_apps()

# Get test suite
suite = config_manager.get_test_suite("basic_show_fps")
# TestSuite(name="basic_show_fps", flags=("--show-fps",), description="...")

# List all test suites
suites = config_manager.get_all_test_suites()

# Get test suites for an app
suites = config_manager.get_test_suites_for_app("detection", mode="default")
suites = config_manager.get_test_suites_for_app("detection", mode="extra")
suites = config_manager.get_test_suites_for_app("detection", mode="all")

# Get test run combination
combo = config_manager.get_test_run_combination("ci_run")
# {"name": "CI Run", "apps": [...], "test_suite_mode": "all", ...}
```

### Test Control Config (`test_control.yaml`)

```python
# Get control parameter
run_time = config_manager.get_control_parameter("default_run_time", default=24)
timeout = config_manager.get_control_parameter("term_timeout", default=5)

# Get logging config
log_config = config_manager.get_logging_config()

# Get enabled run methods
methods = config_manager.get_enabled_run_methods()  # ["pythonpath"]

# Get custom test apps config
custom_apps = config_manager.get_custom_test_apps()

# Check special tests
enabled = config_manager.is_special_test_enabled("h8l_on_h8")  # True/False

# Get enabled test combinations
combos = config_manager.get_enabled_test_combinations()
```

### Cache Management

```python
# Clear cached configurations (useful when files are modified)
config_manager.clear_cache()

# Reload all configurations
config_manager.reload_all()
```

## Data Classes

### ModelEntry

```python
@dataclass(frozen=True)
class ModelEntry:
    name: str       # Model name (e.g., "yolov8m")
    source: str     # "mz" (Model Zoo), "s3", or "gen-ai-mz"
    url: str | None # Optional explicit download URL
```

### AppDefinition

```python
@dataclass(frozen=True)
class AppDefinition:
    name: str                           # Display name
    description: str                    # Brief description
    module: str                         # Python module path
    script: str                         # Script path relative to repo
    cli: str                            # CLI command name
    default_test_suites: tuple[str, ...] # Default test suites
    extra_test_suites: tuple[str, ...]   # Extra test suites
```

### TestSuite

```python
@dataclass(frozen=True)
class TestSuite:
    name: str              # Suite identifier
    description: str       # Brief description
    flags: tuple[str, ...] # Command-line flags
```

## Path Resolution

The `ConfigPaths` class provides centralized path resolution:

```python
from hailo_apps.config.config_manager import ConfigPaths

# Get paths
repo_root = ConfigPaths.repo_root()
main_config = ConfigPaths.main_config()
resources_config = ConfigPaths.resources_config()
test_definition = ConfigPaths.test_definition_config()
test_control = ConfigPaths.test_control_config()
```

## Error Handling

All configuration loading functions raise `ConfigError` for:
- Missing configuration files
- Invalid YAML syntax
- Required fields missing

```python
from hailo_apps.config.config_manager import ConfigError

try:
    config = config_manager.get_main_config()
except ConfigError as e:
    print(f"Configuration error: {e}")
```

## Testing Configuration

Use `--dry-run` to validate all configurations:

```bash
$ python -m hailo_apps.config.config_manager --dry-run

🔍 Configuration Manager Dry Run
======================================================================

======================================================================
  Configuration Files
======================================================================
  ✅ Main Config: /path/to/hailo_apps/config/config.yaml
  ✅ Resources Config: /path/to/hailo_apps/config/resources_config.yaml
  ✅ Test Definition Config: /path/to/hailo_apps/config/test_definition_config.yaml
  ✅ Test Control Config: /path/to/tests/test_control.yaml

======================================================================
  Summary
======================================================================
  ✅ No errors found
  ✅ No warnings
```

## Import Convenience

You can also import common functions from the installation module:

```python
from hailo_apps.installation import (
    get_available_apps,
    get_model_names,
    get_default_model_name,
    is_gen_ai_app,
)

# These are re-exports from config_manager
apps = get_available_apps()
models = get_model_names("detection", "hailo8", tier="default")
```

## Architecture

```
hailo_apps/config/
├── __init__.py              # Package exports
├── config_manager.py        # Unified configuration manager
├── config.yaml              # Main installation/runtime config
├── resources_config.yaml    # Resource definitions
├── test_definition_config.yaml  # Test framework definitions
└── README.md                # This file

tests/
└── test_control.yaml        # Test execution control
```

## Best Practices

1. **Use the unified API**: Always import from `config_manager` instead of direct YAML loading
2. **Cache awareness**: The config manager caches by default; use `clear_cache()` if files change
3. **Type safety**: Use the dataclass return types (`ModelEntry`, `AppDefinition`, `TestSuite`)
4. **Dry run first**: Run `--dry-run` after editing config files to validate
5. **Cross-reference**: Use `get_available_apps()` from resources and `get_defined_apps()` from definitions to ensure consistency

