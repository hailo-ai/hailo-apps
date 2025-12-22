# Hailo Apps Test Framework Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Directory Structure](#directory-structure)
4. [Configuration Files](#configuration-files)
5. [Test Categories](#test-categories)
6. [Running Tests](#running-tests)
7. [Adding New Apps](#adding-new-apps)
8. [Test Suite Modes & Model Selection](#test-suite-modes--model-selection)
9. [Placeholders](#placeholders)
10. [Troubleshooting](#troubleshooting)
11. [Best Practices](#best-practices)

---

## Overview

The Hailo Apps test framework is a **configuration-driven**, **dynamically-generated** testing system for validating pipeline applications across different architectures, models, and execution methods.

### Key Features

| Feature | Description |
|---------|-------------|
| **Dynamic Pipeline Discovery** | Pipelines are automatically loaded from `test_definition_config.yaml` |
| **Configuration-Driven** | All test behavior controlled via YAML files |
| **Multiple Test Categories** | Sanity, Installation, and Pipeline tests |
| **Flexible Test Selection** | Test suite modes (None/default/extra/all) and model selection |
| **Architecture Support** | hailo8, hailo8l, hailo10h with cross-architecture testing |
| **Multiple Run Methods** | pythonpath, cli, module execution |
| **Comprehensive Logging** | Per-app, per-test-suite logging directories |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TEST FRAMEWORK FLOW                              │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│ test_control.yaml│     │test_definition_  │     │ resources_config.yaml│
│    (tests/)      │     │config.yaml       │     │  (hailo_apps/config/)│
│                  │     │(hailo_apps/      │     │                      │
│ • What to run    │     │ config/)         │     │ • Models per app     │
│ • Run methods    │     │                  │     │ • Videos/images      │
│ • Custom tests   │     │ • App definitions│     │ • JSON configs       │
│ • Timeouts       │     │ • Test suites    │     │                      │
└────────┬─────────┘     │ • Flag combos    │     └──────────┬───────────┘
         │               └────────┬─────────┘                │
         │                        │                          │
         └────────────────────────┼──────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │     test_runner.py      │
                    │                         │
                    │ • Load configurations   │
                    │ • Detect architecture   │
                    │ • Generate test cases   │
                    │ • Execute via pytest    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │      all_tests.py       │
                    │                         │
                    │ • Dynamic pipeline      │
                    │   function generation   │
                    │ • Generic test runner   │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │      test_utils.py      │
                    │                         │
                    │ • Pipeline execution    │
                    │ • Process management    │
                    │ • Log file handling     │
                    └─────────────────────────┘
```

---

## Directory Structure

```
hailo-apps/
├── tests/
│   ├── conftest.py               # Pytest fixtures & config parsing
│   ├── test_control.yaml         # Test execution control (WHAT to run)
│   ├── test_runner.py            # Main pipeline test runner
│   ├── test_sanity_check.py      # Environment validation tests
│   ├── test_installation.py      # Installation & resources tests
│   ├── test_vlm_chat.py          # VLM chat unit tests
│   ├── all_tests.py              # Dynamic pipeline test functions
│   ├── test_utils.py             # Test execution utilities
│   ├── verify_configs.py         # Configuration verification
│   ├── README.md                 # Quick reference guide
│   └── TEST_FRAMEWORK_DOCUMENTATION.md  # This file
│
└── hailo_apps/
    └── config/
        ├── test_definition_config.yaml  # Test definitions (HOW to run)
        └── resources_config.yaml        # Models & resources
```

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `conftest.py` | Pytest fixtures, markers, YAML parsing utilities |
| `test_runner.py` | Main test orchestrator, generates parametrized test cases |
| `all_tests.py` | Dynamic pipeline test function generation from config |
| `test_utils.py` | Low-level pipeline execution (subprocess, signals, logging) |
| `test_sanity_check.py` | Environment & runtime validation |
| `test_installation.py` | Installation & resource validation |
| `verify_configs.py` | Configuration file validation script |

---

## Configuration Files

### 1. `test_control.yaml` - Controls WHAT to Run

Located in `tests/`, this file controls test execution:

```yaml
# Timing parameters
control_parameters:
  default_run_time: 40      # seconds per test
  term_timeout: 5           # termination timeout
  human_verification_run_time: 60

# Logging configuration
logging:
  base_dir: "./logs"
  level: "INFO"
  subdirs:
    per_app:
      detection:
        default: "./logs/detection/default"
        extra: "./logs/detection/extra"

# Enable predefined test combinations
test_combinations:
  ci_run:
    enabled: false
  all_default:
    enabled: false

# Custom per-app test configuration
custom_tests:
  enabled: true
  apps:
    detection:
      test_suite_mode: "default"  # None | default | extra | all
      model_selection: "default"  # default | extra | all

# Run method selection
run_methods:
  pythonpath:
    enabled: true
  cli:
    enabled: false
  module:
    enabled: false

# Special test modes
special_tests:
  h8l_on_h8:
    enabled: true  # Run hailo8l models on hailo8 hardware
```

### 2. `test_definition_config.yaml` - Defines HOW to Run

Located in `hailo_apps/config/`, defines apps and test suites:

```yaml
# App definitions - automatically discovered by all_tests.py
apps:
  detection:
    name: "Object Detection Pipeline"
    module: "hailo_apps.python.pipeline_apps.detection.detection"
    script: "hailo_apps/python/pipeline_apps/detection/detection.py"
    cli: "hailo-detect"
    default_test_suites:
      - "basic_show_fps"
      - "basic_input_video"
      - "basic_input_usb"
    extra_test_suites:
      - "input_video_with_labels"
      - "pipeline_disable_sync"

# Test suites with flag combinations
test_suites:
  basic_show_fps:
    flags:
      - "--show-fps"
    description: "Basic test with FPS display"
  
  input_video_with_hef:
    flags:
      - "--input"
      - "${VIDEO_PATH}"
      - "--hef-path"
      - "${HEF_PATH}"
      - "--show-fps"

# Predefined test run combinations
test_run_combinations:
  ci_run:
    apps: [detection, pose_estimation, depth]
    test_suite_mode: "all"
    model_selection: "all"
```

### 3. `resources_config.yaml` - Defines Resources

Located in `hailo_apps/config/`, defines models and resources:

```yaml
# Shared resources
videos:
  - name: example.mp4
    source: s3

images:
  - name: dog_bicycle.jpg
    source: s3

# Per-app model definitions
detection:
  models:
    hailo8:
      default:
        name: yolov8m
        source: mz
      extra:
        - name: yolov8s
          source: mz
    hailo8l:
      default:
        name: yolov8s
        source: mz
  json:
    - name: hailo_4_classes.json
      source: s3
```

---

## Test Categories

### 1. Sanity Checks (`test_sanity_check.py`)

Quick environment validation before running pipeline tests.

```bash
pytest tests/test_sanity_check.py -v
pytest -m sanity -v
```

| Test Class | Validates |
|------------|-----------|
| `TestHailoAppsPackage` | Package import, pip installation |
| `TestPythonEnvironment` | Python version, packages, HailoRT/TAPPAS bindings |
| `TestHailoRuntime` | hailortcli, device detection, architecture |
| `TestGStreamer` | GStreamer installation, critical elements |
| `TestEnvironmentConfiguration` | .env file, host arch, TAPPAS config |

### 2. Installation Tests (`test_installation.py`)

Validates installation and resources.

```bash
pytest tests/test_installation.py -v
pytest -m installation -v
```

| Test Class | Validates |
|------------|-----------|
| `TestDirectoryStructure` | Resources directories exist |
| `TestModelFiles` | HEF files downloaded and valid |
| `TestVideoFiles` | Video files downloaded and valid |
| `TestPostprocessSoFiles` | SO files compiled and valid ELF |
| `TestJsonConfigFiles` | JSON files exist and valid |

### 3. Pipeline Tests (`test_runner.py`)

Functional tests running actual pipelines.

```bash
pytest tests/test_runner.py -v
```

---

## Running Tests

### Quick Start

```bash
# Run all tests
./run_tests.sh

# Run specific categories
./run_tests.sh --sanity      # Environment checks only
./run_tests.sh --install     # Installation checks only
./run_tests.sh --pipelines   # Pipeline tests only
./run_tests.sh --no-download # Skip resource download
```

### Using pytest Directly

```bash
# Run by marker
pytest -m sanity -v
pytest -m installation -v

# Run specific test files
pytest tests/test_sanity_check.py -v
pytest tests/test_runner.py -v

# Run with verbose output
pytest tests/test_runner.py -vv -s

# Stop on first failure
pytest tests/test_runner.py -x

# Run specific test
pytest tests/test_runner.py -k "detection" -v
```

### Verify Configuration First

```bash
python3 tests/verify_configs.py
```

---

## Adding New Apps

Adding a new pipeline app requires **only YAML configuration changes** - no code modifications needed!

### Step 1: Add to `test_definition_config.yaml`

```yaml
apps:
  my_new_app:
    name: "My New Pipeline"
    description: "Description of the pipeline"
    module: "hailo_apps.python.pipeline_apps.my_new_app.my_new_app"
    script: "hailo_apps/python/pipeline_apps/my_new_app/my_new_app.py"
    cli: "hailo-my-new-app"
    default_test_suites:
      - "basic_show_fps"
      - "basic_input_video"
    extra_test_suites:
      - "pipeline_disable_sync"
```

### Step 2: Add to `resources_config.yaml`

```yaml
my_new_app:
  models:
    hailo8:
      default:
        name: my_model
        source: mz
    hailo8l:
      default:
        name: my_model
        source: mz
  json:
    - name: my_config.json
      source: s3
```

### Step 3: Add to `test_control.yaml`

```yaml
logging:
  subdirs:
    per_app:
      my_new_app:
        default: "./logs/my_new_app/default"
        extra: "./logs/my_new_app/extra"

custom_tests:
  apps:
    my_new_app:
      test_suite_mode: "default"
      model_selection: "default"
```

### Step 4: Verify

```bash
python3 tests/verify_configs.py
```

**That's it!** The framework automatically discovers the new app from `test_definition_config.yaml`.

---

## Test Suite Modes & Model Selection

### Test Suite Modes

| Mode | Description |
|------|-------------|
| `None` | Skip this app entirely |
| `default` | Run only `default_test_suites` |
| `extra` | Run only `extra_test_suites` |
| `all` | Run both default and extra |

### Model Selection

| Selection | Description |
|-----------|-------------|
| `default` | Test only default model(s) |
| `extra` | Test only extra models |
| `all` | Test all models |

### Examples

```yaml
# Quick smoke test
custom_tests:
  apps:
    detection:
      test_suite_mode: "default"
      model_selection: "default"

# Comprehensive test
custom_tests:
  apps:
    detection:
      test_suite_mode: "all"
      model_selection: "all"

# Skip an app
custom_tests:
  apps:
    detection:
      test_suite_mode: "None"
```

---

## Placeholders

Test suites support runtime placeholders:

| Placeholder | Resolved To |
|-------------|-------------|
| `${VIDEO_PATH}` | `/usr/local/hailo/resources/videos/example.mp4` |
| `${HEF_PATH}` | `/usr/local/hailo/resources/models/{arch}/{model}.hef` |
| `${LABELS_JSON_PATH}` | `/usr/local/hailo/resources/json/{app_json}.json` |
| `${RESOURCES_ROOT}` | `/usr/local/hailo/resources` |

### Example Usage

```yaml
test_suites:
  input_video_with_hef:
    flags:
      - "--input"
      - "${VIDEO_PATH}"
      - "--hef-path"
      - "${HEF_PATH}"
      - "--show-fps"
```

---

## Troubleshooting

### No Tests Generated

1. Check `custom_tests.enabled: true` in `test_control.yaml`
2. Verify `test_suite_mode` is not `"None"`
3. Check models exist in `resources_config.yaml` for your architecture
4. Verify Hailo device is detected: `hailortcli fw-control identify`

### Test Suite Not Found

1. Verify suite name exists in `test_definition_config.yaml`
2. Check suite is in `default_test_suites` or `extra_test_suites`
3. Ensure `test_suite_mode` includes the suite

### Model Not Found

1. Verify model in `resources_config.yaml`
2. Check `model_selection` includes the model
3. Ensure model HEF is downloaded

### Import Errors

```bash
# Ensure package is installed
pip install -e .

# Activate environment
source setup_env.sh

# Run sanity checks
pytest tests/test_sanity_check.py -v
```

### Verify Configuration

```bash
python3 tests/verify_configs.py
```

---

## Best Practices

1. **Run sanity checks first**
   ```bash
   pytest tests/test_sanity_check.py -v
   ```

2. **Start with defaults** - Use `test_suite_mode: "default"` for quick validation

3. **Verify config before tests** - Run `verify_configs.py` after config changes

4. **Use test combinations** - Create reusable combinations in `test_definition_config.yaml`

5. **Check logs on failure** - Logs are in `./logs/{app}/{mode}/`

6. **Use placeholders** - Never hardcode paths in test suites

7. **Add new apps via config only** - No code changes needed for new pipelines

---

## Test Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.sanity` | Quick environment checks |
| `@pytest.mark.installation` | Installation validation |
| `@pytest.mark.resources` | Resource file validation |
| `@pytest.mark.requires_device` | Requires Hailo device |
| `@pytest.mark.requires_gstreamer` | Requires GStreamer |

---

## Summary

The test framework provides:

- ✅ **Zero-code app addition** - Just update YAML configs
- ✅ **Dynamic pipeline discovery** - Apps loaded from config at runtime
- ✅ **Flexible test selection** - Suite modes and model selection
- ✅ **Multiple execution methods** - pythonpath, cli, module
- ✅ **Architecture support** - hailo8, hailo8l, hailo10h
- ✅ **Cross-architecture testing** - h8l_on_h8 special mode
- ✅ **Comprehensive logging** - Per-app, per-suite log directories
- ✅ **Configuration validation** - verify_configs.py script

For issues, run `verify_configs.py` or check logs in `./logs/`.
