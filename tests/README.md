# Test Framework Documentation

## Overview

The Hailo Apps Infrastructure test framework is a comprehensive, configuration-driven testing system designed to validate the installation, environment, and pipeline applications across different architectures, models, and execution methods.

The framework consists of six test categories:

| Test File | Purpose | Markers |
|-----------|---------|--------|
| `test_sanity_check.py` | Environment & runtime validation | `@pytest.mark.sanity` |
| `test_config_integrity.py` | Configuration cross-validation | `@pytest.mark.sanity` |
| `test_installation.py` | Installation & resources validation | `@pytest.mark.installation`, `@pytest.mark.resources` |
| `test_runner.py` | Pipeline functional tests | `@pytest.mark.pipeline`, `@pytest.mark.requires_device` |
| `test_standalone_runner.py` | Standalone app smoke tests | `@pytest.mark.standalone`, `@pytest.mark.requires_device` |
| `test_gen_ai.py` | GenAI app integration tests | `@pytest.mark.genai` |

## Table of Contents

1. [Quick Start](#quick-start)
2. [Test Categories](#test-categories)
3. [Running Tests](#running-tests)
4. [Configuration Files](#configuration-files)
5. [Test Framework Architecture](#test-framework-architecture)
6. [Adding New Tests](#adding-new-tests)
7. [Troubleshooting](#troubleshooting)

## Quick Start

### Run All Tests

```bash
./run_tests.sh
```

### Run Specific Test Categories

```bash
# Run only environment sanity checks
./run_tests.sh --sanity

# Run only installation/resource validation
./run_tests.sh --install

# Run only pipeline functional tests
./run_tests.sh --pipelines

# Run only standalone app smoke tests
./run_tests.sh --standalone

# Run only GenAI tests
./run_tests.sh --genai

# Run specific apps only (pipeline + standalone)
./run_tests.sh --apps detection,pose_estimation

# Combine suites
./run_tests.sh --pipelines --standalone

# Skip resource download
./run_tests.sh --no-download
```

### Run with pytest Directly

```bash
# Run sanity checks + config integrity
pytest tests/test_sanity_check.py tests/test_config_integrity.py -v

# Run installation tests
pytest tests/test_installation.py -v

# Run pipeline tests
pytest tests/test_runner.py -v

# Run standalone tests
pytest tests/test_standalone_runner.py -v

# Run GenAI tests
pytest tests/test_gen_ai.py -v

# Run tests with specific markers
pytest -m sanity -v
pytest -m pipeline -v
pytest -m standalone -v
pytest -m genai -v
pytest -m "not requires_device" -v

# Run a specific app's tests
pytest tests/test_runner.py -k "detection" -v
pytest tests/test_standalone_runner.py -k "object_detection" -v
```

## Test Categories

### 1. Sanity Checks (`test_sanity_check.py`)

Quick environment validation tests that verify the runtime is properly configured **before** running any actual pipeline tests.

**Test Classes:**

| Class | Tests |
|-------|-------|
| `TestHailoAppsPackage` | Package import, pip installation |
| `TestPythonEnvironment` | Python version, critical packages (gi, numpy, opencv, yaml), HailoRT/GStreamer bindings |
| `TestHailoRuntime` | hailortcli availability, device detection, architecture validation |
| `TestGStreamer` | GStreamer installation, critical elements, Hailo elements |
| `TestEnvironmentConfiguration` | .env file, host arch detection, GStreamer variant, postproc path |
| `TestHostArchitectureSpecific` | RPi-specific tests (picamera2, libcamera) |

**Run:**
```bash
pytest tests/test_sanity_check.py -v -m sanity
```

### Installation Tests (`test_installation.py`)

Validates that `hailo-post-install` completed successfully and all resources are properly downloaded and configured.

**Test Classes:**

| Class | Tests |
|-------|-------|
| `TestDirectoryStructure` | Resources root, symlink, models/videos/so/json directories |
| `TestModelFiles` | Default models downloaded, HEF file validity |
| `TestVideoFiles` | Expected videos downloaded, validity |
| `TestImageFiles` | Expected images downloaded |
| `TestJsonConfigFiles` | JSON files downloaded, valid JSON format |
| `TestPostprocessSoFiles` | SO files compiled (from meson.build), valid ELF |
| `TestConfigFiles` | resources_config.yaml, meson.build existence |
| `TestIntegrationSmoke` | Package imports, HEF loading test |

**Dynamic Parsing:**
- Models, videos, images, JSON files are parsed from `resources_config.yaml`
- Expected SO files are parsed from `postprocess/cpp/meson.build`

**Run:**
```bash
pytest tests/test_installation.py -v -m installation
```

### 2. Config Integrity Tests (`test_config_integrity.py`)

Automated cross-validation of the three configuration YAML files. Catches typos, missing app entries, and drift between configs **before** any pipeline runs.

**Test Classes:**

| Class | Tests |
|-------|-------|
| `TestConfigFilesExist` | Verifies all 3 YAML files exist on disk |
| `TestConfigsLoadable` | Ensures every YAML file parses without errors |
| `TestConfigCrossValidation` | Apps in `test_control.yaml` exist in `test_definition_config.yaml`; apps with `resources_config.yaml` entries have resources defined; every standalone app has a matching base entry |

**Run:**
```bash
pytest tests/test_config_integrity.py -v
```

### 3. Pipeline Tests (`test_runner.py`)

Functional tests that run actual GStreamer pipeline applications with different configurations.

**Configuration Files:**
- `test_control.yaml` - Controls what tests to run
- `test_definition_config.yaml` - Defines test suites and app configurations
- `resources_config.yaml` - Defines models and resources

**Run:**
```bash
# All pipelines
pytest tests/test_runner.py -v

# Specific app
pytest tests/test_runner.py -k "detection" -v
```

### 4. Standalone App Tests (`test_standalone_runner.py`)

Smoke tests for standalone (non-GStreamer) Python applications. Each test launches the standalone script as a subprocess, waits for the configured run time, then checks the exit code.

**Run:**
```bash
# All standalone tests
pytest tests/test_standalone_runner.py -v

# Specific standalone app
pytest tests/test_standalone_runner.py -k "object_detection" -v
```

### 5. GenAI Tests (`test_gen_ai.py`)

Integration tests for generative-AI applications: LLM text generation, VLM image understanding, Whisper speech-to-text, voice assistant, and the agent framework.

**Run:**
```bash
pytest tests/test_gen_ai.py -v
```

## Running Tests

### Using `run_tests.sh`

The `run_tests.sh` script is the recommended way to run tests. It:
1. Activates the virtual environment
2. Installs test dependencies
3. Downloads resources for the **detected architecture only**
4. Runs tests in order: sanity → installation → pipelines

```bash
# Run default suite (sanity + install + pipelines)
./run_tests.sh

# Run only sanity checks
./run_tests.sh --sanity

# Run only installation tests
./run_tests.sh --install

# Run only pipeline tests
./run_tests.sh --pipelines

# Run standalone app tests
./run_tests.sh --standalone

# Run GenAI tests
./run_tests.sh --genai

# Combine suites (e.g., pipelines + standalone)
./run_tests.sh --pipelines --standalone

# Filter to specific apps (applies to pipelines + standalone)
./run_tests.sh --apps detection,pose_estimation

# Skip resource download
./run_tests.sh --no-download
```

**Note:** The script does NOT automatically download hailo8l resources. If you need to test hailo8l models on hailo8 devices, run manually:
```bash
python -m hailo_apps.installation.download_resources --arch hailo8l
```

### Using pytest Directly

```bash
# Run all sanity tests (environment + config integrity)
pytest tests/test_sanity_check.py tests/test_config_integrity.py -v

# Run all installation tests
pytest tests/test_installation.py -v

# Run pipeline tests
pytest tests/test_runner.py -v

# Run standalone app smoke tests
pytest tests/test_standalone_runner.py -v

# Run GenAI tests
pytest tests/test_gen_ai.py -v

# Run tests with specific markers
pytest -m sanity -v
pytest -m pipeline -v
pytest -m standalone -v
pytest -m genai -v
pytest -m installation -v
pytest -m resources -v
pytest -m "not requires_device" -v

# Run specific test class
pytest tests/test_sanity_check.py::TestPythonEnvironment -v

# Run specific test
pytest tests/test_installation.py::TestModelFiles::test_hef_files_valid -v

# Run with verbose output
pytest tests/test_sanity_check.py -vv

# Run with output capture disabled
pytest tests/test_sanity_check.py -v -s

# Stop on first failure
pytest tests/test_sanity_check.py -x
```

### Test Markers

The framework defines custom pytest markers (registered in `pyproject.toml`):

| Marker | Description |
|--------|-------------|
| `@pytest.mark.sanity` | Quick environment sanity checks |
| `@pytest.mark.installation` | Installation validation tests |
| `@pytest.mark.resources` | Resource file validation tests |
| `@pytest.mark.requires_device` | Tests requiring a Hailo device |
| `@pytest.mark.requires_gstreamer` | Tests requiring GStreamer |
| `@pytest.mark.pipeline` | GStreamer pipeline functional tests |
| `@pytest.mark.standalone` | Standalone app smoke tests |
| `@pytest.mark.genai` | GenAI app tests |

## Configuration Files

### Directory Structure

```
tests/
├── conftest.py                  # Shared fixtures, markers, teardown hooks
├── test_sanity_check.py         # Environment validation
├── test_config_integrity.py     # Config cross-validation
├── test_installation.py         # Installation/resource validation
├── test_runner.py               # Pipeline functional tests
├── test_standalone_runner.py    # Standalone app smoke tests
├── test_gen_ai.py               # GenAI integration tests
├── test_control.yaml            # Test execution control
├── all_tests.py                 # Pipeline test functions
├── test_utils.py                # Test utilities (subprocess helpers)
├── verify_configs.py            # Legacy configuration verification
├── voice_assistant_unit_tests.py # Voice assistant unit tests
└── README.md                    # This file

hailo_apps/config/
├── test_definition_config.yaml  # Test definitions
├── resources_config.yaml        # Resources configuration
└── config_manager.py            # Unified config access

hailo_apps/postprocess/cpp/
└── meson.build                  # SO file definitions (parsed for expected SO files)
```

### `test_control.yaml` - Test Execution Control

Controls **what** to run:

```yaml
# Control parameters
control_parameters:
  default_run_time: 40
  term_timeout: 5

# Test combinations (presets override per-app settings)
test_combinations:
  ci_run:
    enabled: false
  all_default:
    enabled: false

# Custom per-app tests
custom_tests:
  enabled: true
  apps:
    detection:
      test_suite_mode: "default"  # None | default | extra | all
      model_selection: "default"  # default | extra | all
    pose_estimation:
      test_suite_mode: "default"
      model_selection: "default"

# Per-app run-time overrides (seconds)
run_time_overrides:
  face_recognition: 60
  paddle_ocr: 60

# Run methods
run_methods:
  pythonpath:
    enabled: true
  cli:
    enabled: false

# Standalone tests
standalone_tests:
  enabled: true
  apps:
    object_detection_standalone:
      test_suite_mode: "default"

# GenAI tests
genai_tests:
  enabled: false
  apps:
    llm:
      enabled: true
    vlm:
      enabled: true

# Special tests
special_tests:
  h8l_on_h8:
    enabled: true
  sanity_checks:
    enabled: true
```

### `resources_config.yaml` - Resources Configuration

Defines models and resources (dynamically parsed by tests):

```yaml
# Videos (shared)
videos:
  - name: example.mp4
    source: s3
  - name: face_recognition.mp4
    source: s3

# Images (shared)
images:
  - name: dog_bicycle.jpg
    source: s3

# Per-app models
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

### `meson.build` - SO File Definitions

The `postprocess/cpp/meson.build` file is parsed to extract expected `.so` files:

```meson
shared_library('yolo_hailortpp_postprocess', ...)
shared_library('depth_postprocess', ...)
# Results in expected: libyolo_hailortpp_postprocess.so, libdepth_postprocess.so, etc.
```

## Test Framework Architecture

### Data Flow

```
                    ┌─────────────────────────┐
                    │  resources_config.yaml  │
                    │  (models, videos, etc)  │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼──────────────┐
                    │       conftest.py        │
                    │ (parse_resources_config) │
                    │ (parse_meson_shared_     │
                    │   libraries)             │
                    └───────────┬──────────────┘
                                │
   ┌────────────┬───────────────┼───────────────┬────────────────┬──────────────┐
   │            │               │               │                │              │
   ▼            ▼               ▼               ▼                ▼              ▼
┌────────┐ ┌──────────┐ ┌────────────┐ ┌─────────────┐ ┌────────────┐ ┌────────┐
│sanity  │ │config    │ │install-    │ │test_runner  │ │standalone  │ │gen_ai  │
│check   │ │integrity │ │ation      │ │(pipelines)  │ │_runner     │ │        │
│        │ │          │ │            │ │             │ │            │ │        │
│•Env    │ │•YAML ok  │ │•Dir struct │ │•Functional  │ │•Smoke test │ │•LLM    │
│•Pkgs   │ │•Cross-   │ │•Models    │ │ tests       │ │•Subprocess │ │•VLM    │
│•GStr   │ │ validate │ │•Videos    │ │•All apps    │ │•Exit code  │ │•Whisper│
│•SDK    │ │•Drift    │ │•SO files  │ │             │ │            │ │•Agent  │
└────────┘ └──────────┘ └────────────┘ └─────────────┘ └────────────┘ └────────┘
```

### Shared Fixtures (conftest.py)

The `conftest.py` file provides:

- **Configuration Parsing:**
  - `parse_resources_config()` - Parses resources_config.yaml
  - `parse_meson_shared_libraries()` - Parses meson.build for SO files

- **Fixtures:**
  - `resources_config` - Parsed resources configuration
  - `expected_videos` - List of expected video files
  - `expected_images` - List of expected image files
  - `expected_so_files` - List of expected SO files
  - `expected_models_for_arch` - Expected models for detected architecture
  - `expected_json_files` - Expected JSON files
  - `detected_hailo_arch` - Detected Hailo architecture (or None)
  - `detected_host_arch` - Detected host architecture
  - `resources_root_path` - Path to resources root
  - `dotenv_path` - Path to .env file

## Adding New Tests

### Adding a Sanity Check

Add to `test_sanity_check.py`:

```python
@pytest.mark.sanity
class TestMyNewCheck:
    def test_my_feature(self):
        """Description of what this tests."""
        # Test implementation
        assert condition, "Error message"
```

### Adding an Installation Test

Add to `test_installation.py`:

```python
@pytest.mark.installation
class TestMyNewResourceCheck:
    def test_my_resource_exists(self, resources_root_path):
        """Verify my resource exists."""
        resource_path = resources_root_path / "my_resource"
        assert resource_path.exists(), f"Resource missing: {resource_path}"
```

### Adding Resources to Validation

Resources are automatically validated by adding them to `resources_config.yaml`:

```yaml
# Add a new video
videos:
  - name: my_new_video.mp4
    source: s3

# Add a new app with models
my_new_app:
  models:
    hailo8:
      default:
        name: my_model
        source: mz
  json:
    - name: my_config.json
      source: s3
```

### Adding SO Files to Validation

Add `shared_library()` calls to `postprocess/cpp/meson.build`:

```meson
shared_library('my_postprocess',
    my_postprocess_sources,
    dependencies : postprocess_dep,
    gnu_symbol_visibility : 'default',
    install: true,
    install_dir: '/usr/local/hailo/resources/so',
)
```

The test framework will automatically expect `libmy_postprocess.so`.

## Troubleshooting

### No Tests Collected

**Problem:** pytest reports no tests collected.

**Solutions:**
1. Check pytest markers are correct
2. Verify test class/function names start with `test_` or `Test`
3. Ensure `conftest.py` is in the tests directory

### Import Errors

**Problem:** Import errors when running tests.

**Solutions:**
1. Ensure package is installed: `pip install -e .`
2. Activate virtual environment: `source setup_env.sh`
3. Run sanity checks: `pytest tests/test_sanity_check.py -v`

### Config Drift

**Problem:** A new app was added but tests skip or fail with "not found in config".

**Solutions:**
1. Run config integrity tests: `pytest tests/test_config_integrity.py -v`
2. Ensure the app appears in all three YAML files:
   - `test_control.yaml` (custom_tests.apps)
   - `test_definition_config.yaml` (app test suites)
   - `resources_config.yaml` (models/resources)

### Missing Resources

**Problem:** Tests fail due to missing resources.

**Solutions:**
1. Run: `python -m hailo_apps.installation.download_resources`
2. Check internet connection
3. Verify `resources_config.yaml` has correct entries

### Device Not Detected

**Problem:** Hailo device not detected, tests skipped.

**Solutions:**
1. Check device connection: `hailortcli fw-control identify`
2. Verify HailoRT is installed
3. Check device permissions

### Verification Script

The legacy `verify_configs.py` script is replaced by automated `test_config_integrity.py` tests. Run:

```bash
pytest tests/test_config_integrity.py -v
```

## Best Practices

1. **Run sanity checks first:** Always run `test_sanity_check.py` + `test_config_integrity.py` before other tests
2. **Check installation:** Run `test_installation.py` to verify resources before pipeline tests
3. **Use markers:** Use `-m pipeline`, `-m standalone`, `-m genai` to run specific categories
4. **Filter by app:** Use `--apps detection` in `run_tests.sh` or `-k detection` with pytest
5. **Review logs:** Check `tests/tests_logs/` for detailed output
6. **Validate config:** Run `pytest tests/test_config_integrity.py` after any config changes

## See Also

- `test_control.yaml` - Test execution control configuration
- `test_definition_config.yaml` - Test definitions and app configurations
- `resources_config.yaml` - Resources and models configuration
- `test_config_integrity.py` - Automated config cross-validation
- `TEST_FRAMEWORK_DOCUMENTATION.md` - Additional detailed documentation
