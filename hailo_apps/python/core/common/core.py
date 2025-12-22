"""Core helpers: arch detection, parser, buffer utils, model resolution."""

import os
import queue
import sys
from pathlib import Path

from dataclasses import dataclass
from dotenv import load_dotenv
from . import parser as common_parser

from .defines import (
    DEFAULT_DOTENV_PATH,
    DEFAULT_LOCAL_RESOURCES_PATH,
    DEPTH_MODEL_NAME,
    DEPTH_PIPELINE,
    DETECTION_MODEL_NAME_H8,
    DETECTION_MODEL_NAME_H8L,
    DETECTION_PIPELINE,
    DIC_CONFIG_VARIANTS,
    FACE_DETECTION_MODEL_NAME_H8,
    FACE_DETECTION_MODEL_NAME_H8L,
    FACE_DETECTION_PIPELINE,
    FACE_RECOGNITION_MODEL_NAME_H8,
    FACE_RECOGNITION_MODEL_NAME_H8L,
    FACE_RECOGNITION_PIPELINE,
    HAILO8_ARCH,
    HAILO8L_ARCH,
    HAILO10H_ARCH,
    HAILO_ARCH_KEY,
    HAILO_FILE_EXTENSION,
    INSTANCE_SEGMENTATION_MODEL_NAME_H8,
    INSTANCE_SEGMENTATION_MODEL_NAME_H8L,
    INSTANCE_SEGMENTATION_PIPELINE,
    POSE_ESTIMATION_MODEL_NAME_H8,
    POSE_ESTIMATION_MODEL_NAME_H8L,
    POSE_ESTIMATION_PIPELINE,
    RESOURCES_JSON_DIR_NAME,
    RESOURCES_MODELS_DIR_NAME,
    # for get_resource_path
    RESOURCES_PHOTOS_DIR_NAME,
    RESOURCES_ROOT_PATH_DEFAULT,
    RESOURCES_SO_DIR_NAME,
    RESOURCES_VIDEOS_DIR_NAME,
    SIMPLE_DETECTION_MODEL_NAME,
    SIMPLE_DETECTION_PIPELINE,
    CLIP_PIPELINE,
    CLIP_MODEL_NAME,
    CLIP_DETECTION_PIPELINE,
    CLIP_DETECTION_MODEL_NAME
)
from .hailo_logger import get_logger
from .installation_utils import detect_hailo_arch

try:
    from hailo_apps.config.config_manager import get_default_models, get_extra_models, get_all_models
except ImportError:
    import sys
    from pathlib import Path
    config_dir = Path(__file__).resolve().parents[3] / "config"
    sys.path.insert(0, str(config_dir))
    from config_manager import get_default_models, get_extra_models, get_all_models

hailo_logger = get_logger(__name__)


def load_environment(env_file=DEFAULT_DOTENV_PATH, required_vars=None) -> bool:
    hailo_logger.debug(f"Loading environment from: {env_file}")
    if env_file is None:
        env_file = DEFAULT_DOTENV_PATH
    load_dotenv(dotenv_path=env_file)

    env_path = Path(env_file)
    if not os.path.exists(env_path):
        hailo_logger.warning(f".env file not found: {env_file}")
        return False
    if not os.access(env_path, os.R_OK):
        hailo_logger.warning(f".env file not readable: {env_file}")
        return False
    if not os.access(env_path, os.W_OK):
        hailo_logger.warning(f".env file not writable: {env_file}")
        return False
    if not os.access(env_path, os.F_OK):
        hailo_logger.warning(f".env file not found (F_OK): {env_file}")
        return False

    if required_vars is None:
        required_vars = DIC_CONFIG_VARIANTS
    missing = []
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing.append(var)

    if missing:
        hailo_logger.warning(f"Missing environment variables: {missing}")
        return False
    hailo_logger.info("All required environment variables loaded successfully.")
    return True


def get_base_parser():
    """Proxy to the shared base parser implementation."""
    return common_parser.get_base_parser()


def get_pipeline_parser():
    """Proxy to the shared pipeline parser implementation."""
    return common_parser.get_pipeline_parser()


def get_standalone_parser():
    """Proxy to the shared standalone parser implementation."""
    return common_parser.get_standalone_parser()


def get_default_parser():
    """Legacy proxy preserved for backward compatibility."""
    return common_parser.get_default_parser()


def configure_multi_model_hef_path(parser):
    """Proxy to configure --hef-path for multi-model apps."""
    return common_parser.configure_multi_model_hef_path(parser)


def get_model_name(pipeline_name: str, arch: str) -> str:
    hailo_logger.debug(f"Getting model name for pipeline={pipeline_name}, arch={arch}")
    is_h8 = arch in (HAILO8_ARCH, HAILO10H_ARCH)
    pipeline_map = {
        DEPTH_PIPELINE: DEPTH_MODEL_NAME,
        CLIP_PIPELINE: CLIP_MODEL_NAME,
        CLIP_DETECTION_PIPELINE: CLIP_DETECTION_MODEL_NAME,
        SIMPLE_DETECTION_PIPELINE: SIMPLE_DETECTION_MODEL_NAME,
        DETECTION_PIPELINE: DETECTION_MODEL_NAME_H8 if is_h8 else DETECTION_MODEL_NAME_H8L,
        INSTANCE_SEGMENTATION_PIPELINE: INSTANCE_SEGMENTATION_MODEL_NAME_H8
        if is_h8
        else INSTANCE_SEGMENTATION_MODEL_NAME_H8L,
        POSE_ESTIMATION_PIPELINE: POSE_ESTIMATION_MODEL_NAME_H8
        if is_h8
        else POSE_ESTIMATION_MODEL_NAME_H8L,
        FACE_DETECTION_PIPELINE: FACE_DETECTION_MODEL_NAME_H8
        if is_h8
        else FACE_DETECTION_MODEL_NAME_H8L,
        FACE_RECOGNITION_PIPELINE: FACE_RECOGNITION_MODEL_NAME_H8
        if is_h8
        else FACE_RECOGNITION_MODEL_NAME_H8L
    }
    name = pipeline_map[pipeline_name]
    hailo_logger.debug(f"Resolved model name: {name}")
    return name


def get_resource_path(
    pipeline_name: str, resource_type: str, arch: str | None = None, model: str | None = None
) -> Path | None:
    hailo_logger.debug(
        f"Getting resource path for pipeline={pipeline_name}, resource_type={resource_type}, model={model}"
    )
    root = Path(RESOURCES_ROOT_PATH_DEFAULT)
    # Auto-detect arch if not provided and needed for RESOURCES_MODELS_DIR_NAME
    if arch is None and resource_type == RESOURCES_MODELS_DIR_NAME:
        arch = os.getenv(HAILO_ARCH_KEY) or detect_hailo_arch()
        hailo_logger.debug(f"Auto-detected arch: {arch}")
    if not arch and resource_type == RESOURCES_MODELS_DIR_NAME:
        hailo_logger.error("Could not detect Hailo architecture.")
        assert False, "Could not detect Hailo architecture."


    if resource_type == RESOURCES_SO_DIR_NAME and model:
        return root / RESOURCES_SO_DIR_NAME / model
    if resource_type == RESOURCES_VIDEOS_DIR_NAME and model:
        return root / RESOURCES_VIDEOS_DIR_NAME / model
    if resource_type == RESOURCES_PHOTOS_DIR_NAME and model:
        return root / RESOURCES_PHOTOS_DIR_NAME / model
    if resource_type == RESOURCES_JSON_DIR_NAME and model:
        return root / RESOURCES_JSON_DIR_NAME / model
    if resource_type == DEFAULT_LOCAL_RESOURCES_PATH and model:
        return root / DEFAULT_LOCAL_RESOURCES_PATH / model

    if resource_type == RESOURCES_MODELS_DIR_NAME:
        if model:
            model_path = root / RESOURCES_MODELS_DIR_NAME / arch / model
            if "." in model:
                return model_path.with_name(model_path.name + HAILO_FILE_EXTENSION)
            return model_path.with_suffix(HAILO_FILE_EXTENSION)
        if pipeline_name:
            name = get_model_name(pipeline_name, arch)
            name_path = root / RESOURCES_MODELS_DIR_NAME / arch / name
            if "." in name:
                return name_path.with_name(name_path.name + HAILO_FILE_EXTENSION)
            return name_path.with_suffix(HAILO_FILE_EXTENSION)
    return None


class FIFODropQueue(queue.Queue):
    def put(self, item, block=False, timeout=None):
        if self.full():
            hailo_logger.debug("Queue full, dropping oldest item.")
            self.get_nowait()
        super().put(item, block, timeout)


# =============================================================================
# Model Resolution and Listing
# =============================================================================

def list_models_for_app(app_name: str, arch: str | None = None) -> None:
    """
    List all available models for an application and exit.

    Args:
        app_name: The app name from resources config (e.g., 'detection', 'vlm_chat')
        arch: Hailo architecture. If None, auto-detects.
    """
    try:
        from hailo_apps.config.config_manager import (
            get_model_names,
            get_supported_architectures,
            is_gen_ai_app,
        )
    except ImportError:
        print("Error: Could not import config_manager. Run 'pip install -e .' first.")
        sys.exit(1)

    # Detect architecture if not provided
    if arch is None:
        arch = os.getenv(HAILO_ARCH_KEY) or detect_hailo_arch()
        if not arch:
            print(
                "\n❌ ERROR: Could not detect Hailo device architecture.\n"
                "   Please ensure:\n"
                "   - A Hailo device is connected\n"
                "   - The HailoRT driver is installed and loaded\n"
                "   - You have permissions to access the device\n"
                "\n   Alternatively, specify the architecture manually with --arch (e.g., --arch hailo8)\n",
                file=sys.stderr
            )
            sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"Available models for: {app_name} ({arch})")
    print(f"{'=' * 60}")

    # Check if architecture is supported
    supported_archs = get_supported_architectures(app_name)
    if arch not in supported_archs:
        if is_gen_ai_app(app_name):
            print(f"\n⚠️  This is a Gen-AI app, only available on: {', '.join(supported_archs)}")
        else:
            print(f"\n⚠️  Architecture '{arch}' not supported. Available: {', '.join(supported_archs)}")
        print()
        sys.exit(0)

    # Get models
    default_models = get_model_names(app_name, arch, tier="default")
    extra_models = get_model_names(app_name, arch, tier="extra")

    if default_models:
        print("\n📦 Default Models:")
        for model in default_models:
            print(f"   • {model}")
    else:
        print("\n📦 Default Models: None")

    if extra_models:
        print("\n📚 Extra Models:")
        for model in extra_models:
            print(f"   • {model}")

    print(f"\n{'=' * 60}")
    print(f"Total: {len(default_models)} default, {len(extra_models)} extra")
    print("\nUsage: --hef-path <model_name>")
    print("       Model will be auto-downloaded if not found locally.")
    print()
    sys.exit(0)


def resolve_hef_path(
    hef_path: str | None,
    app_name: str,
    arch: str
) -> Path | None:
    """
    Main method for resolving HEF (Hailo Executable Format) file paths.

    Provides intelligent path resolution with automatic model downloading.
    See README.md for detailed documentation and usage examples.

    Args:
        hef_path: User-provided path or model name (None uses default model)
        app_name: Application name from resources config (e.g., DETECTION_PIPELINE)
        arch: Hailo architecture ('hailo8', 'hailo8l', or 'hailo10h')

    Returns:
        Path to the HEF file, or None if not found
    """
    try:
        from hailo_apps.config.config_manager import (
            get_model_names,
            get_default_model_name,
        )
    except ImportError:
        hailo_logger.warning("Could not import config_manager, using legacy resolution")
        # Fallback to legacy resolution
        if hef_path is None:
            return get_resource_path(app_name, RESOURCES_MODELS_DIR_NAME, arch)
        return Path(hef_path) if hef_path else None

    resources_root = Path(RESOURCES_ROOT_PATH_DEFAULT)
    models_dir = resources_root / RESOURCES_MODELS_DIR_NAME / arch

    # Get available models for this app/arch
    available_models = get_model_names(app_name, arch, tier="all")
    default_model = get_default_model_name(app_name, arch)
    is_using_default = False

    # Case 1: No hef_path provided - use default model
    if hef_path is None:
        if default_model:
            hef_path = default_model
            is_using_default = True
            hailo_logger.info(f"Using default model: {default_model}")
        else:
            # Fallback to legacy pipeline-based model
            legacy_path = get_resource_path(app_name, RESOURCES_MODELS_DIR_NAME, arch)
            if legacy_path and legacy_path.exists():
                return legacy_path
            hailo_logger.error(f"No default model found for {app_name}/{arch}")
            return None
    
    # Normalize model name (extract basename and remove only the .hef suffix, keep dots in name)
    candidate_name = Path(hef_path).name
    if candidate_name.endswith(HAILO_FILE_EXTENSION):
        model_name = candidate_name[: -len(HAILO_FILE_EXTENSION)]
    else:
        model_name = candidate_name
    
    # Case 2: Check if it's a full path that exists
    hef_full_path = Path(hef_path)
    if hef_full_path.is_absolute() and hef_full_path.exists():
        hailo_logger.info(f"Using HEF from absolute path: {hef_full_path}")
        return hef_full_path

    # Also check with .hef extension
    if not hef_path.endswith(HAILO_FILE_EXTENSION):
        hef_full_path = Path(hef_path + HAILO_FILE_EXTENSION)
        if hef_full_path.exists():
            hailo_logger.info(f"Using HEF from path: {hef_full_path}")
            return hef_full_path

    # Case 3: Check in resources folder
    resource_path = models_dir / f"{model_name}{HAILO_FILE_EXTENSION}"
    if resource_path.exists():
        hailo_logger.info(f"Found HEF in resources: {resource_path}")
        return resource_path

    # Case 4: Model not found locally - check if it's in the available models list
    if model_name in available_models:
        # Show warning before downloading
        if is_using_default:
            print(f"\n⚠️  WARNING: Default model '{model_name}' is not downloaded.")
            print(f"   Downloading model for {app_name}/{arch}...")
            print(f"   This may take a while depending on your internet connection.\n")
        else:
            print(f"\n⚠️  WARNING: Model '{model_name}' is not downloaded.")
            print(f"   Downloading model for {app_name}/{arch}...")
            print(f"   This may take a while depending on your internet connection.\n")

        if _download_model(model_name, arch):
            if resource_path.exists():
                hailo_logger.info(f"Model downloaded successfully: {resource_path}")
                return resource_path
            else:
                hailo_logger.error(f"Download succeeded but file not found: {resource_path}")
                return None
        else:
            hailo_logger.error(f"Failed to download model: {model_name}")
            return None

    # Model not in available list - don't auto-download unknown models
    hailo_logger.error(
        f"Model '{model_name}' not found and not in available models list. "
        f"Available models for {app_name}/{arch}: {', '.join(available_models) if available_models else 'None'}"
    )
    return None


def _download_model(model_name: str, arch: str) -> bool:
    """
    Download a specific model using the download_resources module.

    Args:
        model_name: Name of the model to download
        arch: Hailo architecture

    Returns:
        True if download succeeded, False otherwise
    """
    try:
        from hailo_apps.installation.download_resources import download_resources

        print(f"Downloading model: {model_name} for {arch}...")
        download_resources(
            arch=arch,
            model=model_name,
            dry_run=False,
            force=False,
            parallel=False  # Sequential for single model
        )
        return True
    except Exception as e:
        hailo_logger.error(f"Failed to download model: {e}")
        return False


def handle_list_models_flag(args, app_name: str) -> None:
    """
    Handle the --list-models flag if present.

    Args:
        args: Parsed arguments (or parser to parse)
        app_name: App name from resources config
    """
    # Parse args if it's a parser
    if hasattr(args, 'parse_known_args'):
        options, _ = args.parse_known_args()
    else:
        options = args

    # Check if --list-models flag is set
    if getattr(options, 'list_models', False):
        arch = getattr(options, 'arch', None)
        list_models_for_app(app_name, arch)

def app_requires_multiple_models(app_name: str, arch: str) -> bool:
    models = get_default_models(app_name, arch)
    return len(models) > 1



@dataclass
class ResolvedModel:
    name: str
    path: Path


def resolve_hef_paths(
    hef_paths: list[str] | None,
    app_name: str,
    arch: str,
) -> list[ResolvedModel]:
    """
    Resolve one or more HEF paths for apps that require multiple models.

    Rules:
    - If hef_paths is None:
        → use ALL default models for the app
    - If hef_paths is provided:
        → length must match required model count
    - Each model is resolved via resolve_hef_path()
    """

    default_models = get_default_models(app_name, arch)
    required_count = len(default_models)

    # Normalize inputs
    if hef_paths in (None, [], ""):
        model_names = [m.name for m in default_models]
    elif isinstance(hef_paths, str):
        model_names = [hef_paths]
    else:
        model_names = list(hef_paths)

    # Validate count
    if len(model_names) != required_count:
        raise ValueError(
            f"{app_name} requires {required_count} models "
            f"but {len(model_names)} were provided"
        )

    resolved: list[ResolvedModel] = []

    for model_name in model_names:
        path = resolve_hef_path(
            hef_path=model_name,
            app_name=app_name,
            arch=arch,
        )
        if path is None:
            raise RuntimeError(f"Failed to resolve model: {model_name}")

        resolved.append(ResolvedModel(name=model_name, path=path))

    return resolved
