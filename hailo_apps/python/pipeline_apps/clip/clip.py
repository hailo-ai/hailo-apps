# region imports
# Standard library imports
import sys

# Third-party imports
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

# Local application-specific imports
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.python.pipeline_apps.clip.clip_pipeline import GStreamerClipApp

hailo_logger = get_logger(__name__)
# endregion imports

def check_clip_dependencies():
    """
    Check if all required CLIP dependencies are installed.
    
    Exits the program with installation instructions if any dependencies are missing.
    """
    missing_deps = []
    for dep_name in ["clip", "torch", "torchvision"]:
        try:
            __import__(dep_name)
        except ImportError:
            missing_deps.append(dep_name)

    if missing_deps:
        print("\n" + "="*70)
        print("❌ MISSING REQUIRED DEPENDENCIES")
        print("="*70)
        print("\nThe following dependencies are required but not installed:")
        for dep in missing_deps:
            print(f"  • {dep}")
        print("\n" + "-"*70)
        print("INSTALLATION INSTRUCTIONS:")
        print("-"*70)
        print("\nTo install all dependencies (recommended):")
        print("  1. Navigate to the repository root directory")
        print("  2. Run: pip install -e \".[clip]\"")
        print("\n" + "="*70)
        sys.exit(1)

def app_callback(element, buffer, user_data):
    return

def main():
    check_clip_dependencies()
    hailo_logger.info("Starting CLIP App.")
    user_data = app_callback_class()
    app = GStreamerClipApp(app_callback, user_data)
    app.run()

if __name__ == "__main__":
    main()