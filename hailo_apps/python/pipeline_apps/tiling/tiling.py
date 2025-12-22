# region imports
# Standard library imports

# Third-party imports
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

# Local application-specific imports
import hailo
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.python.pipeline_apps.tiling.tiling_pipeline import GStreamerTilingApp

hailo_logger = get_logger(__name__)
# endregion imports


# User-defined class to be used in the callback function: Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()


# User-defined callback function: This is the callback function that will be called when data is available from the pipeline
def app_callback(element, buffer, user_data):
    # Note: Frame counting is handled automatically by the framework wrapper
    string_to_print = f"Frame count: {user_data.get_count()}\n"
    if buffer is None:
        hailo_logger.warning("Received None buffer at frame=%s", user_data.get_count())
        return
    for detection in hailo.get_roi_from_buffer(buffer).get_objects_typed(hailo.HAILO_DETECTION):
        string_to_print += (f"Detection: {detection.get_label()} Confidence: {detection.get_confidence():.2f}\n")
    print(string_to_print)
    return


def main():
    """Main function for CLI entry point."""
    hailo_logger.info("Starting Tiling App.")
    user_data = user_app_callback_class()
    app = GStreamerTilingApp(app_callback, user_data)
    app.run()


if __name__ == "__main__":
    main()
