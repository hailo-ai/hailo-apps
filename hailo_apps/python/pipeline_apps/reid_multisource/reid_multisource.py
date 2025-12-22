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
from hailo_apps.python.pipeline_apps.reid_multisource.reid_multisource_pipeline import GStreamerREIDMultisourceApp

hailo_logger = get_logger(__name__)
# endregion imports


# User-defined class to be used in the callback function: Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()


# User-defined callback function: This is the callback function that will be called when data is available from the pipeline
def app_callback(element, buffer, user_data):
    if buffer is None:
        hailo_logger.warning("Received None buffer.")
        return
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    for detection in detections:
        ids = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if ids:
            track_id = ids[0].get_id()
            print(f'Unified callback, {roi.get_stream_id()}_{detection.get_label()}_{track_id}')
    return


def main():
    hailo_logger.info("Starting REID Multisource App.")
    user_data = user_app_callback_class()
    app = GStreamerREIDMultisourceApp(app_callback, user_data)
    app.run()


if __name__ == "__main__":
    main()
