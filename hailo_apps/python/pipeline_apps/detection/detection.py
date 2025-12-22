# region imports
# Standard library imports

# Third-party imports
import gi

gi.require_version("Gst", "1.0")
import cv2

# Local application-specific imports
import hailo
from gi.repository import Gst

from hailo_apps.python.pipeline_apps.detection.detection_pipeline import GStreamerDetectionApp
from hailo_apps.python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer,
)

from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class

hailo_logger = get_logger(__name__)
# endregion imports


# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.new_variable = 42

    def new_function(self):
        return "The meaning of life is: "


# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------


def app_callback(element, buffer, user_data):
    if buffer is None:
        hailo_logger.warning("Received None buffer.")
        return

    # Note: Frame counting is handled automatically by the framework wrapper
    frame_idx = user_data.get_count()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    pad = element.get_static_pad("src")
    format, width, height = get_caps_from_pad(pad)

    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    detection_count = 0
    for detection in detections:
        label = detection.get_label()
        confidence = detection.get_confidence()
        if label == "person":
            # Get track ID
            track_id = 0
            track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if len(track) == 1:
                track_id = track[0].get_id()
            string_to_print += (
                f"Detection: ID: {track_id} Label: {label} Confidence: {confidence:.2f}\n"
            )
            detection_count += 1
    if user_data.use_frame:
        cv2.putText(
            frame,
            f"Detections: {detection_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"{user_data.new_function()} {user_data.new_variable}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    print(string_to_print)
    return


def main():
    hailo_logger.info("Starting Detection App.")
    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()


if __name__ == "__main__":
    main()
