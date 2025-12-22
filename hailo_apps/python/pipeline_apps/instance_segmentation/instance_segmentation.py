# region imports
# Standard library imports

# Third-party imports
import gi

gi.require_version("Gst", "1.0")
import cv2

# Local application-specific imports
import hailo
import numpy as np
from gi.repository import Gst

from hailo_apps.python.pipeline_apps.instance_segmentation.instance_segmentation_pipeline import (
    GStreamerInstanceSegmentationApp,
)
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
        self.frame_skip = 2  # Process every 2nd frame to reduce compute


# Predefined colors (BGR format)
COLORS = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 128),  # Purple
    (255, 165, 0),  # Orange
    (0, 128, 128),  # Teal
    (128, 128, 0),  # Olive
]


# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------
def app_callback(element, buffer, user_data):
    # Note: Frame counting is handled automatically by the framework wrapper
    hailo_logger.debug("Callback triggered. Current frame count=%d", user_data.get_count())

    if buffer is None:
        hailo_logger.warning("Received None buffer in callback.")
        return

    hailo_logger.debug("Processing frame %d", user_data.get_count())
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    if user_data.get_count() % user_data.frame_skip != 0:
        return

    pad = element.get_static_pad("src")
    format, width, height = get_caps_from_pad(pad)

    reduced_width = width // 4
    reduced_height = height // 4

    reduced_frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)
        reduced_frame = cv2.resize(
            frame, (reduced_width, reduced_height), interpolation=cv2.INTER_AREA
        )

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()

        if label == "person":
            track_id = 0
            track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if len(track) == 1:
                track_id = track[0].get_id()

            string_to_print += (
                f"Detection: ID: {track_id} Label: {label} Confidence: {confidence:.2f}\n"
            )

            if user_data.use_frame:
                masks = detection.get_objects_typed(hailo.HAILO_CONF_CLASS_MASK)
                if len(masks) != 0:
                    mask = masks[0]
                    mask_height = mask.get_height()
                    mask_width = mask.get_width()

                    data = np.array(mask.get_data())
                    data = data.reshape((mask_height, mask_width))

                    roi_width = int(bbox.width() * reduced_width)
                    roi_height = int(bbox.height() * reduced_height)
                    resized_mask_data = cv2.resize(
                        data, (roi_width, roi_height), interpolation=cv2.INTER_LINEAR
                    )

                    x_min, y_min = (
                        int(bbox.xmin() * reduced_width),
                        int(bbox.ymin() * reduced_height),
                    )
                    x_max, y_max = x_min + roi_width, y_min + roi_height

                    y_min = max(y_min, 0)
                    x_min = max(x_min, 0)
                    y_max = min(y_max, reduced_frame.shape[0])
                    x_max = min(x_max, reduced_frame.shape[1])

                    if x_max > x_min and y_max > y_min:
                        mask_overlay = np.zeros_like(reduced_frame)
                        color = COLORS[track_id % len(COLORS)]
                        mask_overlay[y_min:y_max, x_min:x_max] = (
                            resized_mask_data[: y_max - y_min, : x_max - x_min, np.newaxis] > 0.5
                        ) * color
                        reduced_frame = cv2.addWeighted(reduced_frame, 1, mask_overlay, 0.5, 0)

    print(string_to_print)

    if user_data.use_frame:
        reduced_frame = cv2.cvtColor(reduced_frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(reduced_frame)

    return


def main():
    hailo_logger.info("Starting Instance Segmentation App.")
    user_data = user_app_callback_class()
    app = GStreamerInstanceSegmentationApp(app_callback, user_data)
    app.run()


if __name__ == "__main__":
    main()
