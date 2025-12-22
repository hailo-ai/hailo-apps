"""
Common utilities and constants for GStreamer applications.
"""
import sys
import cv2
import gi
from gi.repository import GLib, GObject, Gst
from hailo_apps.python.core.common.hailo_logger import get_logger

hailo_logger = get_logger(__name__)

# Watchdog constants
WATCHDOG_TIMEOUT = 5  # seconds
WATCHDOG_INTERVAL = 5  # seconds

# Suppress GStreamer buffer warnings by installing a custom log handler
# These warnings are cosmetic in GStreamer 1.26+ with complex pipelines
_suppressed_gstreamer_patterns = ["write map requested on non-writable buffer"]

def gstreamer_log_filter(log_domain, log_level, message, user_data):
    """
    Custom GLib log handler that filters out specific GStreamer warnings.

    This handler suppresses cosmetic warnings that appear in GStreamer 1.26+ when
    complex pipelines (like instance segmentation) use buffers with multiple references.
    These warnings don't indicate functional problems - GStreamer handles them internally.
    """
    # Suppress messages containing our specific patterns
    if message and not any(pattern in message for pattern in _suppressed_gstreamer_patterns):
        # For non-suppressed messages, use default behavior (print to stderr)
        if log_level & (GLib.LogLevelFlags.LEVEL_ERROR | GLib.LogLevelFlags.LEVEL_CRITICAL):
            sys.stderr.write(f"({log_domain}): CRITICAL: {message}\n")
            sys.stderr.flush()

def disable_qos(pipeline):
    """
    Disables QoS on all elements in the pipeline.
    """
    hailo_logger.debug("disable_qos() called")
    if not isinstance(pipeline, Gst.Pipeline):
        hailo_logger.error("Provided object is not a GStreamer Pipeline")
        print("The provided object is not a GStreamer Pipeline")
        return

    it = pipeline.iterate_elements()
    while True:
        result, element = it.next()
        if result != Gst.IteratorResult.OK:
            break

        if "qos" in GObject.list_properties(element):
            element.set_property("qos", False)
            hailo_logger.debug(f"Set qos=False for {element.get_name()}")
            # print(f"Set qos to False for {element.get_name()}")

def display_user_data_frame(user_data):
    """
    Displays frames from user_data in a window.
    """
    hailo_logger.debug("display_user_data_frame() started")
    while user_data.running:
        frame = user_data.get_frame()
        if frame is not None:
            hailo_logger.debug("Displaying user frame")
            cv2.imshow("User Frame", frame)
        cv2.waitKey(1)
    hailo_logger.debug("display_user_data_frame() exiting")
    cv2.destroyAllWindows()

