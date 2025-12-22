# region imports
# Standard library imports

# Third-party imports
import gi

gi.require_version("Gst", "1.0")
import cv2

# Local application-specific imports
import hailo
from gi.repository import Gst

from hailo_apps.python.pipeline_apps.paddle_ocr.paddle_ocr_pipeline import GStreamerPaddleOCRApp
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
        self.ocr_results = []

    def get_ocr_results(self):
        return self.ocr_results

    def add_ocr_result(self, text, confidence, bbox):
        self.ocr_results.append({
            'text': text,
            'confidence': confidence,
            'bbox': bbox
        })

    def clear_ocr_results(self):
        self.ocr_results.clear()


# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------


def app_callback(element, buffer, user_data):
    # Note: Frame counting is handled automatically by the framework wrapper
    if buffer is None:
        hailo_logger.warning("Received None buffer.")
        return

    # Get video format and dimensions from pad caps
    pad = element.get_static_pad("src")
    format, width, height = get_caps_from_pad(pad)

    # Get current frame index
    frame_idx = user_data.get_count()
    hailo_logger.debug("Frame=%s | caps fmt=%s %sx%s", frame_idx, format, width, height)
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    # Clear previous OCR results
    user_data.clear_ocr_results()

    # Get the OCR results from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    text_detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Parse the OCR detections
    text_count = 0
    for detection in text_detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()

        # Check for text regions (OCR detection uses "text_region" label)
        # Use confidence threshold 0.12 to match C++ postprocess code (MIN_CONFIDENCE)
        # Note: Classifications should be added in postprocess
        if label == "text_region" and confidence > 0.12:
            # Get OCR text result if available (from recognition stage)
            text_result = ""
            ocr_objects = detection.get_objects_typed(hailo.HAILO_CLASSIFICATION)
            if len(ocr_objects) > 0:
                # Try to find classification with type "text_region", otherwise use first one
                for cls in ocr_objects:
                    if cls.get_classification_type() == "text_region":
                        text_result = cls.get_label()
                        break
                if not text_result and len(ocr_objects) > 0:
                    text_result = ocr_objects[0].get_label()

            # Filter out empty text results - only store and display non-empty text
            # Empty text usually indicates recognition failed or detection was false positive
            if text_result and text_result.strip():
                # Store OCR result
                user_data.add_ocr_result(text_result, confidence, bbox)

                string_to_print += (
                    f"OCR Detection: Text: '{text_result}' Confidence: {confidence:.2f}\n"
                )
                text_count += 1

    # Extract frame from buffer right before drawing to ensure we have the latest frame
    # Note: This drawing is for the separate window when use_frame=True
    # The main GStreamer display uses hailooverlay which draws from detection metadata automatically
    if user_data.use_frame and format is not None and width is not None and height is not None:
        # Get video frame from the current buffer (extract fresh for each frame)
        frame = get_numpy_from_buffer(buffer, format, width, height)

        if frame is not None:
            # Draw OCR results on frame
            ocr_results = user_data.get_ocr_results()

            for ocr_result in ocr_results:
                bbox = ocr_result['bbox']
                text = ocr_result['text']
                confidence = ocr_result['confidence']

                # Draw bounding box (use video dimensions from caps)
                x1 = int(bbox.xmin() * width)
                y1 = int(bbox.ymin() * height)
                x2 = int((bbox.xmin() + bbox.width()) * width)
                y2 = int((bbox.ymin() + bbox.height()) * height)

                # Ensure coordinates are within frame bounds
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))

                # Draw bounding box (BGR color: green)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw text result (ensure text position is within frame)
                text_label = f"{text} ({confidence:.2f})"
                text_y = max(15, y1 - 10)
                cv2.putText(
                    frame,
                    text_label,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

            # Display total text count
            cv2.putText(
                frame,
                f"Text Regions: {text_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Convert the frame to BGR for OpenCV display (frame is RGB from buffer)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            user_data.set_frame(frame)
        else:
            hailo_logger.warning("Failed to extract frame from buffer.")

    print(string_to_print)
    return


def main():
    hailo_logger.info("Starting OCR App.")
    user_data = user_app_callback_class()
    app = GStreamerPaddleOCRApp(app_callback, user_data)
    app.run()


if __name__ == "__main__":
    main()
