"""
Pose Estimation Overlay Demo
=============================

Demonstrates hailooverlay_community features using the pose estimation pipeline:

- **Custom colors per person** based on pose analysis:
  - Arms raised (both wrists above shoulders) → green
  - Crouching (hip close to ankle) → yellow
  - Default standing pose → cyan
- **Text background** for readability
- **Stats overlay** showing FPS and object count
- **Min-confidence filtering** to hide weak detections

Usage:
    python hailo_apps/postprocess/cpp/overlay_community/examples/pose_estimation_overlay_demo.py

    # With a YAML style config (optional):
    python hailo_apps/postprocess/cpp/overlay_community/examples/pose_estimation_overlay_demo.py \
        --style-config hailo_apps/postprocess/cpp/overlay_community/examples/overlay_style.yaml
"""

# region imports
import os
import argparse

os.environ["GST_PLUGIN_FEATURE_RANK"] = "vaapidecodebin:NONE"

import hailo

from hailo_apps.python.pipeline_apps.pose_estimation.pose_estimation_pipeline import (
    GStreamerPoseEstimationApp,
)
from hailo_apps.python.core.common.buffer_utils import (
    get_caps_from_pad,
)
from hailo_apps.python.core.common.core import get_pipeline_parser
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class

hailo_logger = get_logger(__name__)
# endregion imports


# COCO keypoint indices
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# Colors as packed 0xRRGGBB
COLOR_ARMS_RAISED = 0x00FF00   # green
COLOR_CROUCHING = 0xFFFF00     # yellow
COLOR_DEFAULT = 0x00C8FF       # cyan


class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()


def classify_pose(points, bbox):
    """Analyze pose keypoints and return a packed 0xRRGGBB color.

    Args:
        points: List of HailoPoint objects (17 COCO keypoints).
        bbox: HailoBBox of the detection.

    Returns:
        int: Packed color, or -1 if insufficient keypoints.
    """
    if len(points) < 17:
        return COLOR_DEFAULT

    min_conf = 0.3

    # Check arms raised: both wrists above both shoulders
    l_shoulder = points[LEFT_SHOULDER]
    r_shoulder = points[RIGHT_SHOULDER]
    l_wrist = points[LEFT_WRIST]
    r_wrist = points[RIGHT_WRIST]

    if (l_wrist.confidence() > min_conf and r_wrist.confidence() > min_conf
            and l_shoulder.confidence() > min_conf and r_shoulder.confidence() > min_conf):
        shoulder_y = max(l_shoulder.y(), r_shoulder.y())
        if l_wrist.y() < shoulder_y and r_wrist.y() < shoulder_y:
            return COLOR_ARMS_RAISED

    # Check crouching: average hip-to-ankle distance is small
    l_hip = points[LEFT_HIP]
    r_hip = points[RIGHT_HIP]
    l_ankle = points[LEFT_ANKLE]
    r_ankle = points[RIGHT_ANKLE]

    if (l_hip.confidence() > min_conf and l_ankle.confidence() > min_conf):
        hip_ankle_dist = abs(l_hip.y() - l_ankle.y())
        if hip_ankle_dist < 0.15:  # relative to bbox height
            return COLOR_CROUCHING

    if (r_hip.confidence() > min_conf and r_ankle.confidence() > min_conf):
        hip_ankle_dist = abs(r_hip.y() - r_ankle.y())
        if hip_ankle_dist < 0.15:
            return COLOR_CROUCHING

    return COLOR_DEFAULT


def app_callback(element, buffer, user_data):
    """Callback that attaches overlay_color metadata based on pose analysis."""
    if buffer is None:
        return

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    for detection in detections:
        if detection.get_label() != "person":
            continue

        landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
        if not landmarks:
            continue

        points = landmarks[0].get_points()
        bbox = detection.get_bbox()
        color_packed = classify_pose(points, bbox)

        # Attach overlay_color classification with packed 0xRRGGBB in class_id
        color_cls = hailo.HailoClassification(
            "overlay_color",    # type
            color_packed,       # class_id (index) = packed 0xRRGGBB
            "",                 # label (unused when class_id > 0)
            0.0,                # confidence (unused)
        )
        detection.add_object(color_cls)

    return


class GStreamerPoseOverlayDemo(GStreamerPoseEstimationApp):
    """Pose estimation app with community overlay features enabled."""

    def __init__(self, app_callback, user_data, style_config=None, sprite_config=None):
        parser = get_pipeline_parser()
        parser.add_argument(
            "--style-config",
            type=str,
            default=style_config or "",
            help="Path to YAML style config for per-class overlay overrides.",
        )
        parser.add_argument(
            "--sprite-config",
            type=str,
            default=sprite_config or "",
            help="Path to YAML sprite config mapping keys to PNG files.",
        )
        self._style_config = None
        self._sprite_config = None

        # Parse args early to capture --style-config / --sprite-config
        args, _ = parser.parse_known_args()
        self._style_config = args.style_config
        self._sprite_config = args.sprite_config

        super().__init__(app_callback, user_data, parser=parser)

    def get_pipeline_string(self):
        from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import (
            DISPLAY_PIPELINE,
            INFERENCE_PIPELINE,
            INFERENCE_PIPELINE_WRAPPER,
            SOURCE_PIPELINE,
            TRACKER_PIPELINE,
            USER_CALLBACK_PIPELINE,
        )

        source_pipeline = SOURCE_PIPELINE(
            video_source=self.video_source,
            video_width=self.video_width,
            video_height=self.video_height,
            frame_rate=self.frame_rate,
            sync=self.sync,
        )
        infer_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.post_process_function,
            batch_size=self.batch_size,
        )
        infer_pipeline_wrapper = INFERENCE_PIPELINE_WRAPPER(infer_pipeline)
        tracker_pipeline = TRACKER_PIPELINE(class_id=0)
        user_callback_pipeline = USER_CALLBACK_PIPELINE()

        # Build overlay props
        overlay_props = {
            "use_custom_colors": True,
            "text_background": True,
            "stats_overlay": True,
            "min_confidence": 0.5,
        }
        if self._style_config:
            overlay_props["style_config"] = self._style_config
        if self._sprite_config:
            overlay_props["sprite_config"] = self._sprite_config

        display_pipeline = DISPLAY_PIPELINE(
            video_sink=self.video_sink,
            sync=self.sync,
            show_fps=self.show_fps,
            community_overlay=True,
            overlay_props=overlay_props,
        )

        pipeline_string = (
            f"{source_pipeline} ! "
            f"{infer_pipeline_wrapper} ! "
            f"{tracker_pipeline} ! "
            f"{user_callback_pipeline} ! "
            f"{display_pipeline}"
        )
        hailo_logger.debug("Pipeline string: %s", pipeline_string)
        return pipeline_string


def main():
    hailo_logger.info("Starting Pose Estimation Overlay Demo.")
    user_data = user_app_callback_class()
    app = GStreamerPoseOverlayDemo(app_callback, user_data)
    app.run()


if __name__ == "__main__":
    main()
