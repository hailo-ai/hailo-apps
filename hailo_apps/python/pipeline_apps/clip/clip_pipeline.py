# region imports
# Standard library imports
import os
import signal
import setproctitle

# Third-party imports
import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

# Local application-specific imports
import hailo
from hailo_apps.python.core.common.defines import (
    CLIP_CROPPER_FACE_POSTPROCESS_FUNCTION_NAME,
    CLIP_CROPPER_LICENSE_PLATE_POSTPROCESS_FUNCTION_NAME,
    CLIP_CROPPER_OBJECT_POSTPROCESS_FUNCTION_NAME,
    CLIP_CROPPER_PERSON_POSTPROCESS_FUNCTION_NAME,
    CLIP_CROPPER_VEHICLE_POSTPROCESS_FUNCTION_NAME,
    CLIP_DETECTION_POSTPROCESS_FUNCTION_NAME,
    CLIP_DETECTOR_TYPE_FACE,
    CLIP_DETECTOR_TYPE_LICENSE_PLATE,
    CLIP_DETECTOR_TYPE_PERSON,
    CLIP_DETECTOR_TYPE_VEHICLE,
    CLIP_POSTPROCESS_FUNCTION_NAME,
    RESOURCES_SO_DIR_NAME,
    RESOURCES_VIDEOS_DIR_NAME,
    BASIC_PIPELINES_VIDEO_EXAMPLE_NAME,
    CLIP_APP_TITLE,
    CLIP_VIDEO_NAME,
    CLIP_PIPELINE,
    DETECTION_POSTPROCESS_SO_FILENAME,
    CLIP_POSTPROCESS_SO_FILENAME,
    CLIP_CROPPER_POSTPROCESS_SO_FILENAME,
)
from hailo_apps.python.core.common.core import (
    get_pipeline_parser,
    get_resource_path,
    handle_list_models_flag,
    configure_multi_model_hef_path,
    resolve_hef_paths,
)
from hailo_apps.python.core.common.hef_utils import get_hef_labels_json
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.core.gstreamer.gstreamer_app import GStreamerApp, app_callback_class, dummy_callback

hailo_logger = get_logger(__name__)
from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import (
    QUEUE,
    SOURCE_PIPELINE,
    INFERENCE_PIPELINE,
    INFERENCE_PIPELINE_WRAPPER,
    TRACKER_PIPELINE,
    USER_CALLBACK_PIPELINE,
    DISPLAY_PIPELINE,
    CROPPER_PIPELINE
)
from hailo_apps.python.pipeline_apps.clip.text_image_matcher import text_image_matcher
from hailo_apps.python.pipeline_apps.clip import gui
# endregion imports

class GStreamerClipApp(GStreamerApp):
    def __init__(self, app_callback, user_data, parser=None):
        if parser is None:
            parser = get_pipeline_parser()
        parser.add_argument("--detector", "-d", type=str, choices=["person", "vehicle", "face", "license-plate", "none"], default="none", help="Which detection pipeline to use.")
        parser.add_argument("--json-path", type=str, default=None, help="Path to JSON file to load and save embeddings. If not set, embeddings.json will be used.")
        parser.add_argument("--detection-threshold", type=float, default=0.5, help="Detection threshold.")
        parser.add_argument("--disable-runtime-prompts", action="store_true", help="When set, app will not support runtime prompts. Default is False.")
        parser.add_argument("--labels-json", type=str, default=None, help="Path to custom labels JSON file for detection model.")

        # Configure --hef-path for multi-model support (detection + clip)
        configure_multi_model_hef_path(parser)

        # Handle --list-models flag before full initialization
        handle_list_models_flag(parser, CLIP_PIPELINE)

        super().__init__(parser, user_data)
        setproctitle.setproctitle(CLIP_APP_TITLE)
        if self.options_menu.input is None:
            self.json_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'example_embeddings.json') if self.options_menu.json_path is None else self.options_menu.json_path
        else:
            self.json_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'embeddings.json') if self.options_menu.json_path is None else self.options_menu.json_path
        self.app_callback = app_callback
        self.detector = self.options_menu.detector
        self.text_image_matcher = text_image_matcher
        self.text_image_matcher.set_threshold(self.options_menu.detection_threshold)
        self.win = gui.AppWindow(self.options_menu.detection_threshold, self.options_menu.disable_runtime_prompts, self.text_image_matcher, self.json_file)
        self.detection_batch_size = 8
        self.clip_batch_size = 8

        if BASIC_PIPELINES_VIDEO_EXAMPLE_NAME in self.video_source:
            self.video_source = get_resource_path(pipeline_name=None, resource_type=RESOURCES_VIDEOS_DIR_NAME, model=CLIP_VIDEO_NAME)

        # Resolve HEF paths for multi-model app (detection + clip)
        # Uses --hef-path arguments if provided, otherwise uses defaults
        models = resolve_hef_paths(
            hef_paths=self.options_menu.hef_path,  # List from action='append' or None
            app_name=CLIP_PIPELINE,
            arch=self.arch,
        )

        # order as in hailo_apps/config/resources_config.yaml
        self.hef_path_clip = models[0].path
        self.hef_path_detection = models[1].path

        # User-defined label JSON file for detection model
        self.labels_json = self.options_menu.labels_json
        if self.labels_json is None and self.options_menu.detector != 'none':
            # Auto-detect labels JSON from detection HEF file if not provided
            self.labels_json = get_hef_labels_json(self.hef_path_detection)
            if self.labels_json is not None:
                hailo_logger.info("Auto detected Labels JSON: %s", self.labels_json)

        self.post_process_so_detection = get_resource_path(pipeline_name=None, resource_type=RESOURCES_SO_DIR_NAME, model=DETECTION_POSTPROCESS_SO_FILENAME)
        self.post_process_so_clip = get_resource_path(pipeline_name=None, resource_type=RESOURCES_SO_DIR_NAME, model=CLIP_POSTPROCESS_SO_FILENAME)
        self.post_process_so_cropper = get_resource_path(pipeline_name=None, resource_type=RESOURCES_SO_DIR_NAME, model=CLIP_CROPPER_POSTPROCESS_SO_FILENAME)

        self.clip_post_process_function_name = CLIP_POSTPROCESS_FUNCTION_NAME
        self.detection_post_process_function_name = CLIP_DETECTION_POSTPROCESS_FUNCTION_NAME
        if self.options_menu.detector == CLIP_DETECTOR_TYPE_PERSON:
            self.class_id = 1
            self.cropper_post_process_function_name = CLIP_CROPPER_PERSON_POSTPROCESS_FUNCTION_NAME
        elif self.options_menu.detector == CLIP_DETECTOR_TYPE_VEHICLE:
            self.class_id = 2
            self.cropper_post_process_function_name = CLIP_CROPPER_VEHICLE_POSTPROCESS_FUNCTION_NAME
        elif self.options_menu.detector == CLIP_DETECTOR_TYPE_FACE:
            self.class_id = 3
            self.cropper_post_process_function_name = CLIP_CROPPER_FACE_POSTPROCESS_FUNCTION_NAME
        elif self.options_menu.detector == CLIP_DETECTOR_TYPE_LICENSE_PLATE :
            self.class_id = 4
            self.cropper_post_process_function_name = CLIP_CROPPER_LICENSE_PLATE_POSTPROCESS_FUNCTION_NAME
        else:
            self.class_id = 0
            self.cropper_post_process_function_name = CLIP_CROPPER_OBJECT_POSTPROCESS_FUNCTION_NAME

        self.classified_tracks = set()  # Track which track_ids have already been classified

        self.matching_callback_name = 'matching_identity_callback'

        self.create_pipeline()

        identity = self.pipeline.get_by_name(self.matching_callback_name)
        if identity:
            identity.set_property("signal-handoffs", True)
            identity.connect("handoff", self.matching_identity_callback, self.user_data)

    def run(self):
        self.win.connect('delete-event', self.on_window_close)
        self.win.show_all()
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        super().run()

    def on_window_close(self, window, event):
        self.loop.quit()
        return False

    def get_pipeline_string(self):
        source_pipeline = SOURCE_PIPELINE(self.video_source, self.video_width, self.video_height, frame_rate=self.frame_rate, sync=self.sync)

        detection_pipeline = INFERENCE_PIPELINE(
                hef_path=self.hef_path_detection,
                post_process_so=self.post_process_so_detection,
                post_function_name=self.detection_post_process_function_name,
                batch_size=self.detection_batch_size,
                config_json=self.labels_json,
                scheduler_priority=31,
                scheduler_timeout_ms=100,
                name='detection_inference'
        )

        detection_pipeline_wrapper = ''
        if self.options_menu.detector != 'none':
            detection_pipeline_wrapper = INFERENCE_PIPELINE_WRAPPER(detection_pipeline)

        clip_pipeline = INFERENCE_PIPELINE(
                hef_path=self.hef_path_clip,
                post_process_so=self.post_process_so_clip,
                post_function_name=self.clip_post_process_function_name,
                batch_size=self.clip_batch_size,
                scheduler_priority=16,
                scheduler_timeout_ms=1000,
                name='clip_inference'
        )

        tracker_pipeline = TRACKER_PIPELINE(class_id=self.class_id, keep_past_metadata=True)

        clip_cropper_pipeline = CROPPER_PIPELINE(
            inner_pipeline=clip_pipeline,
            so_path=self.post_process_so_cropper,
            function_name=self.cropper_post_process_function_name,
            name='clip_cropper'
        )

        # Clip pipeline with muxer integration - add explicit resize caps
        clip_pipeline_wrapper = f'tee name=clip_t hailomuxer name=clip_hmux \
            clip_t. ! {QUEUE(name="clip_bypass_q", max_size_buffers=20)} ! clip_hmux.sink_0 \
            clip_t. ! {QUEUE(name="clip_muxer_queue")} ! videoscale qos=false ! {clip_pipeline} ! clip_hmux.sink_1 \
            clip_hmux. ! {QUEUE(name="clip_hmux_queue")} '

        display_pipeline = DISPLAY_PIPELINE(video_sink=self.video_sink, sync=self.sync, show_fps='True')

        matching_callback_pipeline = USER_CALLBACK_PIPELINE(name=self.matching_callback_name)

        user_callback_pipeline = USER_CALLBACK_PIPELINE()

        if self.detector == 'none':
            return (
                f'{source_pipeline} ! '
                f'{clip_pipeline_wrapper} ! '
                f'{matching_callback_pipeline} ! '
                f'{user_callback_pipeline} ! '
                f'{display_pipeline}'
            )
        else:
            return (
                f'{source_pipeline} ! '
                f'{detection_pipeline_wrapper} ! '
                f'{tracker_pipeline} ! '
                f'{clip_cropper_pipeline} ! '
                f'{matching_callback_pipeline} ! '
                f'{user_callback_pipeline} ! '
                f'{display_pipeline}'
            )

    def matching_identity_callback(self, element, buffer, user_data):
        if buffer is None:
            return
        roi = hailo.get_roi_from_buffer(buffer)
        if roi is None:
            return
        top_level_matrix = roi.get_objects_typed(hailo.HAILO_MATRIX)
        if len(top_level_matrix) == 0:
            detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
        else:
            detections = [roi]
        embeddings_np = None
        used_detection = []
        track_id_focus = text_image_matcher.track_id_focus  # Used to focus on a specific track_id
        update_tracked_probability = None
        for detection in detections:
            results = detection.get_objects_typed(hailo.HAILO_MATRIX)
            if len(results) == 0:
                continue
            detection_embeddings = np.array(results[0].get_data())
            used_detection.append(detection)  # used_detection corresponds to embeddings_np
            if embeddings_np is None:
                embeddings_np = detection_embeddings[np.newaxis, :]
            else:
                embeddings_np = np.vstack((embeddings_np, detection_embeddings))  # Stack vertically ("append")
            if track_id_focus is not None:
                track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
                if len(track) == 1:
                    track_id = track[0].get_id()
                    # If we have a track_id_focus, update only the tracked_probability of the focused track
                    if track_id == track_id_focus:
                        update_tracked_probability = len(used_detection) - 1  # The focused detection was just appended, so its index is the last one
        if embeddings_np is not None:
            matches = text_image_matcher.match(embeddings_np, report_all=True, update_tracked_probability=update_tracked_probability)
            for match in matches:  # (row_idx - in embeddings_np or used_detection, text, similarity (confidence), entry_index - TextImageMatcher.entries - which text prompt matched best) = match
                detection = used_detection[match.row_idx]

                # Get old classifications BEFORE adding new ones
                old_classification = detection.get_objects_typed(hailo.HAILO_CLASSIFICATION)

                if (match.passed_threshold and not match.negative):
                    # Add label as classification metadata
                    classification = hailo.HailoClassification('clip', match.text, match.similarity)
                    detection.add_object(classification)

                    # Remove old classifications only when new one is added
                    for old in old_classification:
                        detection.remove_object(old)
        return

if __name__ == "__main__":
    user_data = app_callback_class()
    app = GStreamerClipApp(user_data, dummy_callback)
    app.run()