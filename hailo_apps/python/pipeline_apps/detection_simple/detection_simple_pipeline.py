# region imports
# Standard library imports
import setproctitle

from hailo_apps.python.core.common.core import get_pipeline_parser, get_resource_path, handle_list_models_flag, resolve_hef_path
from hailo_apps.python.core.common.defines import (
    RESOURCES_SO_DIR_NAME,
    RESOURCES_VIDEOS_DIR_NAME,
    SIMPLE_DETECTION_APP_TITLE,
    SIMPLE_DETECTION_PIPELINE,
    SIMPLE_DETECTION_POSTPROCESS_FUNCTION,
    SIMPLE_DETECTION_POSTPROCESS_SO_FILENAME,
    SIMPLE_DETECTION_VIDEO_NAME,
)

from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.core.gstreamer.gstreamer_app import (
    GStreamerApp,
    app_callback_class,
    dummy_callback,
)
from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import (
    DISPLAY_PIPELINE,
    INFERENCE_PIPELINE,
    SOURCE_PIPELINE,
    USER_CALLBACK_PIPELINE,
)

hailo_logger = get_logger(__name__)

# endregion imports

# -----------------------------------------------------------------------------------------------
# User Gstreamer Application
# -----------------------------------------------------------------------------------------------


# This class inherits from the hailo_rpi_common.GStreamerApp class
class GStreamerDetectionSimpleApp(GStreamerApp):
    def __init__(self, app_callback, user_data, parser=None):
        if parser is None:
            parser = get_pipeline_parser()
        parser.add_argument(
            "--labels-json",
            default=None,
            help="Path to costume labels JSON file",
        )
        
        # Handle --list-models flag before full initialization
        handle_list_models_flag(parser, SIMPLE_DETECTION_PIPELINE)
        
        hailo_logger.info("Initializing GStreamer Detection Simple App...")
        super().__init__(parser, user_data)
        # Override width/height if not set via parser
        if self.video_width == 1280:
            self.video_width = 640
        if self.video_height == 720:
            self.video_height = 640

        # Set Hailo parameters - these parameters should be set based on the model used
        # Override batch_size if not set via parser
        if self.batch_size == 1:
            self.batch_size = 2
        nms_score_threshold = 0.3
        nms_iou_threshold = 0.45
        if (
            self.options_menu.input is None
        ):  # Setting up a new application-specific default video (overrides the default video set in the GStreamerApp constructor)
            self.video_source = get_resource_path(
                pipeline_name=SIMPLE_DETECTION_PIPELINE,
                resource_type=RESOURCES_VIDEOS_DIR_NAME,
                arch=self.arch,
                model=SIMPLE_DETECTION_VIDEO_NAME,
            )
        # Architecture is already handled by GStreamerApp parent class
        # Use self.arch which is set by parent

        # Resolve HEF path with smart lookup and auto-download
        self.hef_path = resolve_hef_path(
            self.hef_path,
            app_name=SIMPLE_DETECTION_PIPELINE,
            arch=self.arch
        )
        hailo_logger.info(f"Using HEF path: {self.hef_path}")

        self.post_process_so = get_resource_path(
            pipeline_name=SIMPLE_DETECTION_PIPELINE,
            resource_type=RESOURCES_SO_DIR_NAME,
            arch=self.arch,
            model=SIMPLE_DETECTION_POSTPROCESS_SO_FILENAME,
        )
        hailo_logger.info(f"Using post-process shared object: {self.post_process_so}")

        self.post_function_name = SIMPLE_DETECTION_POSTPROCESS_FUNCTION

        # User-defined label JSON file
        self.labels_json = self.options_menu.labels_json

        self.app_callback = app_callback

        self.thresholds_str = (
            f"nms-score-threshold={nms_score_threshold} "
            f"nms-iou-threshold={nms_iou_threshold} "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )

        hailo_logger.info(f"Using thresholds: {self.thresholds_str}")

        # Set the process title
        setproctitle.setproctitle(SIMPLE_DETECTION_APP_TITLE)

        self.create_pipeline()

    def get_pipeline_string(self):
        source_pipeline = SOURCE_PIPELINE(
            video_source=self.video_source,
            video_width=self.video_width,
            video_height=self.video_height,
            frame_rate=self.frame_rate,
            sync=self.sync,
            no_webcam_compression=True,
        )

        detection_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.post_function_name,
            batch_size=self.batch_size,
            config_json=self.labels_json,
            additional_params=self.thresholds_str,
        )
        user_callback_pipeline = USER_CALLBACK_PIPELINE()
        display_pipeline = DISPLAY_PIPELINE(
            video_sink=self.video_sink, sync=self.sync, show_fps=self.show_fps
        )

        pipeline_string = (
            f"{source_pipeline} ! "
            f"{detection_pipeline} ! "
            f"{user_callback_pipeline} ! "
            f"{display_pipeline}"
        )
        hailo_logger.info(f"Pipeline string: {pipeline_string}")
        return pipeline_string


def main():
    # Create an instance of the user app callback class
    hailo_logger.info("Creating user data for the app callback...")
    user_data = app_callback_class()
    app_callback = dummy_callback
    app = GStreamerDetectionSimpleApp(app_callback, user_data)
    app.run()


if __name__ == "__main__":
    hailo_logger.info("Starting the GStreamer Detection Simple App...")
    main()

