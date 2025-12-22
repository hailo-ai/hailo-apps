# region imports
# Standard library imports
import setproctitle

from hailo_apps.python.core.common.core import (
    get_pipeline_parser,
    get_resource_path,
    handle_list_models_flag,
    resolve_hef_path,
)
from hailo_apps.python.core.common.defines import (
    POSE_ESTIMATION_APP_TITLE,
    POSE_ESTIMATION_PIPELINE,
    POSE_ESTIMATION_POSTPROCESS_FUNCTION,
    POSE_ESTIMATION_POSTPROCESS_SO_FILENAME,
    RESOURCES_SO_DIR_NAME,
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
    INFERENCE_PIPELINE_WRAPPER,
    SOURCE_PIPELINE,
    TRACKER_PIPELINE,
    USER_CALLBACK_PIPELINE,
)

hailo_logger = get_logger(__name__)
# endregion imports


# -----------------------------------------------------------------------------------------------
# User Gstreamer Application
# -----------------------------------------------------------------------------------------------
class GStreamerPoseEstimationApp(GStreamerApp):
    def __init__(self, app_callback, user_data, parser=None):
        if parser is None:
            parser = get_pipeline_parser()
        
        # Handle --list-models flag before full initialization
        handle_list_models_flag(parser, POSE_ESTIMATION_PIPELINE)
        
        hailo_logger.info("Initializing GStreamer Pose Estimation App...")

        super().__init__(parser, user_data)
        hailo_logger.debug("Parser initialized, user_data ready.")

        # Model parameters - override defaults if not set via parser
        if self.batch_size == 1:
            self.batch_size = 2
        # video_width and video_height are already set from parser or defaults
        hailo_logger.debug(
            "Video params set: %dx%d, batch_size=%d",
            self.video_width,
            self.video_height,
            self.batch_size,
        )

        # Architecture is already handled by GStreamerApp parent class
        # Use self.arch which is set by parent

        # Resolve HEF path with smart lookup and auto-download
        self.hef_path = resolve_hef_path(
            self.hef_path,
            app_name=POSE_ESTIMATION_PIPELINE,
            arch=self.arch
        )
        hailo_logger.debug("Using HEF path: %s", self.hef_path)

        self.app_callback = app_callback
        self.post_process_so = get_resource_path(
            POSE_ESTIMATION_PIPELINE, RESOURCES_SO_DIR_NAME, self.arch, POSE_ESTIMATION_POSTPROCESS_SO_FILENAME
        )
        self.post_process_function = POSE_ESTIMATION_POSTPROCESS_FUNCTION
        hailo_logger.debug(
            "Post-process SO: %s, Function: %s", self.post_process_so, self.post_process_function
        )

        setproctitle.setproctitle(POSE_ESTIMATION_APP_TITLE)
        hailo_logger.debug("Process title set: %s", POSE_ESTIMATION_APP_TITLE)

        self.create_pipeline()
        hailo_logger.info("Pipeline created successfully.")

    def get_pipeline_string(self):
        hailo_logger.debug("Building pipeline string...")
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
        display_pipeline = DISPLAY_PIPELINE(
            video_sink=self.video_sink, sync=self.sync, show_fps=self.show_fps
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
    hailo_logger.info("Starting Pose Estimation App main()...")
    user_data = app_callback_class()
    app = GStreamerPoseEstimationApp(dummy_callback, user_data)
    app.run()


if __name__ == "__main__":
    hailo_logger.info("Launching Pose Estimation App...")
    main()
