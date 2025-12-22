# region imports
# Standard library imports
from pathlib import Path

import setproctitle

from hailo_apps.python.core.common.core import (
    get_pipeline_parser,
    get_resource_path,
    handle_list_models_flag,
    resolve_hef_path,
)
from hailo_apps.python.core.common.defines import (
    INSTANCE_SEGMENTATION_APP_TITLE,
    INSTANCE_SEGMENTATION_MODEL_NAME_H8,
    INSTANCE_SEGMENTATION_MODEL_NAME_H8L,
    INSTANCE_SEGMENTATION_PIPELINE,
    INSTANCE_SEGMENTATION_POSTPROCESS_FUNCTION,
    INSTANCE_SEGMENTATION_POSTPROCESS_SO_FILENAME,
    JSON_FILE_EXTENSION,
    RESOURCES_JSON_DIR_NAME,
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
# User GStreamer Application: Instance Segmentation
# -----------------------------------------------------------------------------------------------


class GStreamerInstanceSegmentationApp(GStreamerApp):
    def __init__(self, app_callback, user_data, parser=None):
        if parser is None:
            parser = get_pipeline_parser()
        
        # Handle --list-models flag before full initialization
        handle_list_models_flag(parser, INSTANCE_SEGMENTATION_PIPELINE)

        hailo_logger.info("Initializing GStreamer Instance Segmentation App...")
        super().__init__(parser, user_data)
        hailo_logger.debug(
            "Base app init complete | arch=%s | video_source=%s | fps=%s | sync=%s | show_fps=%s",
            self.arch,
            self.video_source,
            self.frame_rate,
            self.sync,
            self.show_fps,
        )

        # Hailo parameters - override defaults for instance segmentation
        if self.batch_size == 1:
            self.batch_size = 2
        # Override width/height if not set via parser
        if self.video_width == 1280:
            self.video_width = 640
        if self.video_height == 720:
            self.video_height = 640
        hailo_logger.debug(
            "Set batch_size=%d video_width=%d video_height=%d",
            self.batch_size,
            self.video_width,
            self.video_height,
        )

        # Architecture is already handled by GStreamerApp parent class
        # Use self.arch which is set by parent

        # Resolve HEF path with smart lookup and auto-download
        resolved_path = resolve_hef_path(
            self.hef_path,
            app_name=INSTANCE_SEGMENTATION_PIPELINE,
            arch=self.arch
        )
        self.hef_path = str(resolved_path) if resolved_path else None
        hailo_logger.info("HEF path: %s", self.hef_path)

        # Determine which JSON config to use
        hef_name = Path(self.hef_path).name
        if INSTANCE_SEGMENTATION_MODEL_NAME_H8 in hef_name:
            self.config_file = get_resource_path(
                INSTANCE_SEGMENTATION_PIPELINE,
                RESOURCES_JSON_DIR_NAME,
                self.arch,
                INSTANCE_SEGMENTATION_MODEL_NAME_H8 + JSON_FILE_EXTENSION,
            )
            hailo_logger.info("Using config file for H8: %s", self.config_file)
        elif INSTANCE_SEGMENTATION_MODEL_NAME_H8L in hef_name:
            self.config_file = get_resource_path(
                INSTANCE_SEGMENTATION_PIPELINE,
                RESOURCES_JSON_DIR_NAME,
                self.arch,
                INSTANCE_SEGMENTATION_MODEL_NAME_H8L + JSON_FILE_EXTENSION,
            )
            hailo_logger.info("Using config file for H8L: %s", self.config_file)
        else:
            hailo_logger.error("Unsupported HEF version: %s", hef_name)
            raise ValueError(
                "HEF version not supported; please provide a compatible segmentation HEF or config file."
            )

        # Post-process shared object
        self.post_process_so = get_resource_path(
            INSTANCE_SEGMENTATION_PIPELINE,
            RESOURCES_SO_DIR_NAME,
            self.arch,
            INSTANCE_SEGMENTATION_POSTPROCESS_SO_FILENAME,
        )
        self.post_function_name = INSTANCE_SEGMENTATION_POSTPROCESS_FUNCTION
        hailo_logger.debug(
            "Postprocess SO: %s | Function: %s", self.post_process_so, self.post_function_name
        )

        # Callback
        self.app_callback = app_callback

        # Set process title
        setproctitle.setproctitle(INSTANCE_SEGMENTATION_APP_TITLE)
        hailo_logger.debug("Process title set to %s", INSTANCE_SEGMENTATION_APP_TITLE)

        # Create pipeline
        self.create_pipeline()
        hailo_logger.debug("Pipeline created successfully")

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
            post_function_name=self.post_function_name,
            batch_size=self.batch_size,
            config_json=self.config_file,
        )
        infer_pipeline_wrapper = INFERENCE_PIPELINE_WRAPPER(infer_pipeline)
        tracker_pipeline = TRACKER_PIPELINE(class_id=1)
        user_callback_pipeline = USER_CALLBACK_PIPELINE()
        display_pipeline = DISPLAY_PIPELINE(
            video_sink=self.video_sink,
            sync=self.sync,
            show_fps=self.show_fps,
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
    hailo_logger.info("Starting Hailo Instance Segmentation App...")
    user_data = app_callback_class()
    app = GStreamerInstanceSegmentationApp(dummy_callback, user_data)
    app.run()


if __name__ == "__main__":
    hailo_logger.info("Executing __main__")
    main()
