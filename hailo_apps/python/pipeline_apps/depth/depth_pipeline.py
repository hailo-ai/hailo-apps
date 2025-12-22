# region imports
# Standard library imports
from pathlib import Path

# Third-party imports
import gi
import setproctitle

gi.require_version("Gst", "1.0")

# Local application-specific imports
from hailo_apps.python.core.common.core import (
    get_pipeline_parser,
    get_resource_path,
    handle_list_models_flag,
    resolve_hef_path,
)
from hailo_apps.python.core.common.defines import (
    DEPTH_APP_TITLE,
    DEPTH_PIPELINE,
    DEPTH_POSTPROCESS_FUNCTION,
    DEPTH_POSTPROCESS_SO_FILENAME,
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
    USER_CALLBACK_PIPELINE,
)

hailo_logger = get_logger(__name__)  # same run_id everywhere

# endregion imports


# User Gstreamer Application: This class inherits from the common.GStreamerApp class
class GStreamerDepthApp(GStreamerApp):
    def __init__(self, app_callback, user_data, parser=None):
        if parser is None:
            parser = get_pipeline_parser()
            # Note: logging args are already added by get_pipeline_parser() via get_base_parser()
        
        # Handle --list-models flag before full initialization
        handle_list_models_flag(parser, DEPTH_PIPELINE)

        hailo_logger.info("Initializing GStreamer Depth App...")

        super().__init__(parser, user_data)

        hailo_logger.debug(
            "Parent GStreamerApp initialized, options parsed: arch=%s, input=%s, fps=%s, sync=%s, show_fps=%s",
            self.arch,
            self.video_source,
            self.frame_rate,
            self.sync,
            self.show_fps,
        )

        # Architecture is already handled by GStreamerApp parent class
        # Use self.arch which is set by parent

        self.app_callback = app_callback
        setproctitle.setproctitle(DEPTH_APP_TITLE)
        hailo_logger.debug("Process title set to %s", DEPTH_APP_TITLE)

        # Resolve HEF path with smart lookup and auto-download
        self.hef_path = resolve_hef_path(
            self.hef_path,
            app_name=DEPTH_PIPELINE,
            arch=self.arch
        )
        self.post_process_so = get_resource_path(
            DEPTH_PIPELINE, RESOURCES_SO_DIR_NAME, self.arch, DEPTH_POSTPROCESS_SO_FILENAME
        )
        self.post_function_name = DEPTH_POSTPROCESS_FUNCTION
        hailo_logger.debug(
            "HEF path: %s, Post-process SO: %s, Post-process function: %s",
            self.hef_path,
            self.post_process_so,
            self.post_function_name,
        )

        hailo_logger.info(
            "Resources resolved | hef=%s | post_so=%s | post_fn=%s",
            self.hef_path,
            self.post_process_so,
            self.post_function_name,
        )

        # Validate resource paths
        if self.hef_path is None or not Path(self.hef_path).exists():
            hailo_logger.error("HEF path is invalid or missing: %s", self.hef_path)
        if self.post_process_so is None or not Path(self.post_process_so).exists():
            hailo_logger.error(
                "Post-process .so path is invalid or missing: %s", self.post_process_so
            )

        self.create_pipeline()
        hailo_logger.debug("Pipeline created")

    def get_pipeline_string(self):
        source_pipeline = SOURCE_PIPELINE(
            video_source=self.video_source,
            video_width=self.video_width,
            video_height=self.video_height,
            frame_rate=self.frame_rate,
            sync=self.sync,
        )
        depth_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.post_function_name,
            name="depth_inference",
        )
        depth_pipeline_wrapper = INFERENCE_PIPELINE_WRAPPER(
            depth_pipeline, name="inference_wrapper_depth"
        )
        user_callback_pipeline = USER_CALLBACK_PIPELINE()
        display_pipeline = DISPLAY_PIPELINE(
            video_sink=self.video_sink, sync=self.sync, show_fps=self.show_fps
        )

        pipeline_str = (
            f"{source_pipeline} ! "
            f"{depth_pipeline_wrapper} ! "
            f"{user_callback_pipeline} ! "
            f"{display_pipeline}"
        )
        hailo_logger.debug("Pipeline string created: %s", pipeline_str)
        return pipeline_str


def main():
    # Create an instance of the user app callback class
    user_data = app_callback_class()
    app_callback = dummy_callback
    app = GStreamerDepthApp(app_callback, user_data)
    app.run()


if __name__ == "__main__":
    print("Starting Hailo Depth App...")
    main()
