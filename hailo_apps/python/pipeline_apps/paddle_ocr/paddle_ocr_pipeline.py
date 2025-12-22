# region imports
# Standard library imports
from pathlib import Path

import setproctitle

from hailo_apps.python.core.common.core import (
    get_pipeline_parser,
    get_resource_path,
    handle_list_models_flag,
    configure_multi_model_hef_path,
    resolve_hef_paths,
)
from hailo_apps.python.core.common.defines import (
    OCR_APP_TITLE,
    PADDLE_OCR_PIPELINE,
    OCR_POSTPROCESS_SO_FILENAME,
    OCR_DETECTION_POSTPROCESS_FUNCTION,
    OCR_RECOGNITION_POSTPROCESS_FUNCTION,
    OCR_CROPPER_FUNCTION,
    OCR_VIDEO_NAME,
    RESOURCES_SO_DIR_NAME,
    RESOURCES_VIDEOS_DIR_NAME,
    BASIC_PIPELINES_VIDEO_EXAMPLE_NAME,
    REPO_ROOT,
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
    CROPPER_PIPELINE,
    TRACKER_PIPELINE,
    QUEUE,
)

hailo_logger = get_logger(__name__)

# endregion imports

# -----------------------------------------------------------------------------------------------
# User Gstreamer Application
# -----------------------------------------------------------------------------------------------


class GStreamerPaddleOCRApp(GStreamerApp):
    def __init__(self, app_callback, user_data, parser=None):
        if parser is None:
            parser = get_pipeline_parser()

        # Configure --hef-path for multi-model support (OCR detection + OCR recognition)
        configure_multi_model_hef_path(parser)

        # Handle --list-models flag before full initialization
        handle_list_models_flag(parser, PADDLE_OCR_PIPELINE)

        hailo_logger.info("Initializing GStreamer OCR App...")

        super().__init__(parser, user_data)

        hailo_logger.debug(
            "Parent GStreamerApp initialized | arch=%s | input=%s | fps=%s | sync=%s | show_fps=%s",
            self.arch,
            self.video_source,
            self.frame_rate,
            self.sync,
            self.show_fps,
        )

        # Set Hailo parameters - use different batch sizes for detection vs recognition
        # Recognition: batch_size=8 to accumulate multiple cropped regions before inference
        # This improves hardware utilization and reduces inference overhead

        self.recognition_batch_size = 8
        hailo_logger.debug("OCR pipeline: Using batch_size=%d for recognition (will batch up to %d cropped regions)",
                         self.recognition_batch_size, self.recognition_batch_size)

        # Set frame rate to 10FPS for better performance and reduced processing load
        if self.frame_rate > 10:
            self.frame_rate = 10
            hailo_logger.info("OCR pipeline: Frame rate set to %d FPS", self.frame_rate)

        # Resolve HEF paths for multi-model app (OCR detection + OCR recognition)
        # Uses --hef-path arguments if provided, otherwise uses defaults
        models = resolve_hef_paths(
            hef_paths=self.options_menu.hef_path,  # List from action='append' or None
            app_name=PADDLE_OCR_PIPELINE,
            arch=self.arch,
        )
        self.ocr_det_hef_path = models[0].path
        self.ocr_rec_hef_path = models[1].path

        # Post-processing shared object file (contains both detection and recognition functions)
        self.post_process_so = get_resource_path(
            pipeline_name=None,
            resource_type=RESOURCES_SO_DIR_NAME,
            arch=self.arch,
            model=OCR_POSTPROCESS_SO_FILENAME
        )

        # Post-processing function names
        self.ocr_det_post_function = OCR_DETECTION_POSTPROCESS_FUNCTION
        self.ocr_rec_post_function = OCR_RECOGNITION_POSTPROCESS_FUNCTION
        self.cropper_function = OCR_CROPPER_FUNCTION

        # Video source - use get_resource_path if default video, otherwise use user input
        if BASIC_PIPELINES_VIDEO_EXAMPLE_NAME in self.video_source:
            video_path = get_resource_path(
                pipeline_name=None,
                resource_type=RESOURCES_VIDEOS_DIR_NAME,
                arch=self.arch,
                model=OCR_VIDEO_NAME
            )
            self.video_source = str(video_path) if video_path else None

        hailo_logger.info(
            "Resources | ocr_det_hef=%s | ocr_rec_hef=%s | post_so=%s | det_fn=%s | rec_fn=%s | cropper_fn=%s",
            self.ocr_det_hef_path,
            self.ocr_rec_hef_path,
            self.post_process_so,
            self.ocr_det_post_function,
            self.ocr_rec_post_function,
            self.cropper_function,
        )

        # OCR config file - located in local_resources alongside frequency dictionary
        ocr_config_name = "ocr_config.json"
        ocr_config_path = Path(REPO_ROOT) / "local_resources" / ocr_config_name
        self.ocr_config_path = str(ocr_config_path) if ocr_config_path.exists() else None

        if not ocr_config_path.exists():
            hailo_logger.warning("OCR config file not found at: %s", ocr_config_path)

        # Validate resource paths
        if self.ocr_det_hef_path is None or not Path(self.ocr_det_hef_path).exists():
            hailo_logger.error("OCR Detection HEF path is invalid or missing: %s", self.ocr_det_hef_path)
        if self.ocr_rec_hef_path is None or not Path(self.ocr_rec_hef_path).exists():
            hailo_logger.error("OCR Recognition HEF path is invalid or missing: %s", self.ocr_rec_hef_path)
        if self.post_process_so is None or not Path(self.post_process_so).exists():
            hailo_logger.error("Post-process .so path is invalid or missing: %s", self.post_process_so)

        self.app_callback = app_callback

        # Set the process title
        setproctitle.setproctitle(OCR_APP_TITLE)
        hailo_logger.debug("Process title set to %s", OCR_APP_TITLE)

        # Create the pipeline
        self.create_pipeline()
        hailo_logger.debug("Pipeline created")


    def get_pipeline_string(self):
        """Returns the OCR pipeline with detection and recognition."""
        # Full pipeline with detection and recognition
        # 1. Source pipeline
        source_pipeline = SOURCE_PIPELINE(
            video_source=self.video_source,
            video_width=self.video_width,
            video_height=self.video_height,
            frame_rate=self.frame_rate,
            sync=self.sync,
            mirror_image=False,
        )

        # 2. OCR Detection pipeline - detects text regions (bounding boxes)
        ocr_det_pipeline = INFERENCE_PIPELINE(
            hef_path=str(self.ocr_det_hef_path) if self.ocr_det_hef_path else None,
            post_process_so=str(self.post_process_so) if self.post_process_so else None,
            post_function_name=self.ocr_det_post_function,
            batch_size=self.batch_size,
            name="ocr_detection",
        )

        # Wrap detection to preserve original frame size
        ocr_det_wrapper = INFERENCE_PIPELINE_WRAPPER(ocr_det_pipeline)

        # 2.5. Tracker pipeline - tracks text regions across frames
        # Reduced keep_lost_frames and keep_tracked_frames to remove stale tracks faster
        # This prevents boxes from persisting after text disappears
        tracker_pipeline = TRACKER_PIPELINE(
            class_id=-1,
            name="ocr_tracker",
            keep_lost_frames=1,  # Remove lost tracks after 1 frame (faster cleanup)
            keep_tracked_frames=2,  # Consider tracked instances lost after 2 frames without match
        )

        # 3. OCR Recognition pipeline - recognizes text in cropped regions
        # Use batch_size=8 to accumulate multiple cropped regions before inference
        # hailonet will automatically batch incoming crops up to 8, improving throughput
        # This is more efficient than processing crops one by one
        ocr_rec_pipeline = INFERENCE_PIPELINE(
            hef_path=str(self.ocr_rec_hef_path) if self.ocr_rec_hef_path else None,
            post_process_so=str(self.post_process_so) if self.post_process_so else None,
            post_function_name=self.ocr_rec_post_function,
            batch_size=self.recognition_batch_size,  # Use larger batch size for recognition
            config_json=self.ocr_config_path,
            name="ocr_recognition",
        )

        # 4. Cropper pipeline - crops detected text regions and feeds them to OCR recognition
        ocr_cropper = CROPPER_PIPELINE(
            inner_pipeline=ocr_rec_pipeline,
            so_path=str(self.post_process_so) if self.post_process_so else None,
            function_name=self.cropper_function,
            internal_offset=True,
            bypass_max_size_buffers=16,
            name="ocr_cropper",
        )

        # 5. User callback pipeline
        user_callback_pipeline = USER_CALLBACK_PIPELINE()

        # 6. Display pipeline
        display_pipeline = DISPLAY_PIPELINE(
            video_sink=self.video_sink, sync=self.sync, show_fps=self.show_fps
        )

        # Full pipeline: Source -> OCR Detection -> Tracker -> Cropper (with OCR Recognition) -> Callback -> Display
        pipeline_string = (
            f"{source_pipeline} ! "
            f"{ocr_det_wrapper} ! "
            f"{tracker_pipeline} ! "
            f"{ocr_cropper} ! "
            f"{user_callback_pipeline} ! "
            f"{display_pipeline}"
        )

        return pipeline_string


def main():
    # Create an instance of the user app callback class
    hailo_logger.info("Starting Hailo OCR App...")
    user_data = app_callback_class()
    app_callback = dummy_callback
    app = GStreamerPaddleOCRApp(app_callback, user_data)
    app.run()


if __name__ == "__main__":
    main()

