# region imports
# Standard library imports
import setproctitle
import json
import os

# Third-party imports
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

# Local application-specific imports
import hailo
from hailo_apps.python.core.common.core import get_pipeline_parser, get_resource_path, handle_list_models_flag, resolve_hef_path
from hailo_apps.python.core.common.defines import TAPPAS_STREAM_ID_TOOL_SO_FILENAME, MULTI_SOURCE_APP_TITLE, SIMPLE_DETECTION_PIPELINE, DETECTION_PIPELINE, RESOURCES_SO_DIR_NAME, DETECTION_POSTPROCESS_SO_FILENAME, DETECTION_POSTPROCESS_FUNCTION, TAPPAS_POSTPROC_PATH_KEY
from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import get_source_type, USER_CALLBACK_PIPELINE, TRACKER_PIPELINE, QUEUE, SOURCE_PIPELINE, INFERENCE_PIPELINE, DISPLAY_PIPELINE
from hailo_apps.python.core.gstreamer.gstreamer_app import GStreamerApp, app_callback_class, dummy_callback
from hailo_apps.python.core.common.hailo_logger import get_logger

hailo_logger = get_logger(__name__)
# endregion imports

# User Gstreamer Application: This class inherits from the common.GStreamerApp class
class GStreamerMultisourceApp(GStreamerApp):
    def __init__(self, app_callback, user_data, parser=None):

        if parser is None:
            parser = get_pipeline_parser()
        parser.add_argument("--sources", default='', help="The list of sources to use for the multisource pipeline, separated with comma e.g., /dev/video0,/dev/video1")
        
        # Handle --list-models flag - uses detection models
        handle_list_models_flag(parser, DETECTION_PIPELINE)
        
        super().__init__(parser, user_data)
        setproctitle.setproctitle(MULTI_SOURCE_APP_TITLE)

        # Resolve HEF path with smart lookup and auto-download (uses detection models)
        self.hef_path = resolve_hef_path(
            self.hef_path,
            app_name=DETECTION_PIPELINE,
            arch=self.arch
        )
        self.post_process_so = get_resource_path(SIMPLE_DETECTION_PIPELINE, RESOURCES_SO_DIR_NAME, self.arch, DETECTION_POSTPROCESS_SO_FILENAME)
        self.post_function_name = DETECTION_POSTPROCESS_FUNCTION
        self.video_sources_types = [(video_source, get_source_type(video_source)) for video_source in (self.options_menu.sources.split(',') if self.options_menu.sources else [self.video_source, self.video_source])]  # Default to 2 sources if none specified
        self.num_sources = len(self.video_sources_types)

        self.app_callback = app_callback
        self.create_pipeline()

    def get_pipeline_string(self):
        sources_string = ''
        router_string = ''

        tappas_post_process_dir = os.environ.get(TAPPAS_POSTPROC_PATH_KEY, '')
        set_stream_id_so = os.path.join(tappas_post_process_dir, TAPPAS_STREAM_ID_TOOL_SO_FILENAME)
        for id in range(self.num_sources):
            sources_string += SOURCE_PIPELINE(video_source=self.video_sources_types[id][0],
                                              frame_rate=self.frame_rate, sync=self.sync, name=f"source_{id}", no_webcam_compression=False)
            sources_string += f"! hailofilter name=set_src_{id} so-path={set_stream_id_so} config-path=src_{id} "
            sources_string += f"! {QUEUE(name=f'src_q_{id}', max_size_buffers=30)} ! robin.sink_{id} "
            router_string += f"router.src_{id} ! {USER_CALLBACK_PIPELINE(name=f'src_{id}_callback')} ! {QUEUE(name=f'callback_q_{id}', max_size_buffers=30)} ! {DISPLAY_PIPELINE(video_sink=self.video_sink, sync=self.sync, show_fps=self.show_fps, name=f'hailo_display_{id}')} "

        self.thresholds_str = (
            f"nms-score-threshold=0.3 "
            f"nms-iou-threshold=0.45 "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )

        # Create the detection pipeline
        detection_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.post_function_name,
            batch_size=self.batch_size,
            additional_params=self.thresholds_str)

        inference_string = f"hailoroundrobin mode=1 name=robin ! {detection_pipeline} ! {TRACKER_PIPELINE(class_id=-1)} ! {USER_CALLBACK_PIPELINE()} ! {QUEUE(name='call_q', max_size_buffers=30)} ! hailostreamrouter name=router "
        for id in range(self.num_sources):
            inference_string += f"src_{id}::input-streams=\"<sink_{id}>\" "

        pipeline_string = sources_string + inference_string + router_string
        print(pipeline_string)
        return pipeline_string

def main():
    # Create an instance of the user app callback class
    user_data = app_callback_class()
    app_callback = dummy_callback
    app = GStreamerMultisourceApp(app_callback, user_data)
    app.run()

if __name__ == "__main__":
    print("Starting Hailo Multisource App...")
    main()