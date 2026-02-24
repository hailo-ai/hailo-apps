# region imports
import os
os.environ["GST_PLUGIN_FEATURE_RANK"] = "vaapidecodebin:NONE"
# Third-party imports
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

# Local application-specific imports
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.python.pipeline_apps.clip.clip_pipeline import GStreamerClipApp
# endregion

def app_callback(pad, info, user_data):
    return Gst.PadProbeReturn.OK

def main():
    user_data = app_callback_class()
    app = GStreamerClipApp(app_callback, user_data)
    app.run()
    
if __name__ == "__main__":
    main()