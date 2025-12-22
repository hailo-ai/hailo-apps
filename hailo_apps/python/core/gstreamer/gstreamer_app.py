import multiprocessing
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path
import traceback

# Fix for X11 threading issue
import ctypes
import ctypes.util

# Try to find and load X11 to initialize threads
_x11_lib = ctypes.util.find_library('X11')
if _x11_lib is None:
    _x11_lib = 'libX11.so.6'  # Fallback to standard SONAME

try:
    ctypes.CDLL(_x11_lib).XInitThreads()
except OSError:
    pass  # X11 likely not present (headless or non-Linux)

import cv2
import setproctitle

import gi
gi.require_version('Gtk', '3.0')
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

from hailo_apps.python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer,
)
from hailo_apps.python.core.common.camera_utils import (
    get_usb_video_devices,
)
from hailo_apps.python.core.common.core import (
    load_environment,
)
from hailo_apps.python.core.common.installation_utils import detect_hailo_arch

# Absolute imports for your common utilities
from hailo_apps.python.core.common.defines import (
    BASIC_PIPELINES_VIDEO_EXAMPLE_NAME,
    GST_VIDEO_SINK,
    HAILO_ARCH_KEY,
    HAILO_RGB_VIDEO_FORMAT,
    RESOURCES_ROOT_PATH_DEFAULT,
    RESOURCES_VIDEOS_DIR_NAME,
    RPI_NAME_I,
    TAPPAS_POSTPROC_PATH_KEY,
    USB_CAMERA,
)
from hailo_apps.python.core.common.hailo_logger import get_logger, init_logging, level_from_args

# python/core/gstreamer/gstreamer_app.py
# Absolute import for your local helper
from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import (
    get_source_type,
)
from hailo_apps.python.core.gstreamer.gstreamer_common import (
    gstreamer_log_filter,
    disable_qos,
    display_user_data_frame,
    WATCHDOG_TIMEOUT,
    WATCHDOG_INTERVAL,
)

# Install the custom log handler for all GStreamer logs
GLib.log_set_handler("GStreamer", GLib.LogLevelFlags.LEVEL_MASK, gstreamer_log_filter, None)

hailo_logger = get_logger(__name__)

try:
    from picamera2 import Picamera2
except ImportError:
    pass

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
class app_callback_class:
    """
    Base class for user callback data in GStreamer pipeline applications.

    This class provides frame counting and frame queue management. The frame counting
    is handled automatically by an internal framework wrapper - user callbacks do NOT
    need to call increment() manually.

    Key Features:
    - Automatic frame counting: increment() is called by the framework before each callback
    - Thread-safe frame counter access via get_count()
    - Optional frame queue for passing frames between callback and main thread
    - Watchdog monitoring support (when enabled via --enable-watchdog flag)
    - Debug mode: Callback timing statistics (average, max) when logger is at DEBUG level

    Note for Users:
        Do NOT call increment() in your callback functions. The framework automatically
        wraps your callback with _internal_callback_wrapper which handles frame counting.
        Simply use get_count() to read the current frame number.

    Attributes:
        frame_count (int): Current frame number (auto-incremented by framework)
        use_frame (bool): Whether to extract frame data in callback
        frame_queue (Queue): Queue for passing frames to display thread
        running (bool): Flag to control thread lifecycle
        callback_times (list): Debug mode - stores callback execution times
        callback_max_time (float): Debug mode - maximum callback time observed
    """
    def __init__(self):
        hailo_logger.debug("Initializing app_callback_class")
        self.frame_count = 0
        self.use_frame = False
        self.frame_queue = multiprocessing.Queue(maxsize=3)
        self.running = True
        # Debug mode timing statistics
        self.callback_times = []
        self.callback_max_time = 0.0

    def increment(self):
        """
        Increment frame count. Called AUTOMATICALLY by the framework on every frame.

        WARNING: Do NOT call this method in your user callback functions.
        The framework wrapper (_internal_callback_wrapper) calls this automatically.

        Thread-Safety: Lock-free by design. Python's GIL makes integer increment atomic
        enough for watchdog monitoring. Rare race conditions (~0.01% chance) are acceptable
        since we monitor trends, not exact counts. Avoids lock overhead in hot path (30-60+ FPS).
        """
        self.frame_count += 1

    def get_count(self):
        """
        Get current frame count for watchdog monitoring and user reference.
        Thread-safe to call from any thread (atomic read via GIL).

        Returns:
            int: Current frame number (starts at 1 after first frame)
        """
        return self.frame_count

    def set_frame(self, frame):
        if not self.frame_queue.full():
            hailo_logger.debug("Adding frame to queue")
            self.frame_queue.put(frame)
        else:
            hailo_logger.warning("Frame queue is full; dropping frame.")

    def get_frame(self):
        if not self.frame_queue.empty():
            hailo_logger.debug("Retrieving frame from queue")
            return self.frame_queue.get()
        else:
            hailo_logger.debug("Frame queue is empty")
            return None


def dummy_callback(element, buffer, user_data):
    """
    Dummy callback that does nothing. Used as default when no user callback is provided.
    Note: frame counting for watchdog happens automatically in the wrapper, not here.
    """
    return


def _internal_callback_wrapper(element, buffer, user_data, user_callback, disable_callback):
    """
    Internal wrapper that automatically increments frame count before calling user callback.
    This ensures watchdog monitoring works without requiring users to modify their callbacks.

    This wrapper ALWAYS runs to maintain frame counting for watchdog monitoring,
    even if --disable-callback is set. The user callback is skipped when disabled,
    but frame counting continues.

    Debug Mode:
        When logger is at DEBUG level, this wrapper automatically tracks callback performance:
        - Measures execution time for each callback invocation
        - Tracks average time (rolling 100-frame window)
        - Records maximum time observed
        - Prints statistics every 100 frames

        To enable debug mode, set the HAILO_LOG_LEVEL environment variable:
            export HAILO_LOG_LEVEL=debug
            hailo-detect --enable-watchdog

        Or run with inline environment variable:
            HAILO_LOG_LEVEL=debug hailo-detect --enable-watchdog

        Example output:
            DEBUG | gstreamer_app | Callback Performance [100 frames]:
                avg=2.34ms, max=8.12ms, current=2.56ms

    Args:
        element: GStreamer element
        buffer: GStreamer buffer
        user_data: app_callback_class instance
        user_callback: User's actual callback function
        disable_callback: If True, skip calling user callback (but still count frames)
    """
    # Automatically increment frame count for watchdog monitoring (always runs)
    user_data.increment()

    # Skip user callback if disabled, but continue frame counting
    if disable_callback:
        return

    # Debug mode: Track callback timing
    debug_mode = hailo_logger.isEnabledFor(10)  # 10 = DEBUG level
    if debug_mode:
        start_time = time.perf_counter()

    # Call user's callback
    result = None
    if user_callback:
        result = user_callback(element, buffer, user_data)

    # Debug mode: Calculate timing statistics
    if debug_mode:
        elapsed = (time.perf_counter() - start_time) * 1000
        user_data.callback_times.append(elapsed)

        # Update max time
        if elapsed > user_data.callback_max_time:
            user_data.callback_max_time = elapsed

        # Print statistics every 100 frames
        if len(user_data.callback_times) >= 100:
            avg_time = sum(user_data.callback_times) / len(user_data.callback_times)
            hailo_logger.debug(
                "Callback Performance [100 frames]: avg=%.2fms, max=%.2fms, current=%.2fms",
                avg_time, user_data.callback_max_time, elapsed
            )
            # Keep only recent times for rolling average (last 100)
            user_data.callback_times = user_data.callback_times[-100:]

    return result


# -----------------------------------------------------------------------------------------------
# GStreamerApp class
# -----------------------------------------------------------------------------------------------
class GStreamerApp:
    def __init__(self, args, user_data: app_callback_class):
        hailo_logger.debug("Initializing GStreamerApp")
        setproctitle.setproctitle("Hailo Python App")

        self.options_menu = args.parse_args()
        hailo_logger.debug(f"Parsed CLI options: {self.options_menu}")

        # Initialize logging from CLI args if provided
        if hasattr(self.options_menu, 'log_level') or hasattr(self.options_menu, 'debug'):
            init_logging(
                level=level_from_args(self.options_menu),
                log_file=getattr(self.options_menu, 'log_file', None),
                force=True  # Override auto-config
            )

        signal.signal(signal.SIGINT, self.shutdown)

        env_file = os.environ.get("HAILO_ENV_FILE")
        hailo_logger.debug(f"Loading environment from {env_file}")
        load_environment(env_file)

        # Determine the architecture if not specified
        if self.options_menu.arch is None:
            arch = os.getenv(HAILO_ARCH_KEY, detect_hailo_arch())
            if not arch:
                hailo_logger.error("Could not detect Hailo architecture.")
                print(
                    "ERROR: Could not auto-detect Hailo architecture. "
                    "Please specify --arch manually (e.g., --arch hailo8, --arch hailo8l, --arch hailo10h).",
                    file=sys.stderr
                )
                sys.exit(1)
            self.arch = arch
            hailo_logger.debug(f"Auto-detected Hailo architecture: {self.arch}")
        else:
            self.arch = self.options_menu.arch
            hailo_logger.debug("Using user-specified arch: %s", self.arch)

        tappas_post_process_dir = Path(os.environ.get(TAPPAS_POSTPROC_PATH_KEY, ""))
        if tappas_post_process_dir == "":
            hailo_logger.error("TAPPAS_POST_PROC_DIR environment variable not set. Please set it by sourcing set_env.sh")
            exit(1)

        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.postprocess_dir = tappas_post_process_dir

        if self.options_menu.input is None:
            self.video_source = str(
                Path(RESOURCES_ROOT_PATH_DEFAULT)
                / RESOURCES_VIDEOS_DIR_NAME
                / BASIC_PIPELINES_VIDEO_EXAMPLE_NAME
            )
        else:
            self.video_source = self.options_menu.input

        if self.video_source == USB_CAMERA:
            hailo_logger.debug("USB_CAMERA detected; scanning USB devices...")
            self.video_source = get_usb_video_devices()
            if not self.video_source:
                hailo_logger.error("No USB camera found for '--input usb'")
                exit(1)
            else:
                hailo_logger.debug(f"Using USB camera: {self.video_source[0]}")
                self.video_source = self.video_source[0]

        self.source_type = get_source_type(self.video_source)
        hailo_logger.debug(f"Source type determined: {self.source_type}")

        self.frame_rate = self.options_menu.frame_rate
        self.user_data = user_data
        self.video_sink = GST_VIDEO_SINK
        self.pipeline = None
        self.loop = None
        self.threads = []
        self.error_occurred = False
        self.pipeline_latency = 300

        # Handle batch-size from parser (default: 1)
        self.batch_size = getattr(self.options_menu, 'batch_size', 1)

        # Handle width/height from parser (defaults: 1280x720)
        # Note: parser sets default=None, so we need to check for None explicitly
        width = getattr(self.options_menu, 'width', None)
        height = getattr(self.options_menu, 'height', None)
        self.video_width = width if width is not None else 1280
        self.video_height = height if height is not None else 720

        self.video_format = HAILO_RGB_VIDEO_FORMAT

        # Handle hef-path from parser (default: None, apps can override)
        self.hef_path = getattr(self.options_menu, 'hef_path', None)

        self.app_callback = None

        user_data.use_frame = self.options_menu.use_frame

        self.sync = (
            "true" if (self.source_type == "file" and not self.options_menu.disable_sync) else "false"
        )
        self.show_fps = self.options_menu.show_fps

        if self.options_menu.dump_dot:
            hailo_logger.debug("Dump DOT enabled")
            os.environ["GST_DEBUG_DUMP_DOT_DIR"] = os.getcwd()

        self.webrtc_frames_queue = None

        # Watchdog configuration
        self.watchdog_enabled = getattr(self.options_menu, "enable_watchdog", False)
        self.watchdog_timeout = WATCHDOG_TIMEOUT
        self.watchdog_interval = WATCHDOG_INTERVAL
        self.watchdog_thread = None
        self.watchdog_running = False
        self.rebuild_count = 0
        self.watchdog_paused = False

        if self.watchdog_enabled:
            hailo_logger.info(
                f"Watchdog enabled (timeout={self.watchdog_timeout}s, interval={self.watchdog_interval}s)"
            )

    def appsink_callback(self, appsink):
        hailo_logger.debug("appsink_callback triggered")
        sample = appsink.emit("pull-sample")
        if sample:
            buffer = sample.get_buffer()
            if buffer:
                format, width, height = get_caps_from_pad(appsink.get_static_pad("sink"))
                hailo_logger.debug(f"Buffer received: format={format}, size={width}x{height}")
                frame = get_numpy_from_buffer(buffer, format, width, height)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    self.webrtc_frames_queue.put(frame)
                except queue.Full:
                    hailo_logger.warning("Frame queue full; dropping frame")
        return Gst.FlowReturn.OK

    def on_fps_measurement(self, sink, fps, droprate, avgfps):
        hailo_logger.info(f"FPS measurement: {fps:.2f}, drop={droprate:.2f}, avg={avgfps:.2f}")
        return True

    def _watchdog_monitor(self):
        """
        Monitors pipeline health by checking frame processing progress.
        Triggers pipeline rebuild if no frames are processed for watchdog_timeout duration.

        Optimization: Timestamp is only checked here (every WATCHDOG_INTERVAL), not on every frame,
        to minimize overhead in the callback hot path.
        """
        hailo_logger.info("Watchdog monitor thread started")
        last_check_count = -1
        last_progress_time = time.time()

        while self.watchdog_running:
            # Sleep first to avoid checking immediately after start
            time.sleep(self.watchdog_interval)

            if self.watchdog_paused:
                # If paused (e.g. during rebuild), update timestamp to avoid immediate trigger upon resume
                last_progress_time = time.time()
                continue

            current_count = self.user_data.get_count()
            current_time = time.time()

            if current_count > last_check_count:
                # Progress detected - update both counter and timestamp
                last_check_count = current_count
                last_progress_time = current_time
                hailo_logger.debug(f"Watchdog: Pipeline healthy. Frame count: {current_count}")
            else:
                # No progress since last check
                elapsed = current_time - last_progress_time
                hailo_logger.debug(f"Watchdog: No new frames for {elapsed:.1f}s")

                if elapsed >= self.watchdog_timeout:
                    hailo_logger.warning(
                        f"\033[91mWatchdog detected stall! No frames for {elapsed:.1f}s. Initiating rebuild...\033[0m"
                    )
                    # Use timeout_add with high priority instead of idle_add
                    # This ensures rebuild executes even if main loop is busy/stuck
                    # Priority HIGH (default is DEFAULT=0, HIGH=-100) runs before most callbacks
                    GLib.timeout_add(10, self._rebuild_pipeline, priority=GLib.PRIORITY_HIGH)

                    # Reset timer to prevent multiple triggers while waiting for rebuild
                    last_progress_time = current_time
                    # Pause watchdog until rebuild completes
                    self.watchdog_paused = True

        hailo_logger.info("Watchdog monitor thread exited")

    def create_pipeline(self):
        hailo_logger.debug("Creating pipeline...")
        Gst.init(None)
        pipeline_string = self.get_pipeline_string()
        hailo_logger.debug(f"Pipeline string: {pipeline_string}")
        try:
            self.pipeline = Gst.parse_launch(pipeline_string)
        except Exception as e:
            hailo_logger.error(f"Error creating pipeline: {e}")
            sys.exit(1)

        if self.show_fps:
            hailo_logger.debug("Connecting FPS measurement callback")
            hailo_display = self.pipeline.get_by_name("hailo_display")
            if hailo_display is not None:
                hailo_display.connect("fps-measurements", self.on_fps_measurement)
            else:
                # For multi-source pipelines, try connecting to indexed displays (hailo_display_0, etc.)
                for i in range(10):  # Check up to 10 displays
                    display = self.pipeline.get_by_name(f"hailo_display_{i}")
                    if display is not None:
                        display.connect("fps-measurements", self.on_fps_measurement)
                        hailo_logger.debug(f"Connected FPS measurement to hailo_display_{i}")
                        break
                else:
                    hailo_logger.warning("hailo_display not found - FPS measurement disabled")

        self.loop = GLib.MainLoop()

    def bus_call(self, bus, message, loop):
        t = message.type
        hailo_logger.debug(f"Bus message received: {t}")
        if t == Gst.MessageType.EOS:
            hailo_logger.debug("End of Stream")
            self.on_eos()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            hailo_logger.error(f"GStreamer Error: {err}, debug: {debug}")
            self.error_occurred = True
            self.shutdown()
        elif t == Gst.MessageType.QOS:
            if not hasattr(self, "qos_count"):
                self.qos_count = 0
            self.qos_count += 1
            # Only log every 100th QoS message to avoid spam
            # QoS messages are normal during pipeline rebuild/startup
            if self.qos_count % 100 == 0:
                qos_element = message.src.get_name()
                hailo_logger.warning(f"\033[93mQoS messages: {self.qos_count} total (from {qos_element})\033[0m")
        return True

    def on_eos(self):
        hailo_logger.debug("on_eos() called")
        if self.source_type == "file":
            hailo_logger.info("File source detected; rebuilding pipeline")
            # Use GLib.idle_add to defer pipeline rebuild and avoid blocking
            GLib.idle_add(self._rebuild_pipeline)
        else:
            hailo_logger.debug("Non-file source detected; shutting down")
            self.shutdown()

    def _connect_callback(self):
        """
        Connects the callback wrapper to the identity_callback element using the handoff signal.
        The wrapper ALWAYS runs to maintain frame counting for watchdog monitoring.
        The user callback is skipped if --disable-callback is set, but frame counting continues.
        """
        identity = self.pipeline.get_by_name("identity_callback")
        if identity is None:
            hailo_logger.warning("identity_callback not found in pipeline")
        else:
            disable_callback = self.options_menu.disable_callback
            if disable_callback:
                hailo_logger.debug("Connecting wrapper with user callback DISABLED (frame counting still active)")
            else:
                hailo_logger.debug("Connecting wrapper with user callback ENABLED")

            identity.set_property("signal-handoffs", True)
            # Always connect wrapper for frame counting (watchdog monitoring)
            # The wrapper handles skipping user callback if disable_callback is True
            identity.connect("handoff", _internal_callback_wrapper,
                           self.user_data, self.app_callback, disable_callback)

    def _rebuild_pipeline(self):
        """
        Completely rebuild the pipeline from scratch for clean looping.

        This destroys the old pipeline object and creates a new one,
        which is the only truly clean way to loop with stateful elements like trackers.

        Returns:
            False to remove this idle callback after execution.
        """
        hailo_logger.debug("_rebuild_pipeline() executing")

        # Pause watchdog to prevent false triggers during teardown/startup
        self.watchdog_paused = True
        self.rebuild_count += 1

        try:
            # Step 1: Stop and destroy the old pipeline
            hailo_logger.debug("Stopping old pipeline")
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
                # Wait briefly for NULL state
                self.pipeline.get_state(2 * Gst.SECOND)
                # Remove bus watch
                bus = self.pipeline.get_bus()
                bus.remove_signal_watch()
                # Dereference the pipeline
                self.pipeline = None

            hailo_logger.debug("Old pipeline destroyed")

            # Small delay to ensure all resources are released
            time.sleep(0.2)

            # Step 2: Rebuild the pipeline from scratch
            hailo_logger.debug("Creating new pipeline")
            pipeline_string = self.get_pipeline_string()
            hailo_logger.debug(f"New pipeline string: {pipeline_string}")

            self.pipeline = Gst.parse_launch(pipeline_string)

            # Step 3: Reattach bus callback
            hailo_logger.debug("Reattaching bus callback")
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self.bus_call, self.loop)

            # Step 4: Reattach callback
            self._connect_callback()

            # Step 5: Disable QoS on all elements to prevent frame drops
            disable_qos(self.pipeline)

            # Step 6: Start the new pipeline
            hailo_logger.debug("Starting new pipeline")
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                hailo_logger.error("Failed to start new pipeline")
                self.loop.quit()
                return False

            hailo_logger.debug("Pipeline rebuilt and restarted successfully")

            # Resume watchdog monitoring
            self.watchdog_paused = False

        except Exception as e:
            hailo_logger.error(f"Exception during pipeline rebuild: {e}")
            traceback.print_exc()
            self.loop.quit()

        # Return False to remove this idle callback
        return False

    def shutdown(self, signum=None, frame=None):
        hailo_logger.warning("Shutdown initiated")

        # Stop watchdog first
        if self.watchdog_running:
            self.watchdog_running = False
            if self.watchdog_thread and self.watchdog_thread.is_alive():
                self.watchdog_thread.join(timeout=2.0)
            self.watchdog_thread = None

        print("Shutting down... Hit Ctrl-C again to force quit.")
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        self.pipeline.set_state(Gst.State.PAUSED)
        GLib.usleep(100000)

        self.pipeline.set_state(Gst.State.READY)
        GLib.usleep(100000)

        self.pipeline.set_state(Gst.State.NULL)
        GLib.idle_add(self.loop.quit)

    def update_fps_caps(self, new_fps=30, source_name="source"):
        hailo_logger.debug(
            f"update_fps_caps() called with new_fps={new_fps}, source_name={source_name}"
        )
        videorate_name = f"{source_name}_videorate"
        capsfilter_name = f"{source_name}_fps_caps"

        videorate = self.pipeline.get_by_name(videorate_name)
        if videorate is None:
            hailo_logger.error(f"Element {videorate_name} not found")
            return

        current_max_rate = videorate.get_property("max-rate")
        hailo_logger.debug(f"Current max-rate: {current_max_rate}")
        videorate.set_property("max-rate", new_fps)
        updated_max_rate = videorate.get_property("max-rate")
        hailo_logger.debug(f"Updated max-rate to {updated_max_rate}")

        capsfilter = self.pipeline.get_by_name(capsfilter_name)
        if capsfilter:
            new_caps_str = f"video/x-raw, framerate={new_fps}/1"
            hailo_logger.debug(f"Updating capsfilter to: {new_caps_str}")
            capsfilter.set_property("caps", Gst.Caps.from_string(new_caps_str))
        self.frame_rate = new_fps

    def get_pipeline_string(self):
        hailo_logger.debug("get_pipeline_string() called (should be overridden)")
        return ""

    def dump_dot_file(self):
        hailo_logger.info("Dumping GStreamer dot file")
        Gst.debug_bin_to_dot_file(self.pipeline, Gst.DebugGraphDetails.ALL, "pipeline")
        return False

    def run(self):
        hailo_logger.debug("Running GStreamerApp main loop")
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, self.loop)

        self._connect_callback()

        hailo_display = self.pipeline.get_by_name("hailo_display")
        if hailo_display is None and not getattr(self.options_menu, "ui", False):
            hailo_logger.warning("hailo_display not found in pipeline")

        disable_qos(self.pipeline)

        if self.options_menu.use_frame:
            hailo_logger.debug("Starting display_user_data_frame process")
            display_process = multiprocessing.Process(
                target=display_user_data_frame, args=(self.user_data,)
            )
            display_process.start()

        if self.source_type == RPI_NAME_I:
            hailo_logger.debug("Starting picamera_thread")
            picam_thread = threading.Thread(
                target=picamera_thread,
                args=(self.pipeline, self.video_width, self.video_height, self.video_format),
            )
            self.threads.append(picam_thread)
            picam_thread.start()

        self.pipeline.set_state(Gst.State.PAUSED)
        self.pipeline.set_latency(self.pipeline_latency * Gst.MSECOND)
        self.pipeline.set_state(Gst.State.PLAYING)

        if self.watchdog_enabled and not self.watchdog_running:
            self.watchdog_running = True
            self.watchdog_thread = threading.Thread(target=self._watchdog_monitor, daemon=True)
            self.watchdog_thread.start()

        if self.options_menu.dump_dot:
            GLib.timeout_add_seconds(3, self.dump_dot_file)

        self.loop.run()
        # Gtk.main()

        try:
            hailo_logger.debug("Cleaning up after loop exit")
            self.user_data.running = False
            self.pipeline.set_state(Gst.State.NULL)
            if self.options_menu.use_frame:
                display_process.terminate()
                display_process.join()
            for t in self.threads:
                t.join()
        except Exception as e:
            hailo_logger.error(f"Error during cleanup: {e}")
        finally:
            if self.error_occurred:
                hailo_logger.error("Exiting with error")
                sys.exit(1)
            else:
                hailo_logger.info("Exiting successfully")
                sys.exit(0)


def picamera_thread(pipeline, video_width, video_height, video_format, picamera_config=None):
    hailo_logger.debug("picamera_thread started")
    appsrc = pipeline.get_by_name("app_source")
    appsrc.set_property("is-live", True)
    appsrc.set_property("format", Gst.Format.TIME)
    hailo_logger.debug(f"appsrc properties: {appsrc}")

    with Picamera2() as picam2:
        if picamera_config is None:
            main = {"size": (1280, 720), "format": "RGB888"}
            lores = {"size": (video_width, video_height), "format": "RGB888"}
            controls = {"FrameRate": 30}
            config = picam2.create_preview_configuration(main=main, lores=lores, controls=controls)
        else:
            config = picamera_config

        picam2.configure(config)
        lores_stream = config["lores"]
        format_str = "RGB" if lores_stream["format"] == "RGB888" else video_format
        width, height = lores_stream["size"]
        hailo_logger.debug(f"Picamera2 config: width={width}, height={height}, format={format_str}")

        appsrc.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw, format={format_str}, width={width}, height={height}, framerate=30/1, pixel-aspect-ratio=1/1"
            ),
        )
        picam2.start()
        frame_count = 0
        hailo_logger.info("picamera_process started")

        while True:
            frame_data = picam2.capture_array("lores")
            if frame_data is None:
                hailo_logger.error("Failed to capture frame")
                break

            frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
            buffer = Gst.Buffer.new_wrapped(frame.tobytes())
            buffer_duration = Gst.util_uint64_scale_int(1, Gst.SECOND, 30)
            buffer.pts = frame_count * buffer_duration
            buffer.duration = buffer_duration
            ret = appsrc.emit("push-buffer", buffer)

            if ret == Gst.FlowReturn.FLUSHING:
                hailo_logger.warning("Pipeline flushing; stopping picamera_thread")
                break
            if ret != Gst.FlowReturn.OK:
                hailo_logger.error(f"Failed to push buffer: {ret}")
                break
            frame_count += 1


