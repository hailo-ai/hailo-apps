import queue
import threading
import time

import numpy as np

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

from hailo_apps.python.core.common.hailo_logger import get_logger

logger = get_logger(__name__)

Gst.init(None)


class RTSPIngest:
    def __init__(self, rtsp_url: str, camera_id: str):
        self.camera_id = camera_id
        self._url = rtsp_url
        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._running = threading.Event()
        self._pipeline = None
        self._first_frame_logged = False
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name=f"ingest-{camera_id}")

    def start(self):
        self._running.set()
        self._thread.start()

    def stop(self):
        self._running.clear()
        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)
        self._thread.join()

    def _build_pipeline(self) -> Gst.Pipeline:
        pipeline_str = (
            f'rtspsrc location="{self._url}" latency=200 protocols=tcp name=src '
            f"! decodebin ! videoconvert ! video/x-raw,format=BGR "
            f"! appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false"
        )
        pipeline = Gst.parse_launch(pipeline_str)
        sink = pipeline.get_by_name("sink")
        sink.connect("new-sample", self._on_new_sample)
        return pipeline

    def _on_new_sample(self, sink) -> Gst.FlowReturn:
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR
        buf = sample.get_buffer()
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR
        frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape((height, width, 3)).copy()
        buf.unmap(map_info)
        if not self._first_frame_logged:
            logger.info(f"[{self.camera_id}] First frame received: {width}x{height}")
            self._first_frame_logged = True
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass  # discard stale frame — keep newest
        return Gst.FlowReturn.OK

    def _run_loop(self):
        backoff = 2.0
        while self._running.is_set():
            connected = False
            try:
                self._pipeline = self._build_pipeline()
                logger.info(f"[{self.camera_id}] Connecting to {self._url}")
                self._pipeline.set_state(Gst.State.PLAYING)
                self._first_frame_logged = False
                bus = self._pipeline.get_bus()
                while self._running.is_set():
                    msg = bus.timed_pop_filtered(
                        500 * Gst.MSECOND,
                        Gst.MessageType.ERROR | Gst.MessageType.EOS | Gst.MessageType.STATE_CHANGED,
                    )
                    if msg is None:
                        continue
                    if msg.type == Gst.MessageType.STATE_CHANGED:
                        if msg.src == self._pipeline:
                            _old, new, _pending = msg.parse_state_changed()
                            if new == Gst.State.PLAYING and not connected:
                                connected = True
                                logger.info(f"[{self.camera_id}] Stream live")
                    elif msg.type == Gst.MessageType.ERROR:
                        err, debug = msg.parse_error()
                        logger.error(f"[{self.camera_id}] GStreamer error: {err} — {debug}")
                        break
                    elif msg.type == Gst.MessageType.EOS:
                        break
                self._pipeline.set_state(Gst.State.NULL)
            except Exception as exc:
                logger.error(f"[{self.camera_id}] Ingest exception: {exc}")

            if connected:
                backoff = 2.0

            if self._running.is_set():
                logger.info(f"[{self.camera_id}] Reconnecting in {backoff:.0f}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            else:
                break
        logger.info(f"[{self.camera_id}] Ingest stopped")
