# region imports
import gi

gi.require_version("Gst", "1.0")
import numpy as np
from gi.repository import Gst

from .defines import HAILO_NV12_VIDEO_FORMAT, HAILO_RGB_VIDEO_FORMAT, HAILO_YUYV_VIDEO_FORMAT
from .hailo_logger import get_logger

hailo_logger = get_logger(__name__)
# endregion imports


def get_caps_from_pad(pad: Gst.Pad):
    hailo_logger.debug("Getting caps from pad...")
    caps = pad.get_current_caps()
    if caps:
        structure = caps.get_structure(0)
        if structure:
            format = structure.get_value("format")
            width = structure.get_value("width")
            height = structure.get_value("height")
            hailo_logger.debug(
                f"Caps extracted - Format: {format}, Width: {width}, Height: {height}"
            )
            return format, width, height
    hailo_logger.warning("No caps found on pad.")
    return None, None, None


def handle_rgb(map_info, width, height):
    hailo_logger.debug(f"Handling RGB frame - Width: {width}, Height: {height}")
    return np.ndarray(shape=(height, width, 3), dtype=np.uint8, buffer=map_info.data).copy()


def handle_nv12(map_info, width, height):
    hailo_logger.debug(f"Handling NV12 frame - Width: {width}, Height: {height}")
    y_plane_size = width * height
    y_plane = np.ndarray(
        shape=(height, width), dtype=np.uint8, buffer=map_info.data[:y_plane_size]
    ).copy()
    uv_plane = np.ndarray(
        shape=(height // 2, width // 2, 2), dtype=np.uint8, buffer=map_info.data[y_plane_size:]
    ).copy()
    return y_plane, uv_plane


def handle_yuyv(map_info, width, height):
    hailo_logger.debug(f"Handling YUYV frame - Width: {width}, Height: {height}")
    return np.ndarray(shape=(height, width, 2), dtype=np.uint8, buffer=map_info.data).copy()


FORMAT_HANDLERS = {
    HAILO_RGB_VIDEO_FORMAT: handle_rgb,
    HAILO_NV12_VIDEO_FORMAT: handle_nv12,
    HAILO_YUYV_VIDEO_FORMAT: handle_yuyv,
}


def get_numpy_from_buffer(buffer, format, width, height):
    hailo_logger.debug(
        f"Converting GstBuffer to numpy - Format: {format}, Width: {width}, Height: {height}"
    )
    success, map_info = buffer.map(Gst.MapFlags.READ)
    if not success:
        hailo_logger.error("Buffer mapping failed")
        raise ValueError("Buffer mapping failed")

    try:
        handler = FORMAT_HANDLERS.get(format)
        if handler is None:
            hailo_logger.error(f"Unsupported format: {format}")
            raise ValueError(f"Unsupported format: {format}")
        hailo_logger.debug(f"Using handler: {handler.__name__}")
        return handler(map_info, width, height)
    finally:
        buffer.unmap(map_info)
        hailo_logger.debug("Buffer unmapped successfully")


def get_numpy_from_buffer_efficient(buffer, format, width, height):
    hailo_logger.debug(
        f"Efficient conversion GstBuffer to numpy - Format: {format}, Width: {width}, Height: {height}"
    )
    handler = FORMAT_HANDLERS.get(format)
    if handler is None:
        hailo_logger.error(f"Unsupported format: {format}")
        raise ValueError(f"Unsupported format: {format}")

    success, map_info = buffer.map(Gst.MapFlags.READ)
    if not success:
        hailo_logger.error("Buffer mapping failed")
        raise ValueError("Buffer mapping failed")

    try:
        hailo_logger.debug(f"Using handler: {handler.__name__}")
        return handler(map_info, width, height)
    finally:
        buffer.unmap(map_info)
        hailo_logger.debug("Buffer unmapped successfully")
