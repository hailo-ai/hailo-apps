# Standard library imports
import os
import sys
import threading
import queue
from pathlib import Path

# Third-party imports
import gi

gi.require_version("Gst", "1.0")

import hailo
try:
    from hailo import HailoTracker
except Exception:  # pragma: no cover
    HailoTracker = None
import cv2
from PIL import Image

# Local application-specific imports
from hailo_apps.python.pipeline_apps.license_plate_recognition.license_plate_recognition_pipeline import (
    GStreamerLPRApp,
)

from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer_efficient
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class

hailo_logger = get_logger(__name__)


class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.output_file = "ocr_results.txt"
        self.save_vehicle_crops = False
        self.save_lp_crops = False
        self.crops_dir = "lpr_crops"
        self.disable_found_lp_gate = False

        # Per-track state
        self.found_lp_tracks: set[int] = set()
        self.vehicle_tracks: dict[int, dict] = {}

        # Async saver (avoid blocking the GStreamer thread)
        self._save_queue = queue.Queue(maxsize=256)
        self._save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self._save_thread.start()

    def _save_worker(self):
        while True:
            item = self._save_queue.get()
            if item is None:
                return
            path_str, frame = item
            try:
                path = Path(path_str)
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(frame).save(path, format="JPEG", quality=90)
            except Exception as e:
                hailo_logger.debug("Failed saving crop to %s: %s", path_str, e)
            finally:
                self._save_queue.task_done()

    def enqueue_crop_save(self, frame, path: Path) -> None:
        try:
            self._save_queue.put_nowait((str(path), frame))
        except queue.Full:
            # Drop if the system can't keep up
            pass

    def write_ocr_text(self, text: str, confidence: float | None = None) -> None:
        with open(self.output_file, "a", encoding="utf-8") as f:
            prefix = f"Frame {self.get_count()}: "
            if confidence is None:
                f.write(f"{prefix}{text}\n")
            else:
                f.write(f"{prefix}{text} (Confidence: {confidence:.2f})\n")

    @staticmethod
    def _crop_from_bbox(frame, bbox, width: int, height: int, pad_frac: float = 0.0):
        x_min = max(0.0, min(bbox.xmin() - pad_frac, 1.0))
        y_min = max(0.0, min(bbox.ymin() - pad_frac, 1.0))
        x_max = max(0.0, min(bbox.xmax() + pad_frac, 1.0))
        y_max = max(0.0, min(bbox.ymax() + pad_frac, 1.0))
        x_min_i = int(x_min * width)
        y_min_i = int(y_min * height)
        x_max_i = int(x_max * width)
        y_max_i = int(y_max * height)
        if x_max_i <= x_min_i or y_max_i <= y_min_i:
            return None
        return frame[y_min_i:y_max_i, x_min_i:x_max_i]


def _iter_classifications(roi):
    if roi is None:
        return

    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    for det in detections:
        for cls in det.get_objects_typed(hailo.HAILO_CLASSIFICATION):
            yield det, cls

    for cls in roi.get_objects_typed(hailo.HAILO_CLASSIFICATION):
        yield None, cls


def app_callback(element, buffer, user_data):
    if buffer is None:
        return

    roi = hailo.get_roi_from_buffer(buffer)
    if roi is None:
        return

    # Optional frame extraction (needed for crop saving)
    frame = None
    width = height = None
    if getattr(user_data, "save_vehicle_crops", False) or getattr(user_data, "save_lp_crops", False):
        pad = element.get_static_pad("src")
        fmt, width, height = get_caps_from_pad(pad)
        if fmt and width and height:
            try:
                frame = get_numpy_from_buffer_efficient(buffer, fmt, width, height)
                # Ensure saved crops are RGB regardless of negotiated format
                if fmt == "BGR" and hasattr(frame, "shape"):
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                elif fmt == "NV12" and isinstance(frame, tuple) and len(frame) == 2:
                    y_plane, uv_plane = frame
                    frame = cv2.cvtColorTwoPlane(y_plane, uv_plane, cv2.COLOR_YUV2RGB_NV12)
                elif fmt in ("YUY2", "YUYV") and hasattr(frame, "shape") and frame.ndim == 3 and frame.shape[2] == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_YUY2)
            except Exception as e:
                hailo_logger.debug("Failed extracting frame: %s", e)

    # Track vehicles and mark found_lp per tracker ID.
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    vehicles = []
    plates = []
    nested_plates_by_track: dict[int, list] = {}
    for det in detections:
        label = det.get_label() or ""
        if label == "car":
            uid = det.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            track_id = uid[0].get_id() if uid else 0
            vehicles.append((track_id, det))
        elif label == "license_plate":
            plates.append(det)

    # Plates can be nested under vehicle detections depending on the cropper/aggregator
    for _, vdet in vehicles:
        uid = vdet.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        track_id = uid[0].get_id() if uid else 0
        for nested in vdet.get_objects_typed(hailo.HAILO_DETECTION):
            if (nested.get_label() or "") == "license_plate":
                nested_plates_by_track.setdefault(track_id, []).append(nested)
                plates.append(nested)

    # Build a quick index from plate -> best-matching vehicle (center-in-bbox)
    newly_found_tracks: set[int] = set()
    # Prefer direct association when plates are nested under a vehicle detection.
    if nested_plates_by_track:
        for track_id, plate_list in nested_plates_by_track.items():
            if track_id in getattr(user_data, "found_lp_tracks", set()):
                continue
            if not plate_list:
                continue
            newly_found_tracks.add(track_id)

            # Save LP crop once per track (first time it's found).
            # Nested plate bboxes are typically relative to the vehicle ROI; crop vehicle first, then plate.
            if frame is not None and getattr(user_data, "save_lp_crops", False):
                # Find the matching vehicle detection for this track
                vdet = next((d for tid, d in vehicles if tid == track_id), None)
                if vdet is not None:
                    vb = vdet.get_bbox()
                    vehicle_crop = user_data._crop_from_bbox(frame, vb, width, height, pad_frac=0.02)
                    if vehicle_crop is not None:
                        ph, pw = vehicle_crop.shape[0], vehicle_crop.shape[1]
                        pb = plate_list[0].get_bbox()
                        lp_crop = user_data._crop_from_bbox(vehicle_crop, pb, pw, ph, pad_frac=0.02)
                        if lp_crop is not None:
                            out_dir = Path(getattr(user_data, "crops_dir", "lpr_crops")) / "lp" / f"track_{track_id}"
                            out_path = out_dir / f"frame_{user_data.get_count():06d}.jpg"
                            user_data.enqueue_crop_save(lp_crop, out_path)
    elif vehicles and plates:
        for plate in plates:
            pb = plate.get_bbox()
            pcx = (pb.xmin() + pb.xmax()) / 2.0
            pcy = (pb.ymin() + pb.ymax()) / 2.0
            best_track = None
            best_area = None
            for track_id, vdet in vehicles:
                vb = vdet.get_bbox()
                if vb.xmin() <= pcx <= vb.xmax() and vb.ymin() <= pcy <= vb.ymax():
                    area = vb.width() * vb.height()
                    if best_area is None or area < best_area:
                        best_area = area
                        best_track = track_id
            if best_track is None:
                continue
            if best_track in getattr(user_data, "found_lp_tracks", set()):
                continue
            newly_found_tracks.add(best_track)

            # Save LP crop once per track (first time it's found)
            if frame is not None and getattr(user_data, "save_lp_crops", False):
                lp_crop = user_data._crop_from_bbox(frame, pb, width, height, pad_frac=0.02)
                if lp_crop is not None:
                    out_dir = Path(getattr(user_data, "crops_dir", "lpr_crops")) / "lp" / f"track_{best_track}"
                    out_path = out_dir / f"frame_{user_data.get_count():06d}.jpg"
                    user_data.enqueue_crop_save(lp_crop, out_path)

    # Update per-track vehicle info and optionally save vehicle crops (until LP is found)
    for track_id, vdet in vehicles:
        vb = vdet.get_bbox()
        conf = vdet.get_confidence()
        found = (track_id in getattr(user_data, "found_lp_tracks", set())) or (track_id in newly_found_tracks)
        user_data.vehicle_tracks[track_id] = {
            "bbox": (vb.xmin(), vb.ymin(), vb.xmax(), vb.ymax()),
            "confidence": float(conf),
            "last_seen_frame": int(user_data.get_count()),
            "found_lp": bool(found),
        }

        if frame is not None and getattr(user_data, "save_vehicle_crops", False) and not found:
            v_crop = user_data._crop_from_bbox(frame, vb, width, height, pad_frac=0.02)
            if v_crop is not None:
                out_dir = Path(getattr(user_data, "crops_dir", "lpr_crops")) / "vehicle" / f"track_{track_id}"
                out_path = out_dir / f"frame_{user_data.get_count():06d}.jpg"
                user_data.enqueue_crop_save(v_crop, out_path)

    # If we found a plate for a track, mark it as found_lp and gate further LP runs for that track
    if (newly_found_tracks or getattr(user_data, "found_lp_tracks", set())) and not getattr(
        user_data, "disable_found_lp_gate", False
    ):
        try:
            tracker = HailoTracker.get_instance() if HailoTracker is not None else None
            tracker_names = tracker.get_trackers_list() if tracker is not None else []
            tracker_name = tracker_names[0] if tracker_names else None
        except Exception:
            tracker = None
            tracker_name = None

        # 1) Mark newly-found tracks
        for track_id in newly_found_tracks:
            user_data.found_lp_tracks.add(track_id)

        # 2) Ensure gating metadata is present for all found tracks (idempotent)
        for track_id, vdet in vehicles:
            if track_id not in user_data.found_lp_tracks:
                continue

            existing = vdet.get_objects_typed(hailo.HAILO_CLASSIFICATION)
            existing_types = {c.get_classification_type() for c in existing}

            if "found_lp" not in existing_types:
                found_cls = hailo.HailoClassification(type="found_lp", label="yes", confidence=1.0)
                vdet.add_object(found_cls)
                if tracker and tracker_name:
                    try:
                        tracker.remove_classifications_from_track(tracker_name, track_id, "found_lp")
                        tracker.add_object_to_track(tracker_name, track_id, found_cls)
                    except Exception:
                        pass

            # Gate the vehicle cropper (vehicles_without_ocr) by attaching a "text_region" classification type.
            # The cropper checks for classification_type=="text_region" and skips cropping in that case.
            if "text_region" not in existing_types:
                gate_cls = hailo.HailoClassification(type="text_region", label="", confidence=1.0)
                vdet.add_object(gate_cls)
                if tracker and tracker_name:
                    try:
                        tracker.remove_classifications_from_track(tracker_name, track_id, "text_region")
                        tracker.add_object_to_track(tracker_name, track_id, gate_cls)
                    except Exception:
                        pass

    for _, cls in _iter_classifications(roi):
        label = cls.get_label()
        if not label:
            continue
        confidence = cls.get_confidence()
        print(f"OCR Result: '{label}' (Confidence: {confidence:.2f})")
        try:
            user_data.write_ocr_text(label, confidence)
        except Exception as e:
            hailo_logger.debug(f"Failed writing OCR text: {e}")

    return


def main():
    hailo_logger.info("Starting Hailo LPR App...")
    try:
        user_data = user_app_callback_class()
        app = GStreamerLPRApp(app_callback, user_data)
        app.run()
    except KeyboardInterrupt:
        hailo_logger.info("Interrupted by user")
        print("\nInterrupted by user")
    except Exception as e:
        hailo_logger.error(f"Error in main: {e}", exc_info=True)
        print(f"Error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
