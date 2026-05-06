import threading
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from hailo_apps.python.core.common.hailo_inference import HailoInfer
from hailo_apps.python.core.common.hailo_logger import get_logger

logger = get_logger(__name__)

# blank + "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_PLATE_CHARS = [""] + list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


@dataclass
class PlateRead:
    plate_string: str
    confidence: float
    bbox: List[float]       # [y1, x1, y2, x2] normalized
    crop_frame: np.ndarray
    full_frame: np.ndarray


class PlateInference:
    def __init__(self, detector_hef: str, ocr_hef: str):
        self._det_model = HailoInfer(detector_hef)
        self._ocr_model = HailoInfer(ocr_hef)
        det_shape = self._det_model.get_input_shape()
        ocr_shape = self._ocr_model.get_input_shape()
        self._det_input_hw = (det_shape[0], det_shape[1])
        self._ocr_input_hw = (ocr_shape[0], ocr_shape[1])
        logger.info(f"Detector input: {self._det_input_hw}, OCR input: {self._ocr_input_hw}")

    def _preprocess(self, frame: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
        h, w = target_hw
        return cv2.resize(frame, (w, h))

    def _run_detector(self, frame: np.ndarray) -> List[Tuple[List[float], float]]:
        """Returns list of ([y1,x1,y2,x2], score) normalized."""
        processed = self._preprocess(frame, self._det_input_hw)
        result_holder = []
        done = threading.Event()

        def callback(completion_info, bindings_list, _input_batch):
            if completion_info.exception:
                logger.error(f"Detector error: {completion_info.exception}")
            else:
                raw = bindings_list[0].output().get_buffer()
                for class_detections in raw:
                    for det in class_detections:
                        bbox = det[:4].tolist()
                        score = float(det[4])
                        if score > 0.3:
                            result_holder.append((bbox, score))
            done.set()

        self._det_model.run([processed], callback)
        done.wait(timeout=5.0)
        return result_holder

    def _run_ocr(self, crop: np.ndarray) -> np.ndarray:
        """Returns raw OCR model output array shape [1, seq_len, num_chars]."""
        processed = self._preprocess(crop, self._ocr_input_hw)
        result_holder = []
        done = threading.Event()

        def callback(completion_info, bindings_list, _input_batch):
            if completion_info.exception:
                logger.error(f"OCR error: {completion_info.exception}")
            else:
                result_holder.append(bindings_list[0].output().get_buffer())
            done.set()

        self._ocr_model.run([processed], callback)
        done.wait(timeout=5.0)
        return result_holder[0] if result_holder else np.zeros((1, 1, 37), dtype=np.float32)

    @staticmethod
    def _decode_ocr(raw: np.ndarray) -> str:
        """CTC greedy decode: argmax per timestep, remove blanks and repeats."""
        if raw.ndim == 3:
            raw = raw[0]  # (seq_len, num_chars)
        indices = raw.argmax(axis=1)
        chars = []
        prev = -1
        for idx in indices:
            if idx != 0 and idx != prev:  # 0 = CTC blank
                chars.append(_PLATE_CHARS[idx])
            prev = idx
        return "".join(chars)

    def _crop_bbox(self, frame: np.ndarray, bbox: List[float]) -> np.ndarray:
        h, w = frame.shape[:2]
        y1, x1, y2, x2 = bbox
        r = frame[
            max(0, int(y1 * h)):min(h, int(y2 * h)),
            max(0, int(x1 * w)):min(w, int(x2 * w)),
        ]
        return r if r.size > 0 else np.zeros((10, 10, 3), dtype=np.uint8)

    def run(self, frame: np.ndarray) -> List[PlateRead]:
        detections = self._run_detector(frame)
        if not detections:
            return []

        # Sort left-to-right by x1 (index 1 of bbox)
        detections.sort(key=lambda d: d[0][1])

        reads = []
        for bbox, score in detections:
            crop = self._crop_bbox(frame, bbox)
            raw_ocr = self._run_ocr(crop)
            plate_str = self._decode_ocr(raw_ocr)
            if plate_str:
                reads.append(PlateRead(
                    plate_string=plate_str,
                    confidence=score,
                    bbox=bbox,
                    crop_frame=crop,
                    full_frame=frame,
                ))
        return reads
