import threading
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

from hailo_apps.python.core.common.hailo_inference import HailoInfer
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.standalone_apps.carwash_lpr.postprocess import decode_detector, decode_ocr

logger = get_logger(__name__)


@dataclass
class PlateRead:
    plate_string: str
    confidence: float
    bbox: List[float]       # [x1, y1, x2, y2] pixel coords in detector input space
    crop_frame: np.ndarray
    full_frame: np.ndarray


def _load_quant(hef_obj) -> Dict[str, Tuple[float, float]]:
    """Extract (scale, zero_point) for every output vstream from the HEF."""
    return {
        info.name: (info.quant_info.qp_scale, info.quant_info.qp_zp)
        for info in hef_obj.get_output_vstream_infos()
    }


class PlateInference:
    def __init__(self, detector_hef: str, ocr_hef: str):
        self._det_model = HailoInfer(detector_hef)
        self._ocr_model = HailoInfer(ocr_hef)

        det_shape = self._det_model.get_input_shape()
        ocr_shape = self._ocr_model.get_input_shape()
        self._det_input_hw = (det_shape[0], det_shape[1])   # (640, 640)
        self._ocr_input_hw = (ocr_shape[0], ocr_shape[1])   # (128, 256)

        self._det_quant = _load_quant(self._det_model.hef)
        self._ocr_quant = _load_quant(self._ocr_model.hef)

        self._det_lock = threading.Lock()
        self._ocr_lock = threading.Lock()

        logger.info(f"Detector input: {self._det_input_hw}, OCR input: {self._ocr_input_hw}")

    def _preprocess(self, frame: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
        h, w = target_hw
        return cv2.resize(frame, (w, h))

    def _run_raw(self, model: HailoInfer, lock: threading.Lock,
                 frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference and return dict of {output_name: uint8 array}."""
        result_holder: Dict[str, np.ndarray] = {}
        done = threading.Event()

        def callback(completion_info, bindings_list):
            if completion_info.exception:
                logger.error(f"Inference error: {completion_info.exception}")
            else:
                b = bindings_list[0]
                for name in b._output_names:
                    result_holder[name] = b.output(name).get_buffer()
            done.set()

        with lock:
            model.run([frame], callback)
            timed_out = not done.wait(timeout=5.0)

        if timed_out:
            logger.error("Inference timed out")
        return result_holder

    def _crop_bbox(self, frame: np.ndarray, bbox: List[float]) -> np.ndarray:
        """Crop [x1,y1,x2,y2] from frame (which is at detector resolution)."""
        x1, y1, x2, y2 = bbox
        fh, fw = frame.shape[:2]
        r = frame[
            max(0, int(y1)):min(fh, int(y2)),
            max(0, int(x1)):min(fw, int(x2)),
        ]
        return r.copy() if r.size > 0 else np.zeros((10, 10, 3), dtype=np.uint8)

    def close(self):
        self._det_model.close()
        self._ocr_model.close()

    def run(self, frame: np.ndarray) -> List[PlateRead]:
        det_h, det_w = self._det_input_hw
        det_frame = self._preprocess(frame, self._det_input_hw)

        raw_det = self._run_raw(self._det_model, self._det_lock, det_frame)
        if not raw_det:
            return []

        detections = decode_detector(raw_det, self._det_quant, det_h, det_w)
        if not detections:
            return []

        # Sort left-to-right by x1
        detections.sort(key=lambda d: d[0][0])

        reads = []
        for bbox, score in detections:
            crop = self._crop_bbox(det_frame, bbox)
            ocr_frame = self._preprocess(crop, self._ocr_input_hw)
            raw_ocr = self._run_raw(self._ocr_model, self._ocr_lock, ocr_frame)
            if not raw_ocr:
                continue

            ocr_h, ocr_w = self._ocr_input_hw
            plate_str = decode_ocr(raw_ocr, self._ocr_quant, ocr_h, ocr_w)
            if plate_str:
                reads.append(PlateRead(
                    plate_string=plate_str,
                    confidence=score,
                    bbox=bbox,
                    crop_frame=crop,
                    full_frame=frame.copy(),
                ))
        return reads
