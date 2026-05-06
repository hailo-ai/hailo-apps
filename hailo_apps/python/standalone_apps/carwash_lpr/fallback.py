from typing import Optional, Tuple

import numpy as np

from hailo_apps.python.core.common.hailo_logger import get_logger

logger = get_logger(__name__)

# Lazily imported — only load PaddleOCR dependencies on first use
_paddle_infer = None


class PaddleOCRFallback:
    def __init__(self, det_hef_path: str, ocr_hef_path: str):
        self._det_hef = det_hef_path
        self._ocr_hef = ocr_hef_path

    def _run_ocr(self, crop: np.ndarray):
        global _paddle_infer
        if _paddle_infer is None:
            from hailo_apps.python.standalone_apps.paddle_ocr.paddle_ocr import (
                run_inference_pipeline,
            )
            _paddle_infer = run_inference_pipeline
        from hailo_apps.python.standalone_apps.paddle_ocr.paddle_ocr_utils import (
            decode_ocr_results,
        )
        from hailo_apps.python.core.common.hailo_inference import HailoInfer
        import threading

        ocr_model = HailoInfer(self._ocr_hef)
        result_holder = []
        done = threading.Event()

        def callback(completion_info, bindings_list, _input_batch):
            if not completion_info.exception:
                buf = bindings_list[0].output().get_buffer()
                result_holder.extend(decode_ocr_results(buf, _PLATE_CHARS))
            done.set()

        h, w = ocr_model.get_input_shape()[:2]
        import cv2
        resized = cv2.resize(crop, (w, h))
        ocr_model.run([resized], callback)
        done.wait(timeout=5.0)
        return result_holder

    def read_plate(self, crop: np.ndarray) -> Optional[Tuple[str, float]]:
        try:
            results = self._run_ocr(crop)
            if not results:
                return None
            return results[0]
        except Exception as exc:
            logger.warning(f"PaddleOCR fallback failed: {exc}")
            return None


# Alphanumeric character set for US/CA plates (index 0 = CTC blank)
_PLATE_CHARS = [""] + list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
