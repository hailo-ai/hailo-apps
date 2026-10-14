"""DeGirum local inference wrappers — vehicle detector, plate detector, OCR."""

import os
from dataclasses import dataclass
from typing import List

import degirum as dg
import numpy as np

from hailo_apps.python.core.common.hailo_logger import get_logger

logger = get_logger(__name__)

_DG_HOST = "@local"
_DG_ZOO = "degirum/models_hailort"

VEHICLE_MODEL = "yolov8n_relu6_coco--640x640_quant_hailort_hailo8l_1"
PLATE_MODEL   = "yolov8n_relu6_lp--640x640_quant_hailort_hailo8l_1"
OCR_MODEL     = "yolov8n_relu6_lp_ocr--256x128_quant_hailort_hailo8l_1"

# Vehicle classes to keep from a COCO detector (per MODELS.md)
VEHICLE_CLASS_LABELS = {"car", "truck", "bus", "motorcycle"}


def _load(model_name: str, **extra) -> "dg.Model":
    return dg.load_model(
        model_name=model_name,
        inference_host_address=_DG_HOST,
        zoo_url=_DG_ZOO,
        token=os.environ.get("DEGIRUM_CLOUD_TOKEN", ""),
        **extra,
    )


@dataclass
class Detection:
    bbox: list           # [x1, y1, x2, y2] in input-image pixel coords
    score: float
    label: str


@dataclass
class PlateDetection:
    bbox: list           # plate bbox in vehicle-crop pixel coords
    score: float


@dataclass
class OCRResult:
    plate_string: str
    confidence: float


class VehicleDetector:
    def __init__(self, model_name: str = VEHICLE_MODEL):
        self._model = _load(model_name)
        logger.info("VehicleDetector ready: %s", model_name)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        result = self._model(frame)
        out = []
        for r in result.results or []:
            label = (r.get("label") or "").lower()
            if VEHICLE_CLASS_LABELS and label not in VEHICLE_CLASS_LABELS:
                continue
            out.append(Detection(bbox=list(r["bbox"]), score=float(r["score"]), label=label))
        return out


class PlateDetector:
    def __init__(self, model_name: str = PLATE_MODEL):
        self._model = _load(model_name)
        logger.info("PlateDetector ready: %s", model_name)

    def detect(self, vehicle_crop: np.ndarray) -> List[PlateDetection]:
        result = self._model(vehicle_crop)
        return [
            PlateDetection(bbox=list(r["bbox"]), score=float(r["score"]))
            for r in (result.results or [])
        ]


class PlateOCR:
    def __init__(self, model_name: str = OCR_MODEL):
        self._model = _load(
            model_name,
            output_use_regular_nms=False,
            output_confidence_threshold=0.1,
        )
        logger.info("PlateOCR ready: %s", model_name)

    def read(self, plate_crop: np.ndarray) -> OCRResult:
        result = self._model(plate_crop)
        chars = sorted(result.results or [], key=lambda r: r["bbox"][0])
        if not chars:
            return OCRResult(plate_string="", confidence=0.0)
        plate_str = "".join(c["label"] for c in chars)
        avg_conf = sum(c["score"] for c in chars) / len(chars)
        return OCRResult(plate_string=plate_str, confidence=float(avg_conf))
