import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from hailo_apps.python.standalone_apps.carwash_lpr.fallback import PaddleOCRFallback


def _crop():
    return np.zeros((128, 256, 3), dtype=np.uint8)


def test_returns_none_on_exception():
    fallback = PaddleOCRFallback.__new__(PaddleOCRFallback)
    fallback._run_ocr = MagicMock(side_effect=Exception("paddle error"))
    result = fallback.read_plate(_crop())
    assert result is None


def test_returns_plate_string_and_confidence():
    fallback = PaddleOCRFallback.__new__(PaddleOCRFallback)
    fallback._run_ocr = MagicMock(return_value=[("ABC123", 0.88)])
    result = fallback.read_plate(_crop())
    assert result is not None
    plate, conf = result
    assert plate == "ABC123"
    assert conf == pytest.approx(0.88)


def test_returns_none_on_empty_result():
    fallback = PaddleOCRFallback.__new__(PaddleOCRFallback)
    fallback._run_ocr = MagicMock(return_value=[])
    result = fallback.read_plate(_crop())
    assert result is None
