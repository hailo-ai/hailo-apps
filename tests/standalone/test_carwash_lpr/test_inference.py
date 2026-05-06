import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from hailo_apps.python.standalone_apps.carwash_lpr.inference import PlateInference, PlateRead


def _make_ocr_output(char_indices):
    """
    Simulate Hailo OCR output: shape [1, seq_len, num_chars].
    char_indices: list of ints (character indices into _PLATE_CHARS).
    _PLATE_CHARS = [""] + list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    So: blank=0, '0'=1, '1'=2, ..., '9'=10, 'A'=11, 'B'=12, 'C'=13, ...
    """
    num_chars = 37  # blank + 36 alphanumeric
    seq = np.zeros((1, len(char_indices), num_chars), dtype=np.float32)
    for t, idx in enumerate(char_indices):
        seq[0, t, idx] = 1.0
    return seq


def test_no_detections_returns_empty():
    inf = PlateInference.__new__(PlateInference)
    inf._det_model = MagicMock()
    inf._ocr_model = MagicMock()
    inf._det_input_hw = (640, 640)
    inf._ocr_input_hw = (128, 256)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with patch.object(inf, '_run_detector', return_value=[]):
        reads = inf.run(frame)
    assert reads == []


def test_single_plate_detected_and_decoded():
    inf = PlateInference.__new__(PlateInference)
    inf._det_input_hw = (640, 640)
    inf._ocr_input_hw = (128, 256)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = [0.1, 0.2, 0.3, 0.5]  # [y1, x1, y2, x2] normalized

    # A=11, B=12, C=13, '1'=2, '2'=3, '3'=4
    # _PLATE_CHARS = [""] + list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    # index 0=blank, 1='0', 2='1', 3='2', 4='3', ..., 11='A', 12='B', 13='C'
    char_seq = [11, 12, 13, 2, 3, 4]  # "ABC123"
    ocr_out = _make_ocr_output(char_seq)

    with patch.object(inf, '_run_detector', return_value=[(bbox, 0.92)]):
        with patch.object(inf, '_run_ocr', return_value=ocr_out):
            reads = inf.run(frame)

    assert len(reads) == 1
    assert reads[0].plate_string == "ABC123"
    assert reads[0].confidence == pytest.approx(0.92)


def test_multiple_plates_sorted_left_to_right():
    inf = PlateInference.__new__(PlateInference)
    inf._det_input_hw = (640, 640)
    inf._ocr_input_hw = (128, 256)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # [y1, x1, y2, x2]
    bbox_right = [0.1, 0.6, 0.3, 0.9]
    bbox_left  = [0.1, 0.1, 0.3, 0.4]

    # 'N'=24, 'O'=25, 'P'=26; '1'=2, '2'=3, '3'=4
    # _PLATE_CHARS indices: blank=0, digits 1-10, A=11...Z=36
    # N is 14th letter (0-indexed), so N=11+13=24; O=25; P=26
    char_seq_right = [24, 25, 26, 2, 3, 4]  # "NOP123"
    char_seq_left  = [11, 12, 13, 2, 3, 4]  # "ABC123"

    call_count = [0]
    def mock_ocr(crop):
        seq = char_seq_left if call_count[0] == 0 else char_seq_right
        call_count[0] += 1
        return _make_ocr_output(seq)

    with patch.object(inf, '_run_detector', return_value=[
        (bbox_right, 0.90), (bbox_left, 0.88)
    ]):
        with patch.object(inf, '_run_ocr', side_effect=mock_ocr):
            reads = inf.run(frame)

    # After L→R sort by x1: left (x1=0.1) first, then right (x1=0.6)
    assert len(reads) == 2
    assert reads[0].plate_string == "ABC123"
    assert reads[1].plate_string == "NOP123"
