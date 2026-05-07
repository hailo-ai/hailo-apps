"""
YOLOv8 DFL post-processing for the LP detector and OCR models.

Both models output 6 raw uint8 feature maps (3 scales × box-reg + class-scores).
This module dequantizes, DFL-decodes, applies sigmoid, then NMS to produce final
detections.

Detector  → list of (x1, y1, x2, y2, score) in input-image pixel coords
OCR model → list of (char_label_str, x_center_px, score), sorted left-to-right
"""

from typing import Dict, List, Tuple

import cv2
import numpy as np

# 36 character classes: 0-9 then A-Z (index matches class_id from model)
_OCR_CHARS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

REG_MAX = 16  # YOLOv8 DFL reg_max

# Confidence / NMS thresholds
DET_SCORE_THRESH = 0.30
DET_NMS_THRESH   = 0.45
OCR_SCORE_THRESH = 0.25
OCR_NMS_THRESH   = 0.45


def _dequant(arr: np.ndarray, scale: float, zp: float) -> np.ndarray:
    return (arr.astype(np.float32) - zp) * scale


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _dfl_to_xyxy(
    box_feat: np.ndarray,   # (H, W, 4*REG_MAX) float32
    stride: int,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """Return (H*W, 4) array of [x1, y1, x2, y2] pixel coords."""
    H, W = box_feat.shape[:2]

    # Anchor centres
    cols = (np.arange(W, dtype=np.float32) + 0.5) * stride
    rows = (np.arange(H, dtype=np.float32) + 0.5) * stride
    cx = np.tile(cols, H)           # (H*W,)
    cy = np.repeat(rows, W)         # (H*W,)

    # DFL decode
    flat = box_feat.reshape(-1, 4, REG_MAX)       # (H*W, 4, 16)
    sm   = _softmax(flat)                          # (H*W, 4, 16)
    proj = np.arange(REG_MAX, dtype=np.float32)
    dist = (sm * proj).sum(axis=-1)               # (H*W, 4) = [l, t, r, b]

    x1 = np.clip(cx - dist[:, 0], 0, img_w)
    y1 = np.clip(cy - dist[:, 1], 0, img_h)
    x2 = np.clip(cx + dist[:, 2], 0, img_w)
    y2 = np.clip(cy + dist[:, 3], 0, img_h)

    return np.stack([x1, y1, x2, y2], axis=1)     # (H*W, 4)


def _collect_scales(
    raw: Dict[str, np.ndarray],
    quant: Dict[str, Tuple[float, float]],
    img_h: int,
    img_w: int,
    strides: Tuple[int, ...] = (8, 16, 32),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect and decode all three scales.

    raw:   {layer_name: uint8 (H, W, C)}
    quant: {layer_name: (scale, zp)}

    Returns boxes (N, 4) and scores (N, nc).
    """
    boxes_all, scores_all = [], []
    names = sorted(raw.keys())                    # deterministic order
    box_names = [n for n in names if raw[n].shape[2] == 4 * REG_MAX]
    cls_names = [n for n in names if raw[n].shape[2] != 4 * REG_MAX]

    # Sort both by H descending (largest grid first)
    box_names.sort(key=lambda n: raw[n].shape[0], reverse=True)
    cls_names.sort(key=lambda n: raw[n].shape[0], reverse=True)

    for i, (bn, cn) in enumerate(zip(box_names, cls_names)):
        s, z   = quant[bn]
        box_f  = _dequant(raw[bn], s, z)          # (H, W, 64) float32
        s2, z2 = quant[cn]
        cls_f  = _dequant(raw[cn], s2, z2)        # (H, W, nc) float32

        stride  = strides[i]
        boxes   = _dfl_to_xyxy(box_f, stride, img_h, img_w)   # (H*W, 4)
        scores  = 1.0 / (1.0 + np.exp(-cls_f.reshape(-1, cls_f.shape[2])))  # sigmoid

        boxes_all.append(boxes)
        scores_all.append(scores)

    return np.concatenate(boxes_all, axis=0), np.concatenate(scores_all, axis=0)


def decode_detector(
    raw: Dict[str, np.ndarray],
    quant: Dict[str, Tuple[float, float]],
    img_h: int = 640,
    img_w: int = 640,
) -> List[Tuple[List[float], float]]:
    """
    Returns list of ([x1, y1, x2, y2] pixel, score) for detected plates.
    Coordinates are in the 640×640 input space.
    """
    boxes, scores = _collect_scales(raw, quant, img_h, img_w)
    plate_scores  = scores[:, 0]                  # single class

    mask = plate_scores > DET_SCORE_THRESH
    if not mask.any():
        return []

    boxes_f  = boxes[mask].tolist()
    scores_f = plate_scores[mask].tolist()

    indices = cv2.dnn.NMSBoxes(
        [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes_f],
        scores_f,
        DET_SCORE_THRESH,
        DET_NMS_THRESH,
    )

    if len(indices) == 0:
        return []

    return [(boxes_f[i], scores_f[i]) for i in indices.flatten()]


def decode_ocr(
    raw: Dict[str, np.ndarray],
    quant: Dict[str, Tuple[float, float]],
    img_h: int = 128,
    img_w: int = 256,
) -> str:
    """
    Decodes character detections from the OCR model.
    Returns plate string sorted left-to-right, or empty string if no detections.
    """
    boxes, scores = _collect_scales(raw, quant, img_h, img_w)
    class_ids  = scores.argmax(axis=1)
    max_scores = scores.max(axis=1)

    mask = max_scores > OCR_SCORE_THRESH
    if not mask.any():
        return ""

    boxes_f  = boxes[mask].tolist()
    scores_f = max_scores[mask].tolist()
    class_f  = class_ids[mask].tolist()

    indices = cv2.dnn.NMSBoxes(
        [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes_f],
        scores_f,
        OCR_SCORE_THRESH,
        OCR_NMS_THRESH,
    )

    if len(indices) == 0:
        return ""

    chars = []
    for i in indices.flatten():
        x_center = (boxes_f[i][0] + boxes_f[i][2]) / 2.0
        chars.append((x_center, _OCR_CHARS[class_f[i]], scores_f[i]))

    chars.sort(key=lambda c: c[0])
    return "".join(c[1] for c in chars)
