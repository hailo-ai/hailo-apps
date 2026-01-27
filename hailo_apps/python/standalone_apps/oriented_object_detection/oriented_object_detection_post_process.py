import cv2
import numpy as np
from typing import List, Tuple

try:
    from hailo_apps.python.core.common.toolbox import id_to_color
except ImportError:
    from pathlib import Path
    import sys

    core_dir = Path(__file__).resolve().parents[2] / "core"
    sys.path.insert(0, str(core_dir))
    from common.toolbox import id_to_color


def inference_result_handler(original_frame: np.ndarray, infer_results, labels, config_data, tracker=None):
    """
    Run oriented post-processing and draw resulting detections.

    config_data must include an `oriented_postprocess` dict with keys:
      - onnx_postprocess_model_path: path to ONNX decoder (optional)
      - onnx_postprocess_model_input_map: dict
      - score_th: float
      - nms_iou_th: float
    """
    kept_boxes, kept_classes, kept_scores = obb_postprocess(original_frame, infer_results, config_data)
    
    detections = {
        'detection_boxes': kept_boxes,
        'detection_classes': kept_classes,
        'detection_scores': kept_scores,
        'num_detections': len(kept_boxes)
    }

    return draw_detections_local(detections, original_frame, labels, tracker=tracker)

def obb_postprocess(orig_img, infer_results, config_data):
    ops = config_data.get("oriented_postprocess", {})
    obb_input_map = ops.get("obb_model_input_map", {})
    img_size = ops.get("img_size", 640)
    scores_th = ops.get("scores_th", 0.35)
    nms_iou_th = ops.get("nms_iou_th", 0.25)
    cls_num = ops.get("cls_num", 15)

    obb_inputs = prepare_ort_inputs_from_hailo(infer_results, obb_input_map)
    detections = native_obb_postprocess(obb_inputs, img_size)
    
    # extract OBB detections from postprocess outputs, incldues decoding of boxes, scores, classes and angles
    rects, scores, cls_ids = extract_obb_detections(
        detections[0], 
        orig_img=orig_img,
        cls_num=cls_num, 
        img_size=img_size, 
        scores_th=scores_th
    )
    
    # filter detections with a rotated NMS variation
    keep = rotated_nms(rects, scores, iou_thresh=nms_iou_th)
    rects_keep = [rects[i] for i in keep]
    scores_keep = [scores[i] for i in keep]
    cls_keep = [cls_ids[i] for i in keep]
    
    return rects_keep, cls_keep, scores_keep

def prepare_ort_inputs_from_hailo(infer_results, ort_input_map):
    ort_inputs = {}
    for hailo_name, ort_input_name in ort_input_map.items():
        arr = infer_results[hailo_name]
        ort_inputs[ort_input_name] = np.transpose(arr, (0,3,1,2)).astype(np.float32)
    
    return ort_inputs


def native_obb_postprocess(ort_inputs: dict, img_size: int = 640) -> Tuple[np.ndarray]:
    """
    Pure Python implementation of YOLO11 OBB postprocessing matching exact ONNX behavior.
    
    YOLO11 OBB head structure:
    - cv2: bbox regression (4 * reg_max channels with DFL)
    - cv3: class scores (num_classes channels) 
    - cv4: angle prediction (1 channel)
    
    Returns: (1, num_predictions, 4+cls_num+1) where last dim is [cx,cy,w,h, cls_scores..., angle]
    """
    # Group by head type and scale
    bbox_keys = [k for k in ort_inputs if 'cv2' in k]   # bbox with DFL
    cls_keys = [k for k in ort_inputs if 'cv3' in k]    # classes  
    angle_keys = [k for k in ort_inputs if 'cv4' in k]  # angle
    
    bbox_feat = []
    cls_feat = []
    angle_feat = []
    reg_max = 16  # fixed for YOLO11 OBB
    strides = [8, 16, 32] # fixed for YOLO11 OBB
    
    for scale_idx in range(3):
        # Get feature maps for this scale
        bbox_feat.append(ort_inputs[bbox_keys[scale_idx]])    # (1, 4*reg_max, H, W)
        cls_feat.append(ort_inputs[cls_keys[scale_idx]])      # (1, nc, H, W)
        angle_feat.append(ort_inputs[angle_keys[scale_idx]])  # (1, 1, H, W)

    # flatten and normalize angle predictions
    angles_flat = np.concatenate([np.reshape(a, (a.shape[0], a.shape[1], a.shape[2] * a.shape[3])) for a in angle_feat], axis=-1) # (1, 1, H*W)
    angles_sig = sigmoid(angles_flat)  # sigmoid
    angles_rad = (angles_sig - 0.25) * np.pi

    # concatenate boxes and class scores for each scale, and flatten
    boxes_classes_all = []
    for i in range(3):
        curr_boxes_classes = np.concatenate([bbox_feat[i], cls_feat[i]], axis=1)  # (1, 4*reg_max + nc, H, W)
        flat_shape = (curr_boxes_classes.shape[0], curr_boxes_classes.shape[1], curr_boxes_classes.shape[2] * curr_boxes_classes.shape[3])
        boxes_classes_all.append(np.reshape(curr_boxes_classes, flat_shape))  # (1, 4*reg_max + nc, H*W)

    # split boxes and classes
    boxes_classes_out = np.concatenate(boxes_classes_all, axis=-1)  # (1, 4*reg_max + nc, N)
    boxes_out, classes_out = np.split(boxes_classes_out, [4 * reg_max], axis=1)

    # decode DFL boxes
    decode_shape = (boxes_out.shape[0], 4, reg_max, boxes_out.shape[-1])
    dfl_softmax = softmax(np.transpose(np.reshape(boxes_out, decode_shape), [0, 3, 1, 2]), axis=-1)  # (B, N, 4, reg_max)
    dfl_in = np.transpose(dfl_softmax, (0, 3, 2, 1))  # (B, reg_max, 4, N)
    proj = np.arange(reg_max, dtype=np.float32).reshape(1, reg_max, 1, 1)
    decoded = np.sum(dfl_in * proj, axis=1, keepdims=True)  # weighted sum over reg_max bins
    dfl_out = np.reshape(decoded, (decoded.shape[0], 4, decoded.shape[-1]))  # (B, 4, N)

    # calculate classes sigmoid
    cls_sigmoid = sigmoid(classes_out) # sigmoid

    # convert DFL distances to boxes in pixel coordinates
    dfl_left, dfl_right = np.split(dfl_out, [2], axis=1)  # (B, 2, N) each
    dfl_div = (dfl_right - dfl_left) / 2.0
    dfl_div_left, dfl_div_right = np.split(dfl_div, [1], axis=1) # (B, 1, N) each
    
    angle_sin = np.sin(angles_rad)  # (B, 1, N)
    angle_cos = np.cos(angles_rad)  # (B, 1, N)
    dfl_div_left_sin = dfl_div_left * angle_sin  # (B, 1, N)
    dfl_div_left_cos = dfl_div_left * angle_cos  # (B, 1, N)
    dfl_div_right_sin = dfl_div_right * angle_sin  # (B, 1, N)
    dfl_div_right_cos = dfl_div_right * angle_cos  # (B, 1, N)
    dfl_add = dfl_div_left_sin + dfl_div_right_cos # (B, 1, N)
    dfl_sub = dfl_div_left_cos - dfl_div_right_sin # (B, 1, N)
    dfl_concat = np.concatenate([dfl_sub, dfl_add], axis=1)  # (B, 2, N)

    # Generate anchor grids for each scale and concatenate
    all_anchors = []
    for stride in strides:
        fm_size = img_size // stride  # 80, 40, 20 for strides 8, 16, 32
        grid_x = np.tile(np.arange(fm_size) + 0.5, fm_size)      # X coords
        grid_y = np.repeat(np.arange(fm_size) + 0.5, fm_size)    # Y coords
        all_anchors.append(np.stack([grid_x, grid_y], axis=0))  # (2, fm_size^2)
    
    dfl_coords = np.concatenate(all_anchors, axis=1)[np.newaxis, :, :]  # (1, 2, 8400)
    dfl_anchors = dfl_concat + dfl_coords  # (B, 2, N)
    dfl_sum = dfl_left + dfl_right  # (B, 2, N)
    dfl_concat_out = np.concatenate([dfl_anchors, dfl_sum], axis=1)  # (B, 4, N)

    feature_map = []
    for stride in strides:
        fm_size = img_size // stride
        feature_map.append(np.repeat(stride, fm_size * fm_size))

    feature_map_arr = np.concatenate(feature_map).reshape(1, -1)  # (1, N)
    dfl_fm_mul = dfl_concat_out * feature_map_arr
    output = np.concatenate([dfl_fm_mul, cls_sigmoid, angles_rad], axis=1)  # (B, 4 + nc + 1, N)
    return (output,)

def softmax(x, axis=-1):
    """Compute softmax values for array x along specified axis."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def extract_obb_detections(ort_outs, orig_img, cls_num, img_size=640, scores_th=0.35):
    # values used for scaling coords and removing padding
    h0, w0 = orig_img.shape[:2]
    scale = min(img_size / w0, img_size / h0)
    new_unpad = (int(round(w0 * scale)), int(round(h0 * scale)))
    dw = (img_size - new_unpad[0]) / 2
    dh = (img_size - new_unpad[1]) / 2

    idx_box_end = 4
    idx_cls_start = 4
    idx_cls_end = 4 + cls_num
    idx_angle = 4 + cls_num

    preds = np.transpose(ort_outs, (0,2,1))[0] 
    boxes_raw = preds[:, :idx_box_end].copy()               # (N,4)
    cls_part = preds[:, idx_cls_start:idx_cls_end].copy()   # (N, nc)
    angle_part = preds[:, idx_angle].copy()                 # (N,)

    if boxes_raw.max() <= 1.0 + 1e-6:
        boxes_raw = boxes_raw * img_size

    cls_min, cls_max = float(cls_part.min()), float(cls_part.max())
    if cls_min >= 0.0 and cls_max <= 1.0:
        cls_probs = cls_part
    else:
        cls_probs = 1 / (1.0 + np.exp(-cls_part))

    ang_min, ang_max = float(angle_part.min()), float(angle_part.max())
    if ang_min >= -np.pi - 1e-6 and ang_max <= np.pi + 1e-6:
        angles_rad = angle_part.copy()
    else:
        if ang_min >= 0.0 and ang_max <= 1.0:
            angles_rad = (angle_part - 0.25) * np.pi
        else:
            # fallback: treat as radians anyway
            angles_rad = angle_part.copy()

    cls_ids = np.argmax(cls_probs, axis=1)
    cls_scores = cls_probs[np.arange(len(cls_probs)), cls_ids]
    mask = cls_scores > scores_th
    if mask.sum() == 0:
        return [], [], []

    boxes = boxes_raw[mask]
    scores = cls_scores[mask]
    ids = cls_ids[mask]
    angs = angles_rad[mask]

    rects = []
    for i in range(len(boxes)):
        cx, cy, w, h = boxes[i].astype(float)
        a_deg = float(angs[i] * 180.0 / np.pi)
        # ensure positive sizes
        w = max(w, 1.0)
        h = max(h, 1.0)
        rects.append(((cx, cy), (w, h), a_deg))
    
    # denormalize boxes
    for i in range(len(rects)):
        (cx, cy), (w, h), ang = rects[i]
        cx_u = (cx - dw) / scale
        cy_u = (cy - dh) / scale
        w_u = w / scale
        h_u = h / scale
        rects[i] = (((cx_u, cy_u), (w_u, h_u), ang))

    return rects, scores.tolist(), ids.tolist()

def rotated_rect_to_aabox(cx, cy, w, h, angle_deg):
    ang = np.deg2rad(angle_deg)
    dx = w / 2.0
    dy = h / 2.0
    corners = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=np.float32)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], dtype=np.float32)
    rotated = (corners @ R.T) + np.array([cx, cy], dtype=np.float32)
    xmin = float(np.min(rotated[:, 0]))
    ymin = float(np.min(rotated[:, 1]))
    xmax = float(np.max(rotated[:, 0]))
    ymax = float(np.max(rotated[:, 1]))
    return [xmin, ymin, xmax, ymax]


def nms_boxes(boxes: List[List[float]], scores: List[float], iou_thresh: float):
    if not boxes:
        return []
    idxs = np.argsort(scores)[::-1]
    keep = []
    boxes_arr = np.array(boxes, dtype=np.float32)
    for i in idxs:
        box = boxes_arr[i]
        keep_flag = True
        for j in keep:
            if compute_iou(box.tolist(), boxes_arr[j].tolist()) >= iou_thresh:
                keep_flag = False
                break
        if keep_flag:
            keep.append(i)
    return keep


def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(1e-5, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1e-5, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return inter / (areaA + areaB - inter + 1e-5)


def rotated_iou(rect1, rect2) -> float:
    inter_type, inter = cv2.rotatedRectangleIntersection(rect1, rect2)
    if inter is None:
        return 0.0
    # compute polygon area
    try:
        inter_area = abs(cv2.contourArea(inter))
    except Exception:
        return 0.0
    area1 = rect1[1][0] * rect1[1][1]
    area2 = rect2[1][0] * rect2[1][1]
    if area1 + area2 - inter_area <= 0:
        return 0.0
    
    return inter_area / (area1 + area2 - inter_area + 1e-9)


def rotated_nms(rects: List[Tuple], scores: List[float], iou_thresh=0.6):
    if len(rects) == 0:
        return []
    idxs = np.argsort(scores)[::-1].tolist()
    keep = []
    while idxs:
        i = idxs.pop(0)
        keep.append(i)
        remove = []
        for j in idxs:
            iou = rotated_iou(rects[i], rects[j])
            if iou > iou_thresh:
                remove.append(j)
        idxs = [x for x in idxs if x not in remove]
    
    return keep


def draw_detection_local(image, boxes, scores, classes, labels=None):
    for box, score, cid in zip(boxes, scores, classes):
        pts = cv2.boxPoints(box)
        pts = pts.astype(np.int32)
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # label at top-left corner of polygon bbox
        tl = tuple(pts.min(axis=0))
        if labels and cid < len(labels):
            label_text = f"{labels[cid]} {score:.2f}"
        else:
            label_text = f"C{cid} {score:.2f}"
        cv2.putText(image, label_text, (int(tl[0]), int(tl[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)


def draw_detections_local(detections: dict, img_out: np.ndarray, labels, tracker=None):
    boxes = detections["detection_boxes"]
    scores = detections["detection_scores"]
    classes = detections["detection_classes"]
    num_detections = detections["num_detections"]

    if tracker:
        dets_for_tracker = []
        for idx in range(num_detections):
            box = boxes[idx]
            score = scores[idx]
            dets_for_tracker.append([*box, score])
        if not dets_for_tracker:
            return img_out
        online_targets = tracker.update(np.array(dets_for_tracker))
        for track in online_targets:
            track_id = track.track_id
            x1, y1, x2, y2 = track.tlbr
            xmin, ymin, xmax, ymax = map(int, [x1, y1, x2, y2])
            best_idx = find_best_matching_detection_index(track.tlbr, boxes)
            color = tuple(id_to_color(classes[best_idx]).tolist()) if best_idx is not None else (0, 255, 0)
            if best_idx is None:
                draw_detection_local(img_out, [xmin, ymin, xmax, ymax], f"ID {track_id}", track.score * 100.0, color, track=True)
            else:
                draw_detection_local(img_out, [xmin, ymin, xmax, ymax], [labels[classes[best_idx]], f"ID {track_id}"], track.score * 100.0, color, track=True)
    else:
        draw_detection_local(img_out, boxes, scores, classes, labels)

    return img_out


def find_best_matching_detection_index(track_box, detection_boxes):
    best_iou = 0
    best_idx = -1
    for i, det_box in enumerate(detection_boxes):
        iou = compute_iou(track_box, det_box)
        if iou > best_iou:
            best_iou = iou
            best_idx = i
    return best_idx if best_idx != -1 else None
