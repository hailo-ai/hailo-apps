import cv2
import numpy as np
try:
    from hailo_apps.python.core.common.toolbox import id_to_color
except ImportError:
    from pathlib import Path
    import sys

    core_dir = Path(__file__).resolve().parents[2] / "core"
    sys.path.insert(0, str(core_dir))
    from common.toolbox import id_to_color

import os
from collections import deque

# Dictionary to store a limited history of tracklet coordinates.
# The keys will be the track IDs.
tracklet_history = {}
# Maximum number of past frames to display
trail_length = 30 
# Only draw trail for certain classes (e.g., person=0, phone=67 in COCO)
TRACKLET_CLASSES = [0, 67]  # PERSON, SMARTPHONE

def inference_result_handler(original_frame, infer_results, labels, config_data, tracker=None, draw_trail=False):
    """
    Processes inference results and draw detections (with optional tracking).

    Args:
        infer_results (list): Raw output from the model.
        original_frame (np.ndarray): Original image frame.
        labels (list): List of class labels.
        enable_tracking (bool): Whether tracking is enabled.
        tracker (BYTETracker, optional): ByteTrack tracker instance.

    Returns:
        np.ndarray: Frame with detections or tracks drawn.
    """
    detections = extract_detections(original_frame, infer_results, config_data)  # Should return dict with boxes, classes, scores
    frame_with_detections = draw_detections(detections, original_frame, labels, tracker=tracker, draw_trail=draw_trail)
    return frame_with_detections


def draw_detection(image: np.ndarray, box: list, labels: list, score: float, color: tuple, track=False):
    """
    Draw box and label for one detection.

    Args:
        image (np.ndarray): Image to draw on.
        box (list): Bounding box coordinates.
        labels (list): List of labels (1 or 2 elements).
        score (float): Detection score.
        color (tuple): Color for the bounding box.
        track (bool): Whether to include tracking info.
    """
    xmin, ymin, xmax, ymax = map(int, box)
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Compose texts
    top_text = f"{labels[0]}: {score:.1f}%" if not track or len(labels) == 2 else f"{score:.1f}%"
    bottom_text = None

    if track:
        if len(labels) == 2:
            bottom_text = labels[1]
        else:
            bottom_text = labels[0]


    # Set colors
    text_color = (255, 255, 255)  # White
    border_color = (0, 0, 0)  # Black

    # Draw top text with black border first
    cv2.putText(image, top_text, (xmin + 4, ymin + 20), font, 0.5, border_color, 2, cv2.LINE_AA)
    cv2.putText(image, top_text, (xmin + 4, ymin + 20), font, 0.5, text_color, 1, cv2.LINE_AA)

    # Draw bottom text if exists
    if bottom_text:
        pos = (xmax - 50, ymax - 6)
        cv2.putText(image, bottom_text, pos, font, 0.5, border_color, 2, cv2.LINE_AA)
        cv2.putText(image, bottom_text, pos, font, 0.5, text_color, 1, cv2.LINE_AA)


def denormalize_and_rm_pad(box: list, size: int, padding_length: int, input_height: int, input_width: int) -> list:
    """
    Denormalize bounding box coordinates and remove padding.

    Args:
        box (list): Normalized bounding box coordinates.
        size (int): Size to scale the coordinates.
        padding_length (int): Length of padding to remove.
        input_height (int): Height of the input image.
        input_width (int): Width of the input image.

    Returns:
        list: Denormalized bounding box coordinates with padding removed.
    """
    # Scale box coordinates
    box = [int(x * size) for x in box]

    # Apply padding correction
    for i in range(4):
        if i % 2 == 0:  # x-coordinates
            if input_height != size:
                box[i] -= padding_length
        else:  # y-coordinates
            if input_width != size:
                box[i] -= padding_length

    # Swap to [ymin, xmin, ymax, xmax]
    return [box[1], box[0], box[3], box[2]]


def extract_detections(image: np.ndarray, detections: list, config_data) -> dict:
    """
    Extract detections from the input data.

    Args:
        image (np.ndarray): Image to draw on.
        detections (list): Raw detections from the model.
        config_data (Dict): Loaded JSON config containing post-processing metadata.

    Returns:
        dict: Filtered detection results containing 'detection_boxes', 'detection_classes', 'detection_scores', and 'num_detections'.
    """

    visualization_params = config_data["visualization_params"]
    score_threshold = visualization_params.get("score_thres", 0.5)
    max_boxes = visualization_params.get("max_boxes_to_draw", 50)

    img_height, img_width = image.shape[:2]
    size = max(img_height, img_width)
    padding_length = int(abs(img_height - img_width) / 2)

    all_detections = []

    for class_id, detection in enumerate(detections):
        for det in detection:
            bbox, score = det[:4], det[4]
            if score >= score_threshold:
                denorm_bbox = denormalize_and_rm_pad(bbox, size, padding_length, img_height, img_width)
                all_detections.append((score, class_id, denorm_bbox))

    # Sort all detections by score descending
    all_detections.sort(reverse=True, key=lambda x: x[0])

    # Take top max_boxes
    top_detections = all_detections[:max_boxes]

    scores, class_ids, boxes = zip(*top_detections) if top_detections else ([], [], [])

    return {
        'detection_boxes': list(boxes),
        'detection_classes': list(class_ids),
        'detection_scores': list(scores),
        'num_detections': len(top_detections)
    }


def draw_detections(detections: dict, img_out: np.ndarray, labels, tracker=None, draw_trail=False) -> np.ndarray:
    """
    Draw detections or tracking results on the image.

    Args:
        detections (dict): Raw detection outputs.
        img_out (np.ndarray): Image to draw on.
        labels (list): List of class labels.
        enable_tracking (bool): Whether to use tracker output (ByteTrack).
        tracker (BYTETracker, optional): ByteTrack tracker instance.

    Returns:
        np.ndarray: Annotated image.
    """

    # Extract detection data from the dictionary
    boxes = detections["detection_boxes"]  # List of [xmin,ymin,xmaxm, ymax] boxes
    scores = detections["detection_scores"]  # List of detection confidences
    num_detections = detections["num_detections"]  # Total number of valid detections
    classes = detections["detection_classes"]  # List of class indices per detection

    if tracker:
        dets_for_tracker = []

        # Convert detection format to [xmin,ymin,xmaxm ymax,score] for tracker
        for idx in range(num_detections):
            box = boxes[idx]  # [x, y, w, h]
            score = scores[idx]
            dets_for_tracker.append([*box, score])

        # Skip tracking if no detections passed
        if not dets_for_tracker:
            return img_out

        # Run BYTETracker and get active tracks
        online_targets = tracker.update(np.array(dets_for_tracker))

        # Draw tracked bounding boxes with ID labels
        for track in online_targets:
            track_id = track.track_id  # Unique tracker ID
            x1, y1, x2, y2 = track.tlbr  # Bounding box (top-left, bottom-right)
            xmin, ymin, xmax, ymax = map(int, [x1, y1, x2, y2])
            best_idx = find_best_matching_detection_index(track.tlbr, boxes)
            color = tuple(id_to_color(classes[best_idx]).tolist())  # Color based on class
            if best_idx is None:
                draw_detection(img_out, [xmin, ymin, xmax, ymax], f"ID {track_id}",
                               track.score * 100.0, color, track=True)
            else:
                draw_detection(img_out, [xmin, ymin, xmax, ymax], [labels[classes[best_idx]], f"ID {track_id}"],
                               track.score * 100.0, color, track=True)
                               
            if not classes[best_idx] in TRACKLET_CLASSES:
                continue

            # Get the centroid of the current bounding box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            centroid = (center_x, center_y)
            
            # Initialize or update the tracklet history
            if track_id not in tracklet_history:
                tracklet_history[track_id] = deque(maxlen=trail_length)
            tracklet_history[track_id].append(centroid)

            if draw_trail:
                for i in range(1, len(tracklet_history[track_id])):
                    # Get the center point for the current and previous frames
                    point_a = tracklet_history[track_id][i-1]
                    point_b = tracklet_history[track_id][i]

                    # Draw a line between the points and draw the points as circles
                    cv2.line(img_out, point_a, point_b, color, 3) #(255, 0, 0), 2)
                    cv2.circle(img_out, point_b, radius=20, thickness=1, color=color) #, thickness=-1) # -1 for filled circle



    else:
        # No tracking — draw raw model detections
        for idx in range(num_detections):
            color = tuple(id_to_color(classes[idx]).tolist())  # Color based on class
            draw_detection(img_out, boxes[idx], [labels[classes[idx]]], scores[idx] * 100.0, color)

    return img_out


def find_best_matching_detection_index(track_box, detection_boxes):
    """
    Finds the index of the detection box with the highest IoU relative to the given tracking box.

    Args:
        track_box (list or tuple): The tracking box in [x_min, y_min, x_max, y_max] format.
        detection_boxes (list): List of detection boxes in [x_min, y_min, x_max, y_max] format.

    Returns:
        int or None: Index of the best matching detection, or None if no match is found.
    """
    best_iou = 0
    best_idx = -1

    for i, det_box in enumerate(detection_boxes):
        iou = compute_iou(track_box, det_box)
        if iou > best_iou:
            best_iou = iou
            best_idx = i

    return best_idx if best_idx != -1 else None


def compute_iou(boxA, boxB):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    IoU measures the overlap between two boxes:
        IoU = (area of intersection) / (area of union)
    Values range from 0 (no overlap) to 1 (perfect overlap).

    Args:
        boxA (list or tuple): [x_min, y_min, x_max, y_max]
        boxB (list or tuple): [x_min, y_min, x_max, y_max]

    Returns:
        float: IoU value between 0 and 1.
    """
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(1e-5, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1e-5, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return inter / (areaA + areaB - inter + 1e-5)
