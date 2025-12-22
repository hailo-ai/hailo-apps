/**
 * Copyright (c) 2020-2025 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/
/**
 * @file obb_utils.hpp
 * Oriented Bounding Box (OBB) detection utilities for YOLO11 OBB
 **/

#ifndef _OBB_UTILS_HPP_
#define _OBB_UTILS_HPP_

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include "hailo/hailort.h"
#include "toolbox.hpp"
#include <opencv2/opencv.hpp>

using namespace hailo_utils;

// ─────────────────────────────────────────────────────────────────────────────
// STRUCTURES
// ─────────────────────────────────────────────────────────────────────────────

struct OBBDetection {
    cv::RotatedRect box;  // (center, size, angle in degrees)
    size_t class_id;
    float score;
};

// ─────────────────────────────────────────────────────────────────────────────
// POSTPROCESSING FUNCTIONS
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Main OBB postprocessing function - emulates ultralytics implementation.
 * Processes raw model outputs through DFL decoding and anchor generation.
 * 
 * @param output_data_and_infos Raw model outputs (9 tensors: 3 scales x 3 heads)
 * @param img_size Model input size (640)
 * @param cls_num Number of classes (15 for DOTAv1)
 * @return cv::Mat of shape (1, 20, 8400) containing [cx,cy,w,h, cls_scores..., angle]
 */
cv::Mat obb_postprocess(
    const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>>& output_data_and_infos,
    int img_size = 640,
    int cls_num = 15
);

/**
 * Extract OBB detections from postprocessed output - mirrors extract_obb_detections
 * Applies thresholding, denormalization, and padding removal
 * 
 * @param postprocess_output Output from obb_postprocess (1, 20, 8400)
 * @param orig_height Original image height
 * @param orig_width Original image width
 * @param cls_num Number of classes
 * @param img_size Model input size
 * @param score_threshold Detection confidence threshold
 * @return Vector of OBBDetection objects in original image coordinates
 */
std::vector<OBBDetection> extract_obb_detections(
    const cv::Mat& postprocess_output,
    int orig_height,
    int orig_width,
    int cls_num,
    int img_size = 640,
    float score_threshold = 0.35f
);

/**
 * Rotated NMS - mirrors rotated_nms
 * Non-maximum suppression for oriented bounding boxes
 * 
 * @param detections Input detections
 * @param iou_threshold IoU threshold for suppression
 * @return Vector of detection indices to keep
 */
std::vector<size_t> rotated_nms(
    const std::vector<OBBDetection>& detections,
    float iou_threshold = 0.6f
);

/**
 * Calculate IoU between two rotated rectangles
 * Uses cv::rotatedRectangleIntersection
 * 
 * @param rect1 First rotated rectangle
 * @param rect2 Second rotated rectangle
 * @return IoU value [0, 1]
 */
float rotated_iou(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2);

// ─────────────────────────────────────────────────────────────────────────────
// VISUALIZATION
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Draw oriented bounding boxes on frame
 * 
 * @param frame Image to draw on
 * @param detections Vector of OBB detections
 */
void draw_obb_detections(cv::Mat& frame, const std::vector<OBBDetection>& detections);

/**
 * Draw single oriented bounding box
 * 
 * @param frame Image to draw on
 * @param detection Single OBB detection
 * @param color Box color
 */
void draw_single_obb(cv::Mat& frame, const OBBDetection& detection, const cv::Scalar& color);

// ─────────────────────────────────────────────────────────────────────────────
// HELPER FUNCTIONS (Internal DFL/Anchor Generation)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Softmax activation along specified axis
 */
cv::Mat softmax(const cv::Mat& input, int axis = -1);

/**
 * Sigmoid activation
 */
cv::Mat sigmoid(const cv::Mat& input);

/**
 * Decode DFL (Distribution Focal Loss) predictions
 * 
 * @param dfl_input Input tensor (B, 4, reg_max, N)
 * @param reg_max Number of DFL bins (16)
 * @return Decoded distances (B, 4, N)
 */
cv::Mat decode_dfl(const cv::Mat& dfl_input, int reg_max = 16);

/**
 * Generate anchor grids for all scales
 * 
 * @param img_size Model input size (640)
 * @param strides Feature map strides {8, 16, 32}
 * @return Anchor coordinates (2, 8400) [X, Y]
 */
cv::Mat generate_anchors(int img_size, const std::vector<int>& strides);

/**
 * Get class name from integer ID
 */
std::string get_dota_class_name(int class_id);

/**
 * Initialize class colors
 */
void initialize_obb_class_colors(std::unordered_map<int, cv::Scalar>& class_colors);

#endif // _OBB_UTILS_HPP_
