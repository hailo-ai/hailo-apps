/**
 * Copyright (c) 2020-2025 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/
#include "obb_utils.hpp"
#include "../../common/labels/dota_fifteen.hpp"
#include <numeric>
#include <cfloat>
#include <cmath>

// ─────────────────────────────────────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────────────────────────────────────

const std::vector<cv::Scalar> OBB_COLORS = {
    cv::Scalar(255,   0,   0),  // Red
    cv::Scalar(  0, 255,   0),  // Green
    cv::Scalar(  0,   0, 255),  // Blue
    cv::Scalar(255, 255,   0),  // Cyan
    cv::Scalar(255,   0, 255),  // Magenta
    cv::Scalar(  0, 255, 255),  // Yellow
    cv::Scalar(255, 128,   0),  // Orange
    cv::Scalar(128,   0, 128),  // Purple
    cv::Scalar(128, 128,   0),  // Olive
    cv::Scalar(128,   0, 255),  // Violet
    cv::Scalar(  0, 128, 255),  // Sky Blue
    cv::Scalar(255,   0, 128),  // Pink
    cv::Scalar(  0, 128,   0),  // Dark Green
    cv::Scalar(128, 128, 128),  // Gray
    cv::Scalar(255, 255, 255)   // White
};

// ─────────────────────────────────────────────────────────────────────────────
// HELPER FUNCTIONS
// ─────────────────────────────────────────────────────────────────────────────

// Note: sigmoid, softmax, and decode_dfl are now inlined in obb_postprocess for efficiency

cv::Mat generate_anchors(int img_size, const std::vector<int>& strides) {
    // Generate anchor grids for each scale and concatenate
    // Output: (2, total_anchors) where row 0 = X coords, row 1 = Y coords
    
    int total_anchors = 0;
    for (int stride : strides) {
        int fm_size = img_size / stride;
        total_anchors += fm_size * fm_size;
    }
    
    cv::Mat anchors(2, total_anchors, CV_32F);
    int anchor_idx = 0;
    
    for (int stride : strides) {
        int fm_size = img_size / stride;
        
        // Generate grid coordinates (row-major order: iterate x within y)
        for (int y = 0; y < fm_size; ++y) {
            for (int x = 0; x < fm_size; ++x) {
                anchors.at<float>(0, anchor_idx) = x + 0.5f;  // X coord
                anchors.at<float>(1, anchor_idx) = y + 0.5f;  // Y coord
                anchor_idx++;
            }
        }
    }
    
    return anchors;
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN POSTPROCESSING FUNCTION
// ─────────────────────────────────────────────────────────────────────────────

cv::Mat obb_postprocess(
    const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>>& output_data_and_infos,
    int img_size,
    int cls_num
) {
    const int reg_max = 16;
    const std::vector<int> strides = {8, 16, 32};
    const int total_anchors = 8400;  // 80*80 + 40*40 + 20*20
    
    // Output: (4 + cls_num + 1, 8400) = (20, 8400)
    cv::Mat output(4 + cls_num + 1, total_anchors, CV_32F, cv::Scalar(0));
    
    // Generate anchor grids once
    cv::Mat anchors = generate_anchors(img_size, strides);  // (2, 8400)
    
    // Process each scale and accumulate results
    // Outputs come from HailoRT in alphabetical order of layer names:
    // conv54(angle0), conv68(angle1), conv86(angle2), conv53(bbox0), conv57(class0), conv67(bbox1), conv71(class1), conv85(bbox2), conv89(class2)
    // We need to map: scale_idx=0 -> angle[0], bbox[3], class[4]
    //                 scale_idx=1 -> angle[1], bbox[5], class[6]  
    //                 scale_idx=2 -> angle[2], bbox[7], class[8]
    int anchor_offset = 0;
    for (int scale_idx = 0; scale_idx < 3; ++scale_idx) {
        int angle_idx = scale_idx;                  // 0, 1, 2
        int bbox_idx = 3 + scale_idx * 2;           // 3, 5, 7
        int class_idx = 4 + scale_idx * 2;          // 4, 6, 8
        
        auto& angle_info = output_data_and_infos[angle_idx];
        auto& bbox_info = output_data_and_infos[bbox_idx];
        auto& cls_info = output_data_and_infos[class_idx];
        
        int H = bbox_info.second.shape.height;
        int W = bbox_info.second.shape.width;
        int num_anchors = H * W;
        int stride = strides[scale_idx];
        
        // Get data pointers - should be float32 now
        const float* bbox_data_f32 = reinterpret_cast<const float*>(bbox_info.first);
        const float* cls_data_f32 = reinterpret_cast<const float*>(cls_info.first);
        const float* angle_data_f32 = reinterpret_cast<const float*>(angle_info.first);
        
        // Process each anchor in this scale
        for (int spatial_idx = 0; spatial_idx < num_anchors; ++spatial_idx) {
            int anchor_idx = anchor_offset + spatial_idx;
            
            if (anchor_idx >= total_anchors) {
                break;
            }
            
            // 1. Angle: sigmoid and convert to radians (already dequantized to float32)
            float angle_raw = angle_data_f32[spatial_idx];
            float angle_sigmoid = 1.0f / (1.0f + std::exp(-angle_raw));
            float angle_rad = (angle_sigmoid - 0.25f) * M_PI;
            float angle_sin = std::sin(angle_rad);
            float angle_cos = std::cos(angle_rad);
            
            // 2. DFL decode bbox (already dequantized to float32)
            const int bbox_base = spatial_idx * 64;
            float dfl_decoded[4];
            
            for (int coord = 0; coord < 4; ++coord) {
                const int coord_base = bbox_base + coord * reg_max;
                
                // Find max and compute softmax in one pass
                float max_val = -FLT_MAX;
                for (int bin = 0; bin < reg_max; ++bin) {
                    float val = bbox_data_f32[coord_base + bin];
                    max_val = std::max(max_val, val);
                }
                
                // Softmax
                float sum_exp = 0.0f;
                float weighted_sum = 0.0f;
                for (int bin = 0; bin < reg_max; ++bin) {
                    float val = bbox_data_f32[coord_base + bin];
                    float exp_val = std::exp(val - max_val);
                    sum_exp += exp_val;
                    weighted_sum += exp_val * bin;
                }
                dfl_decoded[coord] = weighted_sum / sum_exp;
            }
            
            // 3. Convert DFL to rotated box coordinates
            float dfl_div_left = (dfl_decoded[2] - dfl_decoded[0]) / 2.0f;
            float dfl_div_right = (dfl_decoded[3] - dfl_decoded[1]) / 2.0f;
            
            float offset_x = dfl_div_left * angle_cos - dfl_div_right * angle_sin;
            float offset_y = dfl_div_left * angle_sin + dfl_div_right * angle_cos;
            
            // Get anchor position and add offset
            float anchor_x = anchors.at<float>(0, anchor_idx);
            float anchor_y = anchors.at<float>(1, anchor_idx);
            
            float cx_grid = anchor_x + offset_x;
            float cy_grid = anchor_y + offset_y;
            float w_grid = dfl_decoded[0] + dfl_decoded[2];
            float h_grid = dfl_decoded[1] + dfl_decoded[3];
            
            // Scale to pixel coordinates
            float cx = cx_grid * stride;
            float cy = cy_grid * stride;
            float w = w_grid * stride;
            float h = h_grid * stride;
            
            // 4. Class scores with sigmoid (already dequantized to float32)
            const int cls_base = spatial_idx * cls_num;
            for (int c = 0; c < cls_num; ++c) {
                float raw_score = cls_data_f32[cls_base + c];
                float score = 1.0f / (1.0f + std::exp(-raw_score));
                output.at<float>(4 + c, anchor_idx) = score;
            }
            
            // 5. Write bbox and angle to output
            output.at<float>(0, anchor_idx) = cx;
            output.at<float>(1, anchor_idx) = cy;
            output.at<float>(2, anchor_idx) = w;
            output.at<float>(3, anchor_idx) = h;
            output.at<float>(4 + cls_num, anchor_idx) = angle_rad;
        }
        
        anchor_offset += num_anchors;
    }
    
    // Reshape to (1, 20, 8400)
    cv::Mat reshaped = output.reshape(1, std::vector<int>{1, 4 + cls_num + 1, total_anchors});
    return reshaped;
}

// ─────────────────────────────────────────────────────────────────────────────
// EXTRACTION FUNCTION
// ─────────────────────────────────────────────────────────────────────────────

std::vector<OBBDetection> extract_obb_detections(
    const cv::Mat& postprocess_output,
    int orig_height,
    int orig_width,
    int cls_num,
    int img_size,
    float score_threshold
) {
    // Transpose from (1, C, N) to (N, C) format
    // postprocess_output shape: (1, 20, 8400) -> transpose to (8400, 20)
    
    int total_anchors = postprocess_output.size[2];
    int num_channels = postprocess_output.size[1];
    
    cv::Mat preds(total_anchors, num_channels, CV_32F);
    for (int n = 0; n < total_anchors; ++n) {
        for (int c = 0; c < num_channels; ++c) {
            preds.at<float>(n, c) = postprocess_output.at<float>(0, c, n);
        }
    }
    
    // Calculate letterbox parameters
    float scale = std::min(static_cast<float>(img_size) / orig_width, 
                           static_cast<float>(img_size) / orig_height);
    int new_unpad_w = static_cast<int>(std::round(orig_width * scale));
    int new_unpad_h = static_cast<int>(std::round(orig_height * scale));
    float dw = (img_size - new_unpad_w) / 2.0f;
    float dh = (img_size - new_unpad_h) / 2.0f;
    
    // Extract components (parallel to Python implementation)
    int idx_cls_start = 4;
    int idx_cls_end = 4 + cls_num;
    int idx_angle = 4 + cls_num;
    
    // Check ranges to determine if normalization/scaling is needed (like Python lines 200-217)
    float box_max = 0.0f;
    float cls_min = FLT_MAX, cls_max = -FLT_MAX;
    float ang_min = FLT_MAX, ang_max = -FLT_MAX;
    
    for (int n = 0; n < std::min(total_anchors, 100); ++n) {  // Sample first 100 for efficiency
        box_max = std::max(box_max, std::abs(preds.at<float>(n, 0)));
        box_max = std::max(box_max, std::abs(preds.at<float>(n, 1)));
        box_max = std::max(box_max, std::abs(preds.at<float>(n, 2)));
        box_max = std::max(box_max, std::abs(preds.at<float>(n, 3)));
        
        for (int c = idx_cls_start; c < idx_cls_end; ++c) {
            float val = preds.at<float>(n, c);
            cls_min = std::min(cls_min, val);
            cls_max = std::max(cls_max, val);
        }
        
        float ang = preds.at<float>(n, idx_angle);
        ang_min = std::min(ang_min, ang);
        ang_max = std::max(ang_max, ang);
    }
    
    bool boxes_need_scaling = (box_max <= 1.0f + 1e-6f);
    bool cls_need_sigmoid = !(cls_min >= 0.0f && cls_max <= 1.0f);
    bool ang_is_normalized = (ang_min >= 0.0f && ang_max <= 1.0f);
    bool ang_in_range = (ang_min >= -M_PI - 1e-6f && ang_max <= M_PI + 1e-6f);
    
    std::vector<OBBDetection> detections;
    
    for (int n = 0; n < total_anchors; ++n) {
        // Extract box
        float cx = preds.at<float>(n, 0);
        float cy = preds.at<float>(n, 1);
        float w = preds.at<float>(n, 2);
        float h = preds.at<float>(n, 3);
        
        // Scale boxes if needed
        if (boxes_need_scaling) {
            cx *= img_size;
            cy *= img_size;
            w *= img_size;
            h *= img_size;
        }
        
        // Find best class and apply sigmoid if needed
        float max_score = 0.0f;
        int max_class = 0;
        for (int c = idx_cls_start; c < idx_cls_end; ++c) {
            float score = preds.at<float>(n, c);
            if (cls_need_sigmoid) {
                score = 1.0f / (1.0f + std::exp(-score));  // sigmoid
            }
            if (score > max_score) {
                max_score = score;
                max_class = c - idx_cls_start;
            }
        }
        
        // Threshold check
        if (max_score <= score_threshold) {
            continue;
        }
        
        // Denormalize coordinates (remove padding and scale) - do this before angle handling
        float cx_denorm = (cx - dw) / scale;
        float cy_denorm = (cy - dh) / scale;
        float w_denorm = w / scale;
        float h_denorm = h / scale;
        
        // Ensure positive sizes
        w_denorm = std::max(w_denorm, 1.0f);
        h_denorm = std::max(h_denorm, 1.0f);
        
        // Extract and convert angle (matching Python implementation exactly)
        float angle_rad = preds.at<float>(n, idx_angle);
        if (ang_is_normalized) {
            // Normalized [0,1] -> [-pi/4, 3pi/4] like Python line 214
            angle_rad = (angle_rad - 0.25f) * M_PI;
        } else if (!ang_in_range) {
            // Already in radians or unknown - keep as is (fallback)
        }
        
        // Convert to degrees - NO angle convention adjustments
        // Python simply converts rad->deg without any normalization
        float angle_deg = angle_rad * 180.0f / M_PI;
        
        // Use denormalized sizes directly - NO swapping
        float w_final = w_denorm;
        float h_final = h_denorm;
        float angle_final = angle_deg;
        
        // Create detection
        OBBDetection det;
        det.box = cv::RotatedRect(
            cv::Point2f(cx_denorm, cy_denorm),
            cv::Size2f(w_final, h_final),
            angle_final
        );
        det.class_id = max_class;
        det.score = max_score;
        
        detections.push_back(det);
    }
    
    return detections;
}

// ─────────────────────────────────────────────────────────────────────────────
// ROTATED NMS
// ─────────────────────────────────────────────────────────────────────────────

float rotated_iou(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2) {
    std::vector<cv::Point2f> intersection;
    int result = cv::rotatedRectangleIntersection(rect1, rect2, intersection);
    
    if (result == cv::INTERSECT_NONE || intersection.empty()) {
        return 0.0f;
    }
    
    // Calculate intersection area
    float inter_area = std::abs(cv::contourArea(intersection));
    
    // Calculate union
    float area1 = rect1.size.width * rect1.size.height;
    float area2 = rect2.size.width * rect2.size.height;
    float union_area = area1 + area2 - inter_area;
    
    if (union_area <= 0.0f) {
        return 0.0f;
    }
    
    return inter_area / union_area;
}

std::vector<size_t> rotated_nms(
    const std::vector<OBBDetection>& detections,
    float iou_threshold
) {
    if (detections.empty()) {
        return {};
    }
    
    // Sort indices by score (descending)
    std::vector<size_t> indices(detections.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), 
        [&detections](size_t i1, size_t i2) {
            return detections[i1].score > detections[i2].score;
        });
    
    std::vector<size_t> keep;
    std::vector<bool> suppressed(detections.size(), false);
    
    for (size_t idx : indices) {
        if (suppressed[idx]) {
            continue;
        }
        
        keep.push_back(idx);
        
        // Suppress overlapping boxes
        for (size_t j : indices) {
            if (suppressed[j] || j == idx) {
                continue;
            }
            
            float iou = rotated_iou(detections[idx].box, detections[j].box);
            if (iou > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }
    
    return keep;
}

// ─────────────────────────────────────────────────────────────────────────────
// VISUALIZATION
// ─────────────────────────────────────────────────────────────────────────────

void initialize_obb_class_colors(std::unordered_map<int, cv::Scalar>& class_colors) {
    for (int cls = 0; cls < 15; ++cls) {
        class_colors[cls] = OBB_COLORS[cls % OBB_COLORS.size()];
    }
}

std::string get_dota_class_name(int class_id) {
    auto it = common::dota_fifteen.find(static_cast<uint8_t>(class_id));
    if (it != common::dota_fifteen.end()) {
        return it->second;
    }
    return "unknown";
}

void draw_single_obb(cv::Mat& frame, const OBBDetection& detection, const cv::Scalar& color) {
    // Get corner points
    cv::Point2f vertices[4];
    detection.box.points(vertices);
    
    // Convert to integer points
    std::vector<cv::Point> pts;
    for (int i = 0; i < 4; ++i) {
        pts.push_back(cv::Point(static_cast<int>(vertices[i].x), 
                                static_cast<int>(vertices[i].y)));
    }
    
    // Draw polygon
    cv::polylines(frame, pts, true, color, 2);
    
    // Draw label
    std::string class_name = get_dota_class_name(detection.class_id);
    std::string label = class_name + " " + 
                       std::to_string(static_cast<int>(detection.score * 100)) + "%";
    
    // Find top-left point for label placement
    cv::Point tl = pts[0];
    for (const auto& pt : pts) {
        if (pt.y < tl.y || (pt.y == tl.y && pt.x < tl.x)) {
            tl = pt;
        }
    }
    
    cv::putText(frame, label, cv::Point(tl.x, tl.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
}

void draw_obb_detections(cv::Mat& frame, const std::vector<OBBDetection>& detections) {
    std::unordered_map<int, cv::Scalar> class_colors;
    initialize_obb_class_colors(class_colors);
    
    for (const auto& detection : detections) {
        const auto& color = class_colors[detection.class_id];
        draw_single_obb(frame, detection, color);
    }
}
