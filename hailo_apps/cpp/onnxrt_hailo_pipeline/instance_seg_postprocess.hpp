#pragma once
/**
 * Copyright 2020 (C) Hailo Technologies Ltd.
 * All rights reserved.
 *
 * Hailo Technologies Ltd. ("Hailo") disclaims any warranties, including, but not limited to,
 * the implied warranties of merchantability and fitness for a particular purpose.
 * This software is provided on an "AS IS" basis, and Hailo has no obligation to provide maintenance,
 * support, updates, enhancements, or modifications.
 *
 * You may use this software in the development of any project.
 * You shall not reproduce, modify or distribute this software without prior written permission.
 **/

 #include <vector>
 #include <string>
 #include <utility>
 #include <opencv2/opencv.hpp>
 #include <xtensor/xarray.hpp>
 #include <xtensor/xadapt.hpp>
 
 #include "../common/general/hailo_objects.hpp"
 #include "../common/labels/coco_eighty.hpp"
 

struct DetectionAndMask {
    HailoDetection detection;
    cv::Mat        mask; // CV_32FC1 in frame size
};


cv::Mat pad_frame_letterbox(const cv::Mat &frame, int model_h, int model_w);

HailoROIPtr build_roi_from_outputs(const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &outputs);


std::vector<std::pair<HailoDetection, xt::xarray<float>>>
nms_pairs(std::vector<std::pair<HailoDetection, xt::xarray<float>>> dets,
          float iou_thr,
          bool cross_class = true);

// ------------- Mask composition (coeff * proto) ----------------
/**
 * Compose final masks from mask coefficients and proto feature map.
 * Outputs per detection a mask (org_h x org_w).
 */
std::vector<DetectionAndMask>
decode_masks(const std::vector<std::pair<HailoDetection, xt::xarray<float>>> &kept,
             const xt::xarray<float> &proto,
             int org_h, int org_w,
             int model_h, int model_w,
             int proto_channels = 32);

void draw_masks_and_boxes(cv::Mat &frame,
                          const std::vector<HailoDetection> &dets,
                          const std::vector<cv::Mat> &masks,
                          float alpha = 0.7f,
                          float thresh = 0.5f);           