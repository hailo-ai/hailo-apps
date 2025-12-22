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

/**
 * @file yolov5seg_postprocess.hpp
 * YOLOv5 Segmentation post-processing utilities
 **/

#ifndef _YOLOV5SEG_POSTPROCESS_H_
#define _YOLOV5SEG_POSTPROCESS_H_

#include <iostream>
#include <string>
#include <vector>
#include "hailo/hailort.h"
#include "toolbox.hpp"
#pragma once
#include <opencv2/opencv.hpp>
#include "general/hailo_objects.hpp"
#include "general/hailo_tensors.hpp"
#include "general/hailo_common.hpp"

#include <xtensor/views/xview.hpp>
#include <xtensor/misc/xsort.hpp>

using namespace hailo_utils;
// --------------------------- FUNCTION DECLARATIONS ---------------------------

static const std::vector<cv::Vec3b> color_table = {
    cv::Vec3b(255, 0, 0), cv::Vec3b(0, 255, 0), cv::Vec3b(0, 0, 255), cv::Vec3b(255, 255, 0), cv::Vec3b(0, 255, 255),
    cv::Vec3b(255, 0, 255), cv::Vec3b(255, 170, 0), cv::Vec3b(255, 0, 170), cv::Vec3b(0, 255, 170), cv::Vec3b(170, 255, 0),
    cv::Vec3b(170, 0, 255), cv::Vec3b(0, 170, 255), cv::Vec3b(255, 85, 0), cv::Vec3b(85, 255, 0), cv::Vec3b(0, 255, 85),
    cv::Vec3b(0, 85, 255), cv::Vec3b(85, 0, 255), cv::Vec3b(255, 0, 85)};

struct LetterboxMap {
    float factor{1.0f};
    int pad_w{0};
    int pad_h{0};
    int crop_w{0};
    int crop_h{0};
};

struct Quadruple {
    std::vector<HailoTensorPtr> boxes;
    xt::xarray<float> scores;
    std::vector<HailoTensorPtr> masks;
    xt::xarray<float> proto_data;
};

struct DetectionAndMask {
    HailoDetection detection;
    cv::Mat mask;
};
    
std::string getCmdOptionWithShortFlag(int argc, char *argv[], const std::string &longOption, const std::string &shortOption);

// ─────────────────────────────────────────────────────────────────────────────
// Postprocess of HEF with HailoRT-Postprocess
// ─────────────────────────────────────────────────────────────────────────────

template <typename T>
cv::Mat draw_detections_and_mask(std::vector<T>& logits,
                                 int width, int height,
                                 cv::Mat& frame);

                                 
std::vector<const hailo_detection_with_byte_mask_t*> get_detections(const uint8_t *src_ptr);

cv::Mat draw_detections_and_mask(const uint8_t *src_ptr,
                                          int width, int height,
                                          cv::Mat &frame);

cv::Mat pad_frame_letterbox(const cv::Mat &frame, int model_h, int model_w);

cv::Mat make_model_space_canvas(const cv::Mat &src,
    int model_w, int model_h,
    LetterboxMap &map);

void map_model_to_frame(const cv::Mat &model_space,
    const LetterboxMap &map,
    cv::Mat &dst_frame);


    
// ─────────────────────────────────────────────────────────────────────────────
// Postprocess of HEF without HailoRT-Postprocess
// ─────────────────────────────────────────────────────────────────────────────
std::vector<DetectionAndMask> segmentation_postprocess(std::vector<HailoTensorPtr> &tensors,
                                                                                std::vector<int> network_dims,
                                                                                std::vector<int> strides,
                                                                                int regression_length,
                                                                                int num_classes,
                                                                                int org_image_height, 
                                                                                int org_image_width);
HailoROIPtr build_roi_from_outputs(const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &outputs);
std::vector<HailoDetectionPtr> get_detections_from_roi(const HailoROIPtr &roi);
void draw_masks_and_boxes(cv::Mat &frame,
                                   const std::vector<HailoDetectionPtr> &dets,
                                   const std::vector<cv::Mat> &masks,
                                   float alpha = 0.7f,
                                   float thresh = 0.5f);

__BEGIN_DECLS
std::vector<cv::Mat> filter(HailoROIPtr roi, int org_width, int org_height);
__END_DECLS

#endif