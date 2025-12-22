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
 * @file utils.hpp
 * Common macros and defines used by Hailort Examples
 **/

#ifndef _EXAMPLE_COMMON_H_
#define _EXAMPLE_COMMON_H_

#include <iostream>
#include <string>
#include <vector>
#include "hailo/hailort.h"
#include "toolbox.hpp"
using namespace hailo_utils;
// --------------------------- FUNCTION DECLARATIONS ---------------------------

struct NamedBbox {
    hailo_bbox_float32_t bbox;
    size_t class_id;
};

// ─────────────────────────────────────────────────────────────────────────────
// POSTPROCESSING
// ─────────────────────────────────────────────────────────────────────────────

cv::Rect get_bbox_coordinates(const hailo_bbox_float32_t &bbox, int frame_width, int frame_height);
void draw_label(cv::Mat &frame, const std::string &label, const cv::Point &top_left, const cv::Scalar &color);
void draw_single_bbox(cv::Mat &frame, const NamedBbox &named_bbox, const cv::Scalar &color);
void draw_bounding_boxes(cv::Mat &frame, const std::vector<NamedBbox> &bboxes);
std::vector<NamedBbox> parse_nms_data(uint8_t *data, size_t max_class_count);

// ─────────────────────────────────────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────────────────────────────────────

void initialize_class_colors(std::unordered_map<int, cv::Scalar> &class_colors);
std::string get_coco_name_from_int(int cls);

#endif