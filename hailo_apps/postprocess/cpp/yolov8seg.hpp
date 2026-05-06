/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 *
 * YOLOv8 Instance Segmentation postprocess header.
 **/
#pragma once

#include "hailo_objects.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"

#include <string>
#include <vector>

__BEGIN_DECLS

/**
 * @brief Explicit tensor names for each detection scale plus the prototype tensor.
 *
 * When 'is_set()' returns true the postprocess uses name-based tensor lookup
 * instead of the channel-count heuristic, making it robust to any number of
 * classes and any model naming convention.
 *
 * The three vectors (boxes, scores, masks) must be the same length and ordered
 * by stride ascending (small stride first, i.e. stride-8 → stride-16 → stride-32).
 */
struct Yolov8segOutputsName
{
    std::string proto;                   ///< prototype tensor name
    std::vector<std::string> boxes;      ///< box regression tensors, one per scale
    std::vector<std::string> scores;     ///< class-score tensors, one per scale
    std::vector<std::string> masks;      ///< mask-coefficient tensors, one per scale

    /// Returns true when all name lists are populated (proto + at least one scale)
    bool is_set() const
    {
        return !proto.empty() &&
               !boxes.empty() &&
               boxes.size() == scores.size() &&
               boxes.size() == masks.size();
    }
};

class Yolov8segParams
{
public:
    float iou_threshold;
    float score_threshold;
    std::vector<int> input_shape;
    std::vector<int> strides;
    int num_classes;                   ///< number of detection classes (default: 80 = COCO)
    Yolov8segOutputsName outputs_name; ///< optional explicit tensor names

    Yolov8segParams()
    {
        iou_threshold = 0.6;
        score_threshold = 0.25;
        input_shape = {640, 640};
        strides = {32, 16, 8};
        num_classes = 80;
    }
};

Yolov8segParams *init(const std::string config_path, const std::string function_name);
void yolov8seg(HailoROIPtr roi, void *params_void_ptr);
void free_resources(void *params_void_ptr);
void filter(HailoROIPtr roi, void *params_void_ptr);
void filter_letterbox(HailoROIPtr roi, void *params_void_ptr);

__END_DECLS
