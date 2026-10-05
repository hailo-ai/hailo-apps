/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
**/
#pragma once
#include "general/hailo_objects.hpp"
#include "general/hailo_common.hpp"

#include <xtensor/views/xview.hpp>
#include <xtensor/misc/xsort.hpp>

struct KeyPt {
    float xs;
    float ys;
    float joints_scores;
};

struct PairPairs {
    std::pair<float, float> pt1;
    std::pair<float, float> pt2;
    float s1;
    float s2;
};

struct Triple {
    std::vector<HailoTensorPtr> boxes;
    xt::xarray<float> scores;
    std::vector<HailoTensorPtr> keypoints;
};

struct Decodings {
    HailoDetection detection_box;
    std::pair<xt::xarray<float>, xt::xarray<float>> keypoints;
    std::vector<PairPairs> joint_pairs;
};



std::pair<std::vector<KeyPt>, std::vector<PairPairs>> yolov8(HailoROIPtr roi);
void filter(HailoROIPtr roi);
void filter_letterbox(HailoROIPtr roi);
