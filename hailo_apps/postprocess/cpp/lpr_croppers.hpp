/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
* Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
**/
#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include "hailo_objects.hpp"
#include "hailo_common.hpp"
#include "hailomat.hpp"

#define CROP_RATIO 0.1
#define QUALITY_THRESHOLD 50.0
#define CROP_WIDTH_LIMIT 10
#define CROP_HEIGHT_LIMIT 10

__BEGIN_DECLS
float quality_estimation(std::shared_ptr<HailoMat> hailo_mat, const HailoBBox &roi, const float crop_ratio);
std::vector<HailoROIPtr> license_plate_quality_estimation(std::shared_ptr<HailoMat> image, HailoROIPtr roi);
std::vector<HailoROIPtr> license_plate_no_quality(std::shared_ptr<HailoMat> image, HailoROIPtr roi);
std::vector<HailoROIPtr> license_plate_fullframe(std::shared_ptr<HailoMat> image, HailoROIPtr roi);
std::vector<HailoROIPtr> vehicles_without_ocr(std::shared_ptr<HailoMat> image, HailoROIPtr roi);
__END_DECLS
