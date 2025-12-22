/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
* Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
**/
#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "hailo_objects.hpp"
#include "hailo_common.hpp"
#include "hailomat.hpp"

__BEGIN_DECLS
std::vector<HailoROIPtr> object_cropper(std::shared_ptr<HailoMat> image, HailoROIPtr roi);
std::vector<HailoROIPtr> person_cropper(std::shared_ptr<HailoMat> image, HailoROIPtr roi);
std::vector<HailoROIPtr> vehicle_cropper(std::shared_ptr<HailoMat> image, HailoROIPtr roi);
std::vector<HailoROIPtr> face_cropper(std::shared_ptr<HailoMat> image, HailoROIPtr roi);
std::vector<HailoROIPtr> license_plate_cropper(std::shared_ptr<HailoMat> image, HailoROIPtr roi);

__END_DECLS