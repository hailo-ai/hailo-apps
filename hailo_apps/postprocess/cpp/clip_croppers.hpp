/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
**/
#pragma once
#include <vector>
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