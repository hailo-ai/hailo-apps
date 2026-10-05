/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
**/
#pragma once
#include <vector>
#include "hailo_objects.hpp"
#include "hailo_common.hpp"
#include "hailomat.hpp"

__BEGIN_DECLS
std::vector<HailoROIPtr> all_detections(std::shared_ptr<HailoMat> image, HailoROIPtr roi);
__END_DECLS