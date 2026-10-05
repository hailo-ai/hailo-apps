/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
**/
#pragma once
#include "hailo_objects.hpp"
#include "hailo_common.hpp"

__BEGIN_DECLS
void filter(HailoROIPtr roi);
void mobilenet_ssd(HailoROIPtr roi);
void mobilenet_ssd_merged(HailoROIPtr roi);
void mobilenet_ssd_visdrone(HailoROIPtr roi);
__END_DECLS