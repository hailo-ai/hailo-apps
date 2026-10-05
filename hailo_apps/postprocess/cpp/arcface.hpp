/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
**/
#pragma once
#include "hailo_objects.hpp"
#include "hailo_common.hpp"

__BEGIN_DECLS
void arcface_rgb(HailoROIPtr roi);
void arcface_rgba(HailoROIPtr roi);
void arcface_nv12(HailoROIPtr roi);
void filter(HailoROIPtr roi);
__END_DECLS