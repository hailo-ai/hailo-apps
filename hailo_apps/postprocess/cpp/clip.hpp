/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
**/
#pragma once
#include "hailo_objects.hpp"
#include "hailo_common.hpp"

__BEGIN_DECLS
class ClipParams
{
public:
    std::string tracker_name; // Should have the same name as the relevant hailo_tracker
    ClipParams(std::string tracker_name) : tracker_name(tracker_name) {}
};
ClipParams *init(std::string config_path, std::string func_name);

void filter(HailoROIPtr roi, void *params_void_ptr);
__END_DECLS