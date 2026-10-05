/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 **/
#pragma once
#include "hailo_objects.hpp"
#include "hailo_common.hpp"
class LabelsParams
{
public:
    std::set<std::string> labels_to_remove;
    LabelsParams(std::set<std::string> &&labels_to_remove) : labels_to_remove(std::move(labels_to_remove)) {}
};
__BEGIN_DECLS
LabelsParams *init(const std::string config_path, const std::string function_name);
void filter(HailoROIPtr roi, void *params_void_ptr);
__END_DECLS