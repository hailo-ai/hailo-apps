/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 **/
#include <vector>
#include "common/tensors.hpp"
#include "common/math.hpp"
#include "clip.hpp"
#include "hailo_tracker.hpp"
#include "hailo_xtensor.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"

ClipParams *init(std::string config_path, std::string func_name)
{
    if (config_path == "NULL")
    {
        config_path = "hailo_tracker";
    }
    ClipParams *params = new ClipParams(config_path);
    return params;
}

void clip(HailoROIPtr roi, std::string tracker_name)
{
    if (!roi->has_tensors())
    {
        return;
    }
    // Remove previous matrices
    roi->remove_objects_typed(HAILO_MATRIX);
    
    auto tensors = roi->get_tensors();
    if (tensors.empty()) {
        return;
    }
    HailoTensorPtr tensor = tensors[0];
    
    xt::xarray<float> embeddings = common::get_xtensor_float(tensor);

    // vector normalization
    auto normalized_embedding = common::vector_normalization(embeddings);
    HailoMatrixPtr hailo_matrix = hailo_common::create_matrix_ptr(normalized_embedding);
    roi->add_object(hailo_matrix);
    }

void filter(HailoROIPtr roi, void *params_void_ptr)
{
    ClipParams *params = reinterpret_cast<ClipParams *>(params_void_ptr);
    clip(roi, params->tracker_name);
}
