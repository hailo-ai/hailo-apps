/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 **/
#include <vector>
#include <iostream>
#include "common/tensors.hpp"
#include "common/math.hpp"
#include "repvgg_reid.hpp"
#include "hailo_xtensor.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"

#define OUTPUT_LAYER_NAME "repvgg_a0_person_reid_512/fc1"

void re_id(HailoROIPtr roi)
{

    if (!roi->has_tensors())
    {
        return;
    }

    // Remove previous matrices
    roi->remove_objects_typed(HAILO_MATRIX);

    // Convert the tensor to xarray.
    auto tensor = roi->get_tensor(OUTPUT_LAYER_NAME);
    xt::xarray<float> embedding = common::get_xtensor_float(tensor);

    // vector normalization
    auto normalized_embedding = common::vector_normalization(embedding);

    roi->add_object(hailo_common::create_matrix_ptr(normalized_embedding));
}

void filter(HailoROIPtr roi)
{
    re_id(roi);
}
