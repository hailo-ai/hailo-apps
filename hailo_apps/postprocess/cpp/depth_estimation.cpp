/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 **/
#include "depth_estimation.hpp"
#include "common/tensors.hpp"

#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"

const char *output_layer_name_scdepth = "scdepthv3/conv31";

void filter(HailoROIPtr roi)
{
    filter_scdepth(roi);
}

void filter_scdepth(HailoROIPtr roi) {
    if (!roi->has_tensors())
    {
        return;
    }
    HailoTensorPtr tensor_ptr = roi->get_tensor(output_layer_name_scdepth);

    // get the output buffer in uint16 format, and parse it to xarray in the proper size
    xt::xarray<uint16_t> tensor_data = common::get_xtensor_uint16(tensor_ptr);

    // de-quantization of the xarray from uint16 to float32
    xt::xarray<float> logits_dequantized = common::dequantize(tensor_data, tensor_ptr->quant_info().qp_scale, tensor_ptr->quant_info().qp_zp);
    // here, logits_dequantized containes the estimated depth of each pixel in meters.

    // allocate and memcpy to a new memory so it points to the right data
    std::vector<float> data(logits_dequantized.size());
    memcpy(data.data(), logits_dequantized.data(), sizeof(float) * logits_dequantized.size());

    // Adapt the data vector to an xt::xarray
    xt::xarray<float> input = xt::adapt(data, {tensor_ptr->height(), tensor_ptr->width()});
    xt::xarray<float> output = xt::zeros<float>({tensor_ptr->height(), tensor_ptr->width()});
    // Apply the scaling factor to the output
    output = xt::exp(-input);  // cv::exp(-input, output);
    output = 1 / (1 + output);
    output = 1 / (output * 10 + 0.009);

    // Convert xt::xarray<float> to std::vector<float>
    std::vector<float> output_vector(output.begin(), output.end());

    hailo_common::add_object(roi, std::make_shared<HailoDepthMask>(std::move(output_vector), tensor_ptr->width(), tensor_ptr->height(), 1.0));
}
