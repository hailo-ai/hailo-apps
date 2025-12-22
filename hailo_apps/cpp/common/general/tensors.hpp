/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
* Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
**/
#pragma once

#include "hailo_objects.hpp"
#include "xtensor/containers/xadapt.hpp"
#include "xtensor/containers/xarray.hpp"

namespace common
{
    //-------------------------------
    // COMMON TRANSFORMS
    //-------------------------------
    template <typename T>
    xt::xarray<float> dequantize(const xt::xarray<T> &input, const float &qp_scale, const float &qp_zp)
    {
        // Rescale the input using the given scale and zero-point
        auto rescaled_data = (input - qp_zp) * qp_scale;
        return rescaled_data;
    }

    xt::xarray<uint8_t> get_xtensor(HailoTensorPtr &tensor)
    {
        // Adapt a HailoTensorPtr to an xarray (quantized)
        xt::xarray<uint8_t> xtensor = xt::adapt(tensor->data(), tensor->size(), xt::no_ownership(), tensor->shape());
        return xtensor;
    }

    xt::xarray<uint16_t> get_xtensor_uint16(HailoTensorPtr &tensor)
    {
        // Adapt a HailoTensorPtr to an xarray (quantized)
        uint16_t *data = (uint16_t *)(tensor->data());
        xt::xarray<uint16_t> xtensor = xt::adapt(data, tensor->size(), xt::no_ownership(), tensor->shape());
        return xtensor;
    }

    xt::xarray<float> get_xtensor_float(HailoTensorPtr &tensor)
    {
        // Adapt a HailoTensorPtr to an xarray (quantized)
        auto vstream_info = tensor->vstream_info();
        xt::xarray<uint8_t> xtensor = get_xtensor(tensor);
        return dequantize(xtensor, vstream_info.quant_info.qp_scale, vstream_info.quant_info.qp_zp);
    }
    /**
     * @brief Get the only the tensors (vector) from a map of string->tensor.
     * 
     * @param tensors A map between tensors name to the tensor pointer
     * @return std::vector<HailoTensorPtr> A vector of tensor pointer.
     */
    std::vector<HailoTensorPtr> get_tensor_values(const std::map<std::string, HailoTensorPtr> &tensors)
    {
        std::vector<HailoTensorPtr> _tensors;
        _tensors.reserve(tensors.size());
        for (auto &tensor_pair : tensors)
        {
            _tensors.emplace_back(tensor_pair.second);
        }
        return _tensors;
    }

}