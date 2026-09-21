/**
 * Copyright (c) 2021-2026 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
#include "common/tensors.hpp"

namespace common
{

    xt::xarray<uint8_t> get_xtensor(const HailoTensorPtr &tensor)
    {
        // Adapt a HailoTensorPtr to an xarray (quantized)
        xt::xarray<uint8_t> xtensor = xt::adapt(tensor->data(), tensor->size(), xt::no_ownership(), tensor->shape());
        return xtensor;
    }

    xt::xarray<uint16_t> get_xtensor_uint16(const HailoTensorPtr &tensor)
    {
        // Adapt a HailoTensorPtr to an xarray (quantized)
        auto *data = reinterpret_cast<uint16_t *>(tensor->data());
        xt::xarray<uint16_t> xtensor = xt::adapt(data, tensor->size(), xt::no_ownership(), tensor->shape());
        return xtensor;
    }

    xt::xarray<float> get_xtensor_float(const HailoTensorPtr &tensor)
    {
        // Adapt a HailoTensorPtr to an xarray and dequantize to float
        auto quant_info = tensor->quant_info();
        xt::xarray<uint8_t> xtensor = get_xtensor(tensor);
        return dequantize(xtensor, quant_info.qp_scale, quant_info.qp_zp);
    }

    std::vector<HailoTensorPtr> get_tensor_values(const std::map<std::string, HailoTensorPtr> &tensors)
    {
        std::vector<HailoTensorPtr> result;
        result.reserve(tensors.size());
        for (const auto &tensor_pair : tensors) {
            result.emplace_back(tensor_pair.second);
        }
        return result;
    }

}
