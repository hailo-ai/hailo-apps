/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
#pragma once

#include "xtensor/containers/xarray.hpp"
#include "xtensor/core/xeval.hpp"
#include "xtensor/misc/xsort.hpp"
#include "xtensor/views/xview.hpp"
#include "xtensor/io/xio.hpp"

namespace common
{

    //-------------------------------
    // COMMON FILTERS
    //-------------------------------
    template <typename T>
    xt::xarray<int> top_k(xt::xarray<T> &data, const int k)
    {
        // First we negate the array so that we sort in descending order.
        auto descending_order_array = xt::eval(-data);
        // Partition the array so the k "smallest" items are first (argpartition returns this as an index array).
        // Since we negated the values first, we are actually partitioning the largest values.
        xt::xarray<int> krange = xt::arange<int>(0, k);
        xt::xarray<int> index_array = xt::argpartition(descending_order_array, krange, xt::xnone());

        // We only want the k first indices, so make a view that "takes" these.
        auto topk_index_array = xt::view(xt::reshape_view(index_array, data.shape()), xt::all(), xt::range(0, k));
        return topk_index_array;
    }

    xt::xarray<float> vector_normalization(xt::xarray<float> &data)
    {
        xt::xarray<float> data_squared = xt::square(data);
        xt::xarray<float> data_sum = xt::sum(data_squared);
        xt::xarray<float> data_sqrt = xt::sqrt(data_sum);
        xt::xarray<float> normalized = data / data_sqrt;
        return normalized;
    }

    xt::xarray<float> softmax_xtensor(xt::xarray<float> &scores)
    {
        // Compute softmax values for each sets of scores in x.
        auto maxes = xt::amax(scores, -1);
        xt::xarray<float> e_scores = xt::exp(scores - xt::expand_dims(maxes, 2));
        return std::move(e_scores / xt::expand_dims(xt::sum(e_scores, -1), 2));
    }

    void softmax_1D(float *data, const int size)
    {
        float sum = 0;
        for (int i = 0; i < size; i++)
            sum += std::exp(data[i]);
        for (int i = 0; i < size; i++)
            data[i] = std::exp(data[i]) / sum;
    }

    void softmax_2D(float *data, const int num_rows, const int num_cols)
    {
        int size = num_rows * num_cols;
        for (int i = 0; i < size; i += num_cols)
            softmax_1D(&data[i], num_cols);
    }

    void softmax_3D(float *data, const int dim1_size, const int dim2_size, const int dim3_size)
    {
        int size = dim1_size * dim2_size * dim3_size;
        for (int i = 0; i < size; i += dim2_size * dim3_size)
            softmax_2D(&data[i], dim2_size, dim3_size);
    }

    void sigmoid(float *data, const int size)
    {
        for (int i = 0; i < size; i++)
            data[i] = 1.0f / (1.0f + std::exp(-1.0 * data[i]));
    }

}