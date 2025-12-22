/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
* Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
**/
/**
 * @file hailo_tensors.hpp
 * @authors Hailo
 **/

#pragma once
#include "hailo/hailort.h"
#include <memory>
#include <string>
#include <vector>

class HailoTensor
{
private:
    uint8_t *m_data;                     // Pointer to the data of the tensor.
    hailo_vstream_info_t m_vstream_info; // Pointer to vstream info.
    std::string m_name;                  // Name of output tensor.
public:
    /**
     * @brief Construct a new Hailo Tensor object
     *
     * @param data - Pointer to the tensor output.
     * @param vstream_info - pointer to info about the output, represented as hailo_vstream_info_t.
     */
    HailoTensor(uint8_t *data, const hailo_vstream_info_t &vstream_info) : m_data(data), m_vstream_info(vstream_info), m_name(m_vstream_info.name){};
    // Destructor
    ~HailoTensor() = default;
    // Copy constructor
    HailoTensor(const HailoTensor &other) = default;
    // Move constructor
    HailoTensor(HailoTensor &&other) = default;

    // Getters:
    std::string name()
    {
        return m_name;
    }
    hailo_vstream_info_t &vstream_info()
    {
        return m_vstream_info;
    }
    uint8_t *data()
    {
        return m_data;
    }
    const uint32_t width() { return m_vstream_info.shape.width; }
    const uint32_t height() { return m_vstream_info.shape.height; }
    const uint32_t features() { return m_vstream_info.shape.features; }
    const uint32_t size() const
    {
        return m_vstream_info.shape.width * m_vstream_info.shape.height * m_vstream_info.shape.features;
    }
    std::vector<std::size_t> shape()
    {
        std::vector<std::size_t> shape = {height(), width(), features()};
        return shape;
    }

    // Methods:
    /**
     * @brief Gets a quantized number and returns its dequantized value (float).
     *
     * @param num number to dequantize.
     * @return float dequantized number.
     */
    template <typename T>  
    float fix_scale(T num)
    {
        return (float(num) - m_vstream_info.quant_info.qp_zp) * m_vstream_info.quant_info.qp_scale;
    }

    /**
     * @brief Gets a dequantized number and returns its quantized value (template).
     *
     * @param num number to quantize.
     * @return T quantized number.
     */
    template <typename T>  
    T quantize(T num)
    {
        return T((float(num) / m_vstream_info.quant_info.qp_scale)  + m_vstream_info.quant_info.qp_zp);
    }

    /**
     * @brief Gets a specific cell of this tensor.
     *
     * @param row The row of the cell
     * @param col The column of the cell
     * @param channel The channel of the cell
     * @return uint8_t value of this tensor at the specified place.
     * @note number is quantized.
     */
    uint8_t get(uint row, uint col, uint channel)
    {
        uint width = m_vstream_info.shape.width;
        uint features = m_vstream_info.shape.features;
        int pos = (width * features) * row + features * col + channel;
        return m_data[pos];
    }
    uint16_t get_uint16(uint row, uint col, uint channel)
    {
        uint width = m_vstream_info.shape.width;
        uint features = m_vstream_info.shape.features;
        int pos = (width * features) * row + features * col + channel;
        uint16_t *data_uint16 = (uint16_t *)m_data;
        return data_uint16[pos];
    }

    /**
     * @brief Gets a specific cell of this tensor in full percision (dequantized).
     *
     * @param row The row of the cell
     * @param col The column of the cell
     * @param channel The channel of the cell
     * @return float value of this tensor at the specified place (dequantized).
     */
    float get_full_percision(uint row, uint col, uint channel, bool is_uint16)
    {
        if (is_uint16)
            return fix_scale(get_uint16(row, col, channel));
        else
            return fix_scale(get(row, col, channel));
    }
};

using HailoTensorPtr = std::shared_ptr<HailoTensor>;