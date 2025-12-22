/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
#pragma once

#include "hailo_objects.hpp"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/views/xview.hpp"
#include "xtensor/io/xnpy.hpp"
#include <string>





namespace hailo_common
{
    inline HailoMatrixPtr create_matrix_ptr(xt::xarray<float> &xmatrix)
    {
        // allocate and memcpy to a new memory so it points to the right data
        std::vector<float> data(xmatrix.size());
        memcpy(data.data(), xmatrix.data(), sizeof(float) * xmatrix.size());
        return std::make_shared<HailoMatrix>(std::move(data),
                                            xmatrix.shape(0), xmatrix.shape(1), xmatrix.shape(2));
    }

    inline void dump_matrix_to_file(const std::string &filename, xt::xarray<float> xmatrix)
    {
        xt::dump_npy(filename, xmatrix);
    }

    inline void dump_hailo_matrix_to_file(const std::string &filename, HailoMatrixPtr &hailo_matrix)
    {
        xt::xarray<float> xmatrix = xt::adapt(hailo_matrix->get_data(),
                                              hailo_matrix->shape());
        dump_matrix_to_file(filename, xmatrix);
    }

    inline void add_landmarks_to_detection(HailoDetection &detection,
                                           std::string landmarks_type,
                                           xt::xarray<float> landmarks,
                                           float threshold = 1.0f,
                                           const std::vector<std::pair<int, int>> pairs = {})
    {
        HailoBBox bbox = detection.get_bbox();
        std::vector<HailoPoint> points;
        bool has_confidence;
        /**
         * @brief Determening whether the shape of the landmarks is valid:
         *  When there are 2 columns: each point has x and y positions.
         *  When there are 3 columns: Each point has x and y positions, and a confidence.
         *  Every other number of columns is invalid.
         */
        if (landmarks.shape(1) == 2)
        {
            has_confidence = false;
        }
        else if (landmarks.shape(1) == 3)
        {
            has_confidence = true;
        }
        else
        {
            throw std::invalid_argument("Landmarks should be with 2 or 3 columns only");
        }
        // Make these points relative to the BBox they are related too.
        xt::view(landmarks, xt::all(), 0) = (xt::view(landmarks, xt::all(), 0) - bbox.xmin()) / bbox.width();
        xt::view(landmarks, xt::all(), 1) = (xt::view(landmarks, xt::all(), 1) - bbox.ymin()) / bbox.height();
        // Create HailoPoint for each point and add it to the vector of points.
        points.reserve(landmarks.shape(0));
        for (uint i = 0; i < landmarks.shape(0); i++)
        {
            if (has_confidence)
            {
                points.emplace_back(landmarks(i, 0), landmarks(i, 1), landmarks(i, 2));
            }
            else
            {
                points.emplace_back(landmarks(i, 0), landmarks(i, 1));
            }
        }
        // Add HailoLandmarks pointer to the detection.
        detection.add_object(std::make_shared<HailoLandmarks>(landmarks_type, points, threshold, pairs));
    }
}