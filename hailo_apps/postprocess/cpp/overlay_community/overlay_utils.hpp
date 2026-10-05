/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 *
 * Community fork: local overlay utils.
 **/
#pragma once

#include <opencv2/opencv.hpp>

__BEGIN_DECLS
#define CONFIDENCE 0.5

class Parallel_pixel_opencv : public cv::ParallelLoopBody
{
protected:
    cv::Vec3b *p;
    float transparency;
    int image_cols;
    int roi_cols;

public:
    Parallel_pixel_opencv(uint8_t *ptr, float transparency, int image_cols, int roi_cols) : p((cv::Vec3b *)ptr), transparency(transparency), image_cols(image_cols), roi_cols(roi_cols) {}
};

class ParallelPixelClassMask : public Parallel_pixel_opencv
{
private:
    uint8_t *mask_data;

public:
    ParallelPixelClassMask(uint8_t *ptr, uint8_t *mask_data, float transparency, int image_cols, int roi_cols) : Parallel_pixel_opencv(ptr, transparency, image_cols, roi_cols), mask_data(mask_data) {}

    virtual void operator()(const cv::Range &r) const
    {
        for (int i = r.start; i != r.end; ++i)
        {
            int index = i / roi_cols * image_cols + i % roi_cols;
            int pixel_id = mask_data[i];

            p[index][0] = p[index][0] * (1 - transparency) + indexToColor(pixel_id)[0] * transparency;
            p[index][1] = p[index][1] * (1 - transparency) + indexToColor(pixel_id)[1] * transparency;
            p[index][2] = p[index][2] * (1 - transparency) + indexToColor(pixel_id)[2] * transparency;
        }
    }
};

class ParallelPixelClassConfMask : public Parallel_pixel_opencv
{
private:
    float *mask_data;
    cv::Scalar mask_color;

public:
    ParallelPixelClassConfMask(uint8_t *ptr, uint8_t *mask_data, float transparency, int image_cols, int roi_cols, cv::Scalar mask_color) : Parallel_pixel_opencv(ptr, transparency, image_cols, roi_cols), mask_data((float *)mask_data), mask_color(mask_color) {}

    virtual void operator()(const cv::Range &r) const
    {
        for (int i = r.start; i != r.end; ++i)
        {
            if (mask_data[i] > CONFIDENCE)
            {
                int index = i / roi_cols * image_cols + i % roi_cols;

                p[index][0] = p[index][0] * (1 - transparency) + mask_color[0] * transparency;
                p[index][1] = p[index][1] * (1 - transparency) + mask_color[1] * transparency;
                p[index][2] = p[index][2] * (1 - transparency) + mask_color[2] * transparency;
            }
        }
    }
};

class ParallelPixelDepthMask : public Parallel_pixel_opencv
{
private:
    float *mask_data;

public:
    ParallelPixelDepthMask(uint8_t *ptr, uint8_t *mask_data, float transparency, int image_cols, int roi_cols) : Parallel_pixel_opencv(ptr, transparency, image_cols, roi_cols), mask_data((float *)mask_data) {}

    virtual void operator()(const cv::Range &r) const
    {
        for (int i = r.start; i != r.end; ++i)
        {
            int index = i / roi_cols * image_cols + i % roi_cols;

            int depth = p[index][0] * (1 - transparency) + std::clamp(255 * mask_data[i], 0.0f, 255.0f) * transparency;
            p[index][0] = depth;
            p[index][1] = depth;
            p[index][2] = depth;
        }
    }
};

__END_DECLS
