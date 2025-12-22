/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/

#include <gst/video/video-format.h>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <typeinfo>
#include <math.h>

// Hailo includes
#include "lpr_overlay.hpp"
#include "image.hpp"
#include "hailo_cv_singleton.hpp"

// Open source includes
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

// General
#define TEXT_THICKNESS (1)
#define TEXT_CLS_THICKNESS (2)

#define OCR_LIMIT 5

static bool lpr_debug_enabled()
{
    static int enabled = -1;
    if (enabled == -1)
    {
        const char *val = std::getenv("HAILO_LPR_DEBUG");
        enabled = (!val || val[0] == '\0' || val[0] != '0') ? 1 : 0;
    }
    return enabled == 1;
}

static void lpr_dbg(const char *fmt, ...)
{
    if (!lpr_debug_enabled())
        return;
    std::fprintf(stderr, "[lpr_overlay] ");
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
}

void draw_lpr(cv::Mat &mat)
{
    lpr_dbg("draw_lpr: ENTER mat=%dx%d type=%d", mat.cols, mat.rows, mat.type());
    
    cv::Mat singleton_image;
    cv::Mat destinationROI;
    cv::Mat y_destinationROI;
    cv::Mat uv_destinationROI;
    hailo_mat_t mat_format;
    int drawn_count = 0;
    
    for (int i = 0; i < OCR_LIMIT; i++)
    {
        // Get the cv::Mat for this index, if any
        singleton_image = CVMatSingleton::GetInstance().get_mat_at_key(i);
        mat_format = CVMatSingleton::GetInstance().get_mat_type();
        if (!singleton_image.empty() && mat_format != HAILO_MAT_NONE)
        {
            lpr_dbg("draw_lpr: slot[%d] has image %dx%d format=%d", i, singleton_image.cols, singleton_image.rows, mat_format);
            drawn_count++;
            switch (mat_format)
            {
            case HAILO_MAT_YUY2:
            case HAILO_MAT_RGB:
            {
                // The CV Map singleton had an image for this key, so draw it
                int xmin = mat.cols - singleton_image.cols;
                int ymin = i * singleton_image.rows;

                cv::Rect roi(cv::Point(xmin, ymin), cv::Size(singleton_image.cols, singleton_image.rows));
                destinationROI = mat(roi);
                singleton_image.copyTo(destinationROI);
                break;
            }
            case HAILO_MAT_NV12:
            {
                // Split mat image planes
                int y_h = mat.rows * 2 / 3;
                int y_w = mat.cols;
                int uv_h = mat.rows / 3;
                int uv_w = mat.cols / 2;
                cv::Mat y_mat = cv::Mat(y_h, y_w, CV_8UC1, (char *)mat.data, mat.step);
                cv::Mat uv_mat = cv::Mat(uv_h, uv_w, CV_8UC2, (char *)mat.data + (y_h * y_w), mat.step);

                // Split singleton image planes
                int s_y_h = singleton_image.rows * 2 / 3;
                int s_y_w = singleton_image.cols;
                int s_uv_h = singleton_image.rows / 3;
                int s_uv_w = singleton_image.cols / 2;
                cv::Mat y_singleton_mat = cv::Mat(s_y_h, s_y_w, CV_8UC1, (char *)singleton_image.data, singleton_image.step);
                cv::Mat uv_singleton_mat = cv::Mat(s_uv_h, s_uv_w, CV_8UC2, (char *)singleton_image.data + (s_y_h * s_y_w), singleton_image.step);
                
                // Copy over the y channels
                int y_xmin = y_mat.cols - y_singleton_mat.cols;
                int y_ymin = i * y_singleton_mat.rows;
                cv::Rect y_roi(cv::Point(y_xmin, y_ymin), cv::Size(y_singleton_mat.cols, y_singleton_mat.rows));
                y_destinationROI = y_mat(y_roi);
                y_singleton_mat.copyTo(y_destinationROI);

                // Copy over the uv channels
                int uv_xmin = uv_mat.cols - uv_singleton_mat.cols;
                int uv_ymin = i * uv_singleton_mat.rows;
                cv::Rect uv_roi(cv::Point(uv_xmin, uv_ymin), cv::Size(uv_singleton_mat.cols, uv_singleton_mat.rows));
                uv_destinationROI = uv_mat(uv_roi);
                uv_singleton_mat.copyTo(uv_destinationROI);
                
                break;
            }
            default:
                break;
            }
        }

        singleton_image.release();
        destinationROI.release();
    }
    lpr_dbg("draw_lpr: EXIT - drew %d license plate overlays", drawn_count);
}

void filter(HailoROIPtr roi, GstVideoFrame *frame)
{
    lpr_dbg("========== lpr_overlay filter: ENTER ==========");
    if (!frame)
    {
        lpr_dbg("lpr_overlay filter: null frame => EXIT");
        return;
    }
    
    auto image_planes = get_mat_from_gst_frame(frame);
    (void)roi;
    lpr_dbg("lpr_overlay filter: mat=%dx%d type=%d", image_planes.cols, image_planes.rows, image_planes.type());
    draw_lpr(image_planes);

    image_planes.release();
    lpr_dbg("========== lpr_overlay filter: EXIT ==========");
}
