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
#include "hailo_objects.hpp"
#include "hailo_common.hpp"
#include "lpr_ocrsink.hpp"
#include "hailo_cv_singleton.hpp"
#include "hailo_tracker.hpp"
#include "image.hpp"

// Open source includes
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

// General
#define MAP_LIMIT (5)              // Number of license plates to store at any time
#define OCR_SCORE_THRESHOLD (0.90) // OCR score threshold
int singleton_map_key = 0;
std::vector<int> seen_ocr_track_ids;
const gchar *OCR_LABEL_TYPE = "text_region";
std::string tracker_name = "hailo_tracker";

static bool lpr_debug_enabled()
{
    static int enabled = -1;
    if (enabled == -1)
    {
        const char *val = std::getenv("HAILO_LPR_DEBUG");
        enabled = (val && val[0] && val[0] != '0') ? 1 : 0;
    }
    return enabled == 1;
}

static void lpr_dbg(const char *fmt, ...)
{
    if (!lpr_debug_enabled())
        return;
    std::fprintf(stderr, "[lpr_ocrsink] ");
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
}

void catalog_nv12_mat(std::string text, std::vector<cv::Mat> &mat)
{
    // Resize the mat to a presentable size, add padding
    int target_h = 114;
    int target_w = 300;
    std::vector<cv::Mat> resized_nv12_vec;
    cv::Mat resized_nv12_y_mat = cv::Mat(target_h * 2 / 3, target_w, CV_8UC1);
    cv::Mat resized_nv12_uv_mat = cv::Mat(target_h / 3, target_w / 2, CV_8UC2);
    resized_nv12_vec.emplace_back(std::move(resized_nv12_y_mat));
    resized_nv12_vec.emplace_back(std::move(resized_nv12_uv_mat));

    resize_nv12(mat, resized_nv12_vec);

    // To make padding, prepare a padded mat and split channels from that mat
    int padded_h = target_h + 45;
    cv::Mat padded_nv12 = cv::Mat(padded_h, target_w, CV_8UC1, cv::Scalar(0));
    int padded_y_h = padded_h * 2 / 3;
    int padded_y_w = target_w;
    int padded_uv_h = padded_h / 3;
    int padded_uv_w = target_w / 2;
    cv::Mat padded_y_mat = cv::Mat(padded_y_h, padded_y_w, CV_8UC1, (char *)padded_nv12.data, padded_nv12.step);
    cv::Mat padded_uv_mat = cv::Mat(padded_uv_h, padded_uv_w, CV_8UC2, (char *)padded_nv12.data + (padded_y_h * padded_y_w), padded_nv12.step);

    // Fill the padded image with white padding
    cv::copyMakeBorder(resized_nv12_vec[0], padded_y_mat, 30, 0, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(235));
    cv::copyMakeBorder(resized_nv12_vec[1], padded_uv_mat, 15, 0, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(128, 128));

    // Draw text on the two channels
    cv::Point y_position = cv::Point(4, 24);
    cv::Point uv_position = cv::Point(2, 12);
    cv::putText(padded_y_mat, text, y_position, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(65), 2);
    cv::putText(padded_uv_mat, text, uv_position, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(110, 255), 1);

    CVMatSingleton::GetInstance().set_mat_at_key(singleton_map_key % MAP_LIMIT, padded_nv12);
    CVMatSingleton::GetInstance().set_mat_type(HAILO_MAT_NV12);
}

void catalog_yuy2_mat(std::string text, cv::Mat &mat)
{
    // Resize the mat to a presentable size, add padding
    cv::Mat padded_yuy2;
    cv::Mat resized_yuy2 = cv::Mat(75, 150, CV_8UC4);
    resize_yuy2(mat, resized_yuy2);

    // Add padding top, bottom, left, right, borderType
    cv::copyMakeBorder(resized_yuy2, padded_yuy2, 30, 0, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(235, 128, 235, 128));

    // write the OCR text on that padding (view as 2 channel su the yuy2 draws correctly)
    cv::Mat image_2_channel = cv::Mat(padded_yuy2.rows, padded_yuy2.cols * 2, CV_8UC2, (char *)padded_yuy2.data, padded_yuy2.step);
    auto text_position = cv::Point(5, 25);
    cv::putText(image_2_channel, text, text_position, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(81, 90, 81, 239), 2);

    // Set the new license plate in our CV Map singleton
    CVMatSingleton::GetInstance().set_mat_at_key(singleton_map_key % MAP_LIMIT, padded_yuy2);
    CVMatSingleton::GetInstance().set_mat_type(HAILO_MAT_YUY2);
}

void catalog_rgb_mat(std::string text, cv::Mat &mat)
{
    // Resize the mat to a presentable size, add padding
    cv::Mat resized_image;
    cv::Mat padded_image;
    cv::resize(mat, resized_image, cv::Size(300, 75), 0, 0, cv::INTER_AREA);

    // Add padding top, bottom, left, right, borderType
    cv::copyMakeBorder(resized_image, padded_image, 30, 0, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

    // write the OCR text on that padding
    auto text_position = cv::Point(10, 25);
    cv::putText(padded_image, text, text_position, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);

    // Set the new license plate in our CV Map singleton
    CVMatSingleton::GetInstance().set_mat_at_key(singleton_map_key % MAP_LIMIT, padded_image);
    CVMatSingleton::GetInstance().set_mat_type(HAILO_MAT_RGB);
}

void catalog_license_plate(std::string label, float confidence, HailoBBox license_plate_box, std::shared_ptr<HailoMat> hmat, HailoROIPtr crop_roi)
{
    if (!hmat)
    {
        lpr_dbg("catalog_license_plate: null HailoMat (label='%s')", label.c_str());
        return;
    }
    cv::Mat &mat = hmat->get_matrices()[0];
    // Prepare the cropped license plate and text
    std::string text = label + " " + std::to_string((int)(confidence * 100)) + "%";
    cv::Rect rect;
    rect.x = CLAMP(license_plate_box.xmin() * mat.cols, 0, mat.cols);
    rect.y = CLAMP(license_plate_box.ymin() * mat.rows, 0, mat.rows);
    rect.width = CLAMP(license_plate_box.width() * mat.cols, 0, mat.cols - rect.x);
    rect.height = CLAMP(license_plate_box.height() * mat.rows, 0, mat.rows - rect.y);
    if (rect.width == 0 || rect.height == 0)
    {
        lpr_dbg("catalog_license_plate: zero rect (label='%s' conf=%.3f)", label.c_str(), confidence);
        return;
    }

    const float xmin = CLAMP(license_plate_box.xmin(), 0.0f, 1.0f);
    const float ymin = CLAMP(license_plate_box.ymin(), 0.0f, 1.0f);
    const float xmax = CLAMP(license_plate_box.xmax(), xmin, 1.0f);
    const float ymax = CLAMP(license_plate_box.ymax(), ymin, 1.0f);
    auto safe_crop_roi = std::make_shared<HailoROI>(HailoBBox(xmin, ymin, (xmax - xmin), (ymax - ymin)));

    lpr_dbg(
        "catalog_license_plate: label='%s' conf=%.3f mat_type=%d bbox=[%.3f,%.3f,%.3f,%.3f] safe=[%.3f,%.3f,%.3f,%.3f] rect=[%d,%d,%d,%d]",
        label.c_str(),
        confidence,
        static_cast<int>(hmat->get_type()),
        license_plate_box.xmin(),
        license_plate_box.ymin(),
        license_plate_box.xmax(),
        license_plate_box.ymax(),
        xmin,
        ymin,
        xmax,
        ymax,
        rect.x,
        rect.y,
        rect.width,
        rect.height);

    std::vector<cv::Mat> cropped_image_vec;
    try
    {
        cropped_image_vec = hmat->crop(safe_crop_roi);
    }
    catch (const cv::Exception &e)
    {
        lpr_dbg("catalog_license_plate: cv::Exception during crop: %s", e.what());
        return;
    }
    catch (const std::exception &e)
    {
        lpr_dbg("catalog_license_plate: std::exception during crop: %s", e.what());
        return;
    }
    catch (...)
    {
        lpr_dbg("catalog_license_plate: unknown exception during crop");
        return;
    }

    if (cropped_image_vec.empty() || cropped_image_vec[0].empty())
    {
        lpr_dbg("catalog_license_plate: crop returned empty mats (vec=%zu)", cropped_image_vec.size());
        return;
    }

    switch (hmat->get_type())
    {
    case HAILO_MAT_YUY2:
    {
        catalog_yuy2_mat(text, cropped_image_vec[0]);
        break;
    }
    case HAILO_MAT_RGB:
    {
        catalog_rgb_mat(text, cropped_image_vec[0]);
        break;
    }
    case HAILO_MAT_NV12:
    {
        catalog_nv12_mat(text, cropped_image_vec);
        break;
    }
    default:
        break;
    }

    singleton_map_key++;
}

void ocr_sink(HailoROIPtr roi, std::shared_ptr<HailoMat> hmat)
{
    lpr_dbg("========== ocr_sink: ENTER ==========");
    if (nullptr == roi)
    {
        lpr_dbg("ocr_sink: null ROI => EXIT");
        return;
    }
    if (!hmat)
    {
        lpr_dbg("ocr_sink: null HailoMat => EXIT");
        return;
    }

    std::vector<HailoDetectionPtr> vehicle_detections;   // The vehicle detections in the ROI
    std::vector<HailoUniqueIDPtr> unique_ids;            // The unique ids of those vehicle detections
    std::vector<HailoDetectionPtr> lp_detections;        // The license plate detections in those vehicle detections
    std::vector<HailoClassificationPtr> classifications; // The classifications of those license plate detections
    float confidence;                                    // The confidence of those classifications
    std::string license_plate_ocr_label;                 // The labels of those classifications
    std::string jde_tracker_name = tracker_name + "_" + roi->get_stream_id();

    // For each roi, check the detections
    vehicle_detections = hailo_common::get_hailo_detections(roi);
    lpr_dbg("ocr_sink: stream_id='%s' total_vehicles=%zu OCR_LABEL_TYPE='%s'", 
            roi->get_stream_id().c_str(), vehicle_detections.size(), OCR_LABEL_TYPE);
    lpr_dbg("ocr_sink: seen_ocr_track_ids count=%zu", seen_ocr_track_ids.size());
    
    int veh_idx = 0;
    for (HailoDetectionPtr &vehicle_detection : vehicle_detections)
    {
        std::string veh_label = vehicle_detection->get_label();
        float veh_conf = vehicle_detection->get_confidence();
        lpr_dbg("ocr_sink: [veh %d] label='%s' conf=%.3f", veh_idx, veh_label.c_str(), veh_conf);
        
        // Get the unique id of the detection
        unique_ids = hailo_common::get_hailo_unique_id(vehicle_detection);
        if (unique_ids.empty())
        {
            lpr_dbg("ocr_sink: [veh %d] SKIP - no unique ID", veh_idx);
            veh_idx++;
            continue;
        }
        int track_id = unique_ids[0]->get_id();
        lpr_dbg("ocr_sink: [veh %d] track_id=%d", veh_idx, track_id);

        // For each vehicle, get the license plate detection
        lp_detections = hailo_common::get_hailo_detections(vehicle_detection);
        lpr_dbg("ocr_sink: [veh %d] nested LP detections=%zu", veh_idx, lp_detections.size());
        
        int lp_idx = 0;
        for (HailoDetectionPtr &lp_detection : lp_detections)
        {
            std::string lp_label = lp_detection->get_label();
            float lp_conf = lp_detection->get_confidence();
            HailoBBox lp_bbox = lp_detection->get_bbox();
            lpr_dbg("ocr_sink: [veh %d][lp %d] label='%s' conf=%.3f bbox=[%.3f,%.3f,%.3f,%.3f]", 
                    veh_idx, lp_idx, lp_label.c_str(), lp_conf,
                    lp_bbox.xmin(), lp_bbox.ymin(), lp_bbox.width(), lp_bbox.height());
            
            HailoBBox license_plate_box = hailo_common::create_flattened_bbox(lp_detection->get_bbox(), lp_detection->get_scaling_bbox());
            
            // For each license plate detection, check the classifications
            classifications = hailo_common::get_hailo_classifications(lp_detection);
            lpr_dbg("ocr_sink: [veh %d][lp %d] classifications count=%zu (expected exactly 1)", 
                    veh_idx, lp_idx, classifications.size());
            
            // List all classifications for debugging
            for (size_t cls_i = 0; cls_i < classifications.size(); cls_i++)
            {
                lpr_dbg("ocr_sink: [veh %d][lp %d]   cls[%zu] type='%s' label='%s' conf=%.3f", 
                        veh_idx, lp_idx, cls_i,
                        classifications[cls_i]->get_classification_type().c_str(),
                        classifications[cls_i]->get_label().c_str(),
                        classifications[cls_i]->get_confidence());
            }
            
            if (classifications.size() != 1)
            {
                lpr_dbg("ocr_sink: [veh %d][lp %d] REMOVE LP - wrong classification count (%zu != 1)", 
                        veh_idx, lp_idx, classifications.size());
                vehicle_detection->remove_object(lp_detection); // If no ocr was found then remove this license plate
                lp_idx++;
                continue;
            }
            
            HailoClassificationPtr classification = classifications[0];
            std::string cls_type = classification->get_classification_type();
            std::string cls_label = classification->get_label();
            float cls_conf = classification->get_confidence();
            
            lpr_dbg("ocr_sink: [veh %d][lp %d] checking classification type='%s' vs expected='%s'", 
                    veh_idx, lp_idx, cls_type.c_str(), OCR_LABEL_TYPE);
            
            if (OCR_LABEL_TYPE == cls_type)
            {
                confidence = cls_conf;
                license_plate_ocr_label = cls_label;
                lpr_dbg("ocr_sink: [veh %d][lp %d] *** OCR MATCH *** text='%s' conf=%.3f", 
                        veh_idx, lp_idx, license_plate_ocr_label.c_str(), confidence);
                
                // Prominent debug print for detected license plate - easy to grep
                lpr_dbg(">>>>>> DETECTED LP: \"%s\" (confidence: %.1f%%, track_id: %d) <<<<<<", 
                        license_plate_ocr_label.c_str(), confidence * 100.0f, track_id);
                
                bool already_seen = std::find(seen_ocr_track_ids.begin(), seen_ocr_track_ids.end(), track_id) != seen_ocr_track_ids.end();
                if (already_seen)
                {
                    lpr_dbg("ocr_sink: [veh %d][lp %d] SKIP - track_id=%d already processed", veh_idx, lp_idx, track_id);
                    lp_idx++;
                    continue;
                }
                
                lpr_dbg("ocr_sink: [veh %d][lp %d] adding track_id=%d to seen list", veh_idx, lp_idx, track_id);
                seen_ocr_track_ids.emplace_back(track_id);

                // Update the tracker with the found ocr
                lpr_dbg("ocr_sink: [veh %d][lp %d] updating tracker '%s' with OCR result", veh_idx, lp_idx, jde_tracker_name.c_str());
                HailoTracker::GetInstance().add_object_to_track(jde_tracker_name,
                                                                track_id,
                                                                classification);

                lpr_dbg("ocr_sink: [veh %d][lp %d] cataloging license plate", veh_idx, lp_idx);
                catalog_license_plate(license_plate_ocr_label, confidence, license_plate_box, hmat, lp_detection);
            }
            else
            {
                lpr_dbg("ocr_sink: [veh %d][lp %d] SKIP - classification type mismatch (got '%s', expected '%s')", 
                        veh_idx, lp_idx, cls_type.c_str(), OCR_LABEL_TYPE);
            }
            lp_idx++;
        }
        veh_idx++;
    }
    lpr_dbg("========== ocr_sink: EXIT ==========");
    return;
}

void filter(HailoROIPtr roi, GstVideoFrame *frame)
{
    lpr_dbg("========== lpr_ocrsink filter: ENTER ==========");
    if (!frame)
    {
        lpr_dbg("lpr_ocrsink filter: null frame => EXIT");
        return;
    }
    if (!roi)
    {
        lpr_dbg("lpr_ocrsink filter: null ROI => EXIT");
        return;
    }
    
    std::shared_ptr<HailoMat> hmat = get_mat_by_format(*(&frame->buffer), &frame->info, 1, 1);
    lpr_dbg("lpr_ocrsink filter: frame=%dx%d format=%d mat_valid=%d", 
            frame->info.width, frame->info.height, frame->info.finfo->format, hmat ? 1 : 0);
    
    // Print ROI structure
    auto all_dets = hailo_common::get_hailo_detections(roi);
    lpr_dbg("lpr_ocrsink filter: ROI has %zu top-level detections", all_dets.size());
    
    ocr_sink(roi, hmat);
    lpr_dbg("========== lpr_ocrsink filter: EXIT ==========");
}
