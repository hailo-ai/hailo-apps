/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
#include "lpr_croppers.hpp"
#include <array>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sys/stat.h>
#include <atomic>

#define LICENSE_PLATE_LABEL "license_plate"
#define OCR_LABEL "text_region"

static constexpr std::array<const char *, 2> VEHICLE_LABELS = {"car", "vehicle"};

// Frame counter for unique filenames
static std::atomic<int> g_frame_counter{0};
static std::atomic<int> g_vehicle_crop_counter{0};
static std::atomic<int> g_lp_crop_counter{0};

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

static bool lpr_save_crops_enabled()
{
    static int enabled = -1;
    if (enabled == -1)
    {
        const char *val = std::getenv("HAILO_LPR_SAVE_CROPS");
        enabled = (val && val[0] && val[0] != '0') ? 1 : 0;
    }
    return enabled == 1;
}

static const char* get_crops_dir()
{
    static const char* dir = nullptr;
    if (dir == nullptr)
    {
        dir = std::getenv("HAILO_LPR_CROPS_DIR");
        if (dir == nullptr || dir[0] == '\0')
            dir = "lpr_debug_crops";
    }
    return dir;
}

static void ensure_dir_exists(const std::string& path)
{
    mkdir(path.c_str(), 0755);
}

static void lpr_dbg(const char *fmt, ...)
{
    if (!lpr_debug_enabled())
        return;
    std::fprintf(stderr, "[lpr_croppers] ");
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
}

static bool is_vehicle_label(const std::string &label)
{
    for (const auto *vehicle_label : VEHICLE_LABELS)
    {
        if (label == vehicle_label)
            return true;
    }
    return false;
}

/**
 * @brief Save a crop image to disk for debugging
 */
static void save_crop_image(std::shared_ptr<HailoMat> image, const HailoBBox& bbox, 
                            const std::string& prefix, int id)
{
    if (!lpr_save_crops_enabled() || !image)
        return;
    
    try
    {
        cv::Mat& mat = image->get_matrices()[0];
        int img_width = image->width();
        int img_height = image->height();
        
        // Calculate crop coordinates
        int x1 = static_cast<int>(std::max(0.0f, bbox.xmin()) * img_width);
        int y1 = static_cast<int>(std::max(0.0f, bbox.ymin()) * img_height);
        int x2 = static_cast<int>(std::min(1.0f, bbox.xmax()) * img_width);
        int y2 = static_cast<int>(std::min(1.0f, bbox.ymax()) * img_height);
        
        if (x2 <= x1 || y2 <= y1)
            return;
        
        cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
        if (roi.x < 0 || roi.y < 0 || roi.x + roi.width > mat.cols || roi.y + roi.height > mat.rows)
            return;
        
        cv::Mat crop = mat(roi).clone();
        
        // Convert to BGR if needed for saving
        cv::Mat bgr_crop;
        if (crop.channels() == 3)
        {
            cv::cvtColor(crop, bgr_crop, cv::COLOR_RGB2BGR);
        }
        else
        {
            bgr_crop = crop;
        }
        
        // Create output directory
        std::string base_dir = get_crops_dir();
        ensure_dir_exists(base_dir);
        std::string sub_dir = base_dir + "/" + prefix;
        ensure_dir_exists(sub_dir);
        
        // Save image
        char filename[512];
        std::snprintf(filename, sizeof(filename), "%s/%s_%05d.jpg", sub_dir.c_str(), prefix.c_str(), id);
        cv::imwrite(filename, bgr_crop);
        
        lpr_dbg("SAVED: %s (%dx%d)", filename, bgr_crop.cols, bgr_crop.rows);
    }
    catch (const std::exception& e)
    {
        lpr_dbg("Failed to save crop: %s", e.what());
    }
}

/**
 * @brief Returns the calculate the variance of edges.
 *
 * @param image  -  cv::Mat
 *        The original image.
 *
 * @param roi  -  HailoBBox
 *        The ROI to read from the image
 *
 * @param crop_ratio  -  float
 *        The percent of the image to crop in from the edges (default 10%).
 *
 * @return float
 *         The variance of edges in the image.
 */
float quality_estimation(std::shared_ptr<HailoMat> hailo_mat, const HailoBBox &roi, const float crop_ratio = 0.1)
{
    lpr_dbg("  quality_estimation: roi=[%.3f,%.3f,%.3f,%.3f] crop_ratio=%.2f", 
            roi.xmin(), roi.ymin(), roi.width(), roi.height(), crop_ratio);
    
    // Crop the center of the roi from the image, avoid cropping out of bounds
    float roi_width = roi.width();
    float roi_height = roi.height();
    float roi_xmin = roi.xmin();
    float roi_ymin = roi.ymin();
    float roi_xmax = roi.xmax();
    float roi_ymax = roi.ymax();
    float x_offset = roi_width * crop_ratio;
    float y_offset = roi_height * crop_ratio;
    float cropped_xmin = CLAMP(roi_xmin + x_offset, 0, 1);
    float cropped_ymin = CLAMP(roi_ymin + y_offset, 0, 1);
    float cropped_xmax = CLAMP(roi_xmax - x_offset, cropped_xmin, 1);
    float cropped_ymax = CLAMP(roi_ymax - y_offset, cropped_ymin, 1);
    float cropped_width_n = cropped_xmax - cropped_xmin;
    float cropped_height_n = cropped_ymax - cropped_ymin;
    int cropped_width = int(cropped_width_n * hailo_mat->native_width());
    int cropped_height = int(cropped_height_n * hailo_mat->native_height());
    
    lpr_dbg("  quality_estimation: crop size=%dx%d (limits: w>%d, h>%d)", 
            cropped_width, cropped_height, CROP_WIDTH_LIMIT, CROP_HEIGHT_LIMIT);

    // If the cropepd image is too small then quality is zero
    if (cropped_width <= CROP_WIDTH_LIMIT || cropped_height <= CROP_HEIGHT_LIMIT)
    {
        lpr_dbg("  quality_estimation: FAIL - crop too small => returning -1.0");
        return -1.0;
    }

    // If it is not too small then we can make the crop
    HailoROIPtr crop_roi = std::make_shared<HailoROI>(HailoBBox(cropped_xmin, cropped_ymin, cropped_width_n, cropped_height_n));
    std::vector<cv::Mat> cropped_image_vec = hailo_mat->crop(crop_roi);

    // Convert image to BGR
    cv::Mat bgr_image;
    switch (hailo_mat->get_type())
    {
    case HAILO_MAT_YUY2:
    {
        cv::Mat cropped_image = cropped_image_vec[0];
        cv::Mat yuy2_image = cv::Mat(cropped_image.rows, cropped_image.cols * 2, CV_8UC2, (char *)cropped_image.data, cropped_image.step);
        cv::cvtColor(yuy2_image, bgr_image, cv::COLOR_YUV2BGR_YUY2);
        break;
    }
    case HAILO_MAT_NV12:
    {
        cv::Mat full_mat = cv::Mat(cropped_image_vec[0].rows + cropped_image_vec[1].rows, cropped_image_vec[0].cols, CV_8UC1);
        memcpy(full_mat.data, cropped_image_vec[0].data, cropped_image_vec[0].rows * cropped_image_vec[0].cols);
        memcpy(full_mat.data + cropped_image_vec[0].rows * cropped_image_vec[0].cols, cropped_image_vec[1].data, cropped_image_vec[1].rows * cropped_image_vec[1].cols);
        cv::cvtColor(full_mat, bgr_image, cv::COLOR_YUV2BGR_NV12);

        break;
    }
    default:
        bgr_image = cropped_image_vec[0];
        break;
    }

    // Resize the frame
    cv::Mat resized_image;
    cv::resize(bgr_image, resized_image, cv::Size(200, 40), 0, 0, cv::INTER_AREA);

    // Gaussian Blur
    cv::Mat gaussian_image;
    cv::GaussianBlur(resized_image, gaussian_image, cv::Size(3, 3), 0);

    // Convert to grayscale
    cv::Mat gray_image;
    cv::Mat gray_image_normalized;
    cv::cvtColor(gaussian_image, gray_image, cv::COLOR_BGR2GRAY);
    cv::normalize(gray_image, gray_image_normalized, 255, 0, cv::NORM_INF);

    // Compute the Laplacian of the gray image
    cv::Mat laplacian_image;
    cv::Laplacian(gray_image_normalized, laplacian_image, CV_64F);

    // Calculate the variance of edges
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian_image, mean, stddev, cv::Mat());
    float variance = stddev.val[0] * stddev.val[0];
    return variance;
}

/**
 * @brief Returns a vector of HailoROIPtr to crop and resize.
 *        Specific to LPR pipelines, this function assumes that
 *        license plate ROIs are nested inside vehicle detection ROIs.
 *
 * @param image  -  cv::Mat
 *        The original image.
 *
 * @param roi  -  HailoROIPtr
 *        The main ROI of this picture.
 *
 * @return std::vector<HailoROIPtr>
 *         vector of ROI's to crop and resize.
 */
std::vector<HailoROIPtr> license_plate_quality_estimation(std::shared_ptr<HailoMat> image, HailoROIPtr roi)
{
    std::vector<HailoROIPtr> crop_rois;
    float variance;
    lpr_dbg("========== license_plate_quality_estimation: ENTER ==========");
    if (!image || !roi)
    {
        lpr_dbg("license_plate_quality_estimation: null image=%d roi=%d => EXIT", image ? 1 : 0, roi ? 1 : 0);
        return crop_rois;
    }
    lpr_dbg("license_plate_quality_estimation: image size=%dx%d, QUALITY_THRESHOLD=%.1f", 
            image->width(), image->height(), QUALITY_THRESHOLD);
    
    // Get all detections.
    std::vector<HailoDetectionPtr> vehicle_ptrs = hailo_common::get_hailo_detections(roi);
    lpr_dbg("license_plate_quality_estimation: total detections=%zu (looking for vehicles)", vehicle_ptrs.size());
    
    int veh_idx = 0;
    for (HailoDetectionPtr &vehicle : vehicle_ptrs)
    {
        std::string veh_label = vehicle->get_label();
        lpr_dbg("license_plate_quality_estimation: [veh %d] label='%s' conf=%.3f", 
                veh_idx, veh_label.c_str(), vehicle->get_confidence());
        
        if (!is_vehicle_label(veh_label))
        {
            lpr_dbg("license_plate_quality_estimation: [veh %d] SKIP - not a vehicle", veh_idx);
            veh_idx++;
            continue;
        }
        
        // For each detection, check the inner detections
        std::vector<HailoDetectionPtr> license_plate_ptrs = hailo_common::get_hailo_detections(vehicle);
        lpr_dbg("license_plate_quality_estimation: [veh %d] nested detections=%zu (looking for LICENSE_PLATE_LABEL='%s')", 
                veh_idx, license_plate_ptrs.size(), LICENSE_PLATE_LABEL);
        
        int lp_idx = 0;
        for (HailoDetectionPtr &license_plate : license_plate_ptrs)
        {
            std::string lp_label = license_plate->get_label();
            float lp_conf = license_plate->get_confidence();
            HailoBBox lp_bbox = license_plate->get_bbox();
            
            lpr_dbg("license_plate_quality_estimation: [veh %d][lp %d] label='%s' conf=%.3f bbox=[%.3f,%.3f,%.3f,%.3f]", 
                    veh_idx, lp_idx, lp_label.c_str(), lp_conf,
                    lp_bbox.xmin(), lp_bbox.ymin(), lp_bbox.width(), lp_bbox.height());
            
            if (LICENSE_PLATE_LABEL != lp_label)
            {
                lpr_dbg("license_plate_quality_estimation: [veh %d][lp %d] SKIP - label mismatch (got '%s', expected '%s')", 
                        veh_idx, lp_idx, lp_label.c_str(), LICENSE_PLATE_LABEL);
                lp_idx++;
                continue;
            }
            
            HailoBBox license_plate_box = hailo_common::create_flattened_bbox(license_plate->get_bbox(), license_plate->get_scaling_bbox());
            lpr_dbg("license_plate_quality_estimation: [veh %d][lp %d] flattened bbox=[%.3f,%.3f,%.3f,%.3f]", 
                    veh_idx, lp_idx, license_plate_box.xmin(), license_plate_box.ymin(), 
                    license_plate_box.width(), license_plate_box.height());

            // Get the variance of the image, only add ROIs that are above threshold.
            variance = quality_estimation(image, license_plate_box, CROP_RATIO);
            lpr_dbg("license_plate_quality_estimation: [veh %d][lp %d] quality variance=%.3f (threshold=%.1f)", 
                    veh_idx, lp_idx, variance, QUALITY_THRESHOLD);

            if (variance >= QUALITY_THRESHOLD)
            {
                lpr_dbg("license_plate_quality_estimation: [veh %d][lp %d] KEEP - good quality, sending to OCR", veh_idx, lp_idx);
                
                // Save LP crop for debugging (before it goes to OCR)
                int crop_id = g_lp_crop_counter.fetch_add(1);
                save_crop_image(image, license_plate_box, "lp_to_ocr", crop_id);
                
                crop_rois.emplace_back(license_plate);
            }
            else
            {
                lpr_dbg("license_plate_quality_estimation: [veh %d][lp %d] REMOVE - quality too low (%.3f < %.1f)", 
                        veh_idx, lp_idx, variance, QUALITY_THRESHOLD);
                vehicle->remove_object(license_plate); // If it is not a good license plate, then remove it!
            }
            lp_idx++;
        }
        veh_idx++;
    }
    lpr_dbg("license_plate_quality_estimation: result crop_rois=%zu (plates to send to OCR)", crop_rois.size());
    lpr_dbg("========== license_plate_quality_estimation: EXIT ==========");
    return crop_rois;
}

/**
 * @brief Returns a vector of HailoROIPtr to crop and resize - all license plates
 *        without any quality filtering. This is a simplified version of
 *        license_plate_quality_estimation that sends all detected plates to OCR.
 *
 * @param image  -  cv::Mat
 *        The original image.
 *
 * @param roi  -  HailoROIPtr
 *        The main ROI of this picture.
 *
 * @return std::vector<HailoROIPtr>
 *         vector of ROI's to crop and resize.
 */
std::vector<HailoROIPtr> license_plate_no_quality(std::shared_ptr<HailoMat> image, HailoROIPtr roi)
{
    std::vector<HailoROIPtr> crop_rois;
    lpr_dbg("========== license_plate_no_quality: ENTER ==========");
    if (!image || !roi)
    {
        lpr_dbg("license_plate_no_quality: null image=%d roi=%d => EXIT", image ? 1 : 0, roi ? 1 : 0);
        return crop_rois;
    }
    lpr_dbg("license_plate_no_quality: image size=%dx%d", image->width(), image->height());
    
    // Get all detections (vehicles).
    std::vector<HailoDetectionPtr> vehicle_ptrs = hailo_common::get_hailo_detections(roi);
    lpr_dbg("license_plate_no_quality: total detections=%zu (looking for vehicles)", vehicle_ptrs.size());
    
    int veh_idx = 0;
    for (HailoDetectionPtr &vehicle : vehicle_ptrs)
    {
        std::string veh_label = vehicle->get_label();
        lpr_dbg("license_plate_no_quality: [veh %d] label='%s' conf=%.3f", 
                veh_idx, veh_label.c_str(), vehicle->get_confidence());
        
        if (!is_vehicle_label(veh_label))
        {
            lpr_dbg("license_plate_no_quality: [veh %d] SKIP - not a vehicle", veh_idx);
            veh_idx++;
            continue;
        }
        
        // For each vehicle, get all nested detections (license plates)
        std::vector<HailoDetectionPtr> license_plate_ptrs = hailo_common::get_hailo_detections(vehicle);
        lpr_dbg("license_plate_no_quality: [veh %d] nested detections=%zu (looking for LICENSE_PLATE_LABEL='%s')", 
                veh_idx, license_plate_ptrs.size(), LICENSE_PLATE_LABEL);
        
        int lp_idx = 0;
        for (HailoDetectionPtr &license_plate : license_plate_ptrs)
        {
            std::string lp_label = license_plate->get_label();
            float lp_conf = license_plate->get_confidence();
            HailoBBox lp_bbox = license_plate->get_bbox();
            
            lpr_dbg("license_plate_no_quality: [veh %d][lp %d] label='%s' conf=%.3f bbox=[%.3f,%.3f,%.3f,%.3f]", 
                    veh_idx, lp_idx, lp_label.c_str(), lp_conf,
                    lp_bbox.xmin(), lp_bbox.ymin(), lp_bbox.width(), lp_bbox.height());
            
            if (LICENSE_PLATE_LABEL != lp_label)
            {
                lpr_dbg("license_plate_no_quality: [veh %d][lp %d] SKIP - label mismatch (got '%s', expected '%s')", 
                        veh_idx, lp_idx, lp_label.c_str(), LICENSE_PLATE_LABEL);
                lp_idx++;
                continue;
            }
            
            // No quality check - just add all plates with matching label
            lpr_dbg("license_plate_no_quality: [veh %d][lp %d] KEEP - sending to OCR (no quality filter)", veh_idx, lp_idx);
            
            // Save LP crop for debugging (before it goes to OCR)
            // Use flattened bbox to get actual image coordinates
            HailoBBox lp_flat_bbox = hailo_common::create_flattened_bbox(lp_bbox, license_plate->get_scaling_bbox());
            int crop_id = g_lp_crop_counter.fetch_add(1);
            save_crop_image(image, lp_flat_bbox, "lp_to_ocr", crop_id);
            
            crop_rois.emplace_back(license_plate);
            lp_idx++;
        }
        veh_idx++;
    }
    lpr_dbg("license_plate_no_quality: result crop_rois=%zu (plates to send to OCR)", crop_rois.size());
    lpr_dbg("========== license_plate_no_quality: EXIT ==========");
    return crop_rois;
}

/**
 * @brief Returns a vector of HailoROIPtr to crop and resize.
 *        Specific to LPR pipelines, this function searches if
 *        a detected vehicle has an OCR classification. If not,
 *        then it is submitted for cropping.
 *        This function also throws out car detections that are not yet
 *        fully in the image.
 *
 * @param image  -  cv::Mat
 *        The original image.
 *
 * @param roi  -  HailoROIPtr
 *        The main ROI of this picture.
 *
 * @return std::vector<HailoROIPtr>
 *         vector of ROI's to crop and resize.
 */
std::vector<HailoROIPtr> vehicles_without_ocr(std::shared_ptr<HailoMat> image, HailoROIPtr roi)
{
    std::vector<HailoROIPtr> crop_rois;
    bool has_ocr = false;
    lpr_dbg("========== vehicles_without_ocr: ENTER ==========");
    if (!image || !roi)
    {
        lpr_dbg("vehicles_without_ocr: null image=%d roi=%d => EXIT", image ? 1 : 0, roi ? 1 : 0);
        return crop_rois;
    }
    lpr_dbg("vehicles_without_ocr: image size=%dx%d", image->width(), image->height());
    
    // Get all detections.
    std::vector<HailoDetectionPtr> detections_ptrs = hailo_common::get_hailo_detections(roi);
    lpr_dbg("vehicles_without_ocr: total detections=%zu", detections_ptrs.size());
    
    int det_idx = 0;
    for (HailoDetectionPtr &detection : detections_ptrs)
    {
        std::string label = detection->get_label();
        float conf = detection->get_confidence();
        lpr_dbg("vehicles_without_ocr: [%d] label='%s' conf=%.3f", det_idx, label.c_str(), conf);
        
        if (!is_vehicle_label(label))
        {
            lpr_dbg("vehicles_without_ocr: [%d] SKIP - not a vehicle label (expected 'car' or 'vehicle')", det_idx);
            det_idx++;
            continue;
        }
        
        HailoBBox vehicle_bbox = detection->get_bbox();
        lpr_dbg("vehicles_without_ocr: [%d] bbox=[%.3f,%.3f,%.3f,%.3f] (xmin,ymin,w,h)", 
                det_idx, vehicle_bbox.xmin(), vehicle_bbox.ymin(), vehicle_bbox.width(), vehicle_bbox.height());
        
        // If the bbox is not yet in the image, then throw it out
        if ((vehicle_bbox.xmin() < 0.0) ||
            (vehicle_bbox.xmax() > 1.0) ||
            (vehicle_bbox.ymin() < 0.0) ||
            (vehicle_bbox.ymax() > 1.0))
        {
            lpr_dbg("vehicles_without_ocr: [%d] SKIP - bbox out of bounds [xmin=%.3f,xmax=%.3f,ymin=%.3f,ymax=%.3f]", 
                    det_idx, vehicle_bbox.xmin(), vehicle_bbox.xmax(), vehicle_bbox.ymin(), vehicle_bbox.ymax());
            det_idx++;
            continue;
        }

        int vehicle_width = vehicle_bbox.width() * image->width();
        int vehicle_height = vehicle_bbox.height() * image->height();
        int vehicle_area = vehicle_width * vehicle_height;
        lpr_dbg("vehicles_without_ocr: [%d] vehicle size=%dx%d area=%d (min required: 40000)", 
                det_idx, vehicle_width, vehicle_height, vehicle_area);
        
        if (vehicle_area < 40000)
        {
            lpr_dbg("vehicles_without_ocr: [%d] SKIP - vehicle too small (area=%d < 40000)", det_idx, vehicle_area);
            det_idx++;
            continue;
        }

        // if the bbox is above the top half of the image then throw it out
        if (vehicle_bbox.ymax() < 0.75)
        {
            lpr_dbg("vehicles_without_ocr: [%d] SKIP - vehicle too high in frame (ymax=%.3f < 0.75)", det_idx, vehicle_bbox.ymax());
            det_idx++;
            continue;
        }

        has_ocr = false;
        // For each detection, check the classifications
        std::vector<HailoClassificationPtr> vehicle_classifications = hailo_common::get_hailo_classifications(detection);
        lpr_dbg("vehicles_without_ocr: [%d] checking %zu classifications for OCR_LABEL='%s'", 
                det_idx, vehicle_classifications.size(), OCR_LABEL);
        
        for (HailoClassificationPtr &classification : vehicle_classifications)
        {
            std::string cls_type = classification->get_classification_type();
            std::string cls_label = classification->get_label();
            lpr_dbg("vehicles_without_ocr: [%d]   classification type='%s' label='%s'", det_idx, cls_type.c_str(), cls_label.c_str());
            if (OCR_LABEL == cls_type)
            {
                lpr_dbg("vehicles_without_ocr: [%d]   => MATCH! Vehicle already has OCR", det_idx);
                has_ocr = true;
                break;
            }
        }
        
        if (!has_ocr)
        {
            lpr_dbg("vehicles_without_ocr: [%d] ENQUEUE - vehicle needs LP detection", det_idx);
            
            // Save vehicle crop for debugging (before it goes to LP detection)
            int crop_id = g_vehicle_crop_counter.fetch_add(1);
            save_crop_image(image, vehicle_bbox, "vehicle_to_lp_det", crop_id);
            
            crop_rois.emplace_back(detection);
        }
        else
        {
            lpr_dbg("vehicles_without_ocr: [%d] SKIP - vehicle already has OCR classification", det_idx);
        }
        det_idx++;
    }
    lpr_dbg("vehicles_without_ocr: result crop_rois=%zu", crop_rois.size());
    lpr_dbg("========== vehicles_without_ocr: EXIT ==========");
    return crop_rois;
}
