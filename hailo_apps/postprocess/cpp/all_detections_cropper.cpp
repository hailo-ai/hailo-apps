/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
**/
#include <vector>
#include <cmath>
#include <unordered_map>
#include "all_detections_cropper.hpp"

/**
* @brief Get the tracking Hailo Unique Id object from a Hailo Detection.
* 
* @param detection HailoDetectionPtr
* @return HailoUniqueIdPtr pointer to the Hailo Unique Id object
*/
HailoUniqueIDPtr get_tracking_id(HailoDetectionPtr detection)
{
    for (auto obj : detection->get_objects_typed(HAILO_UNIQUE_ID))
    {
        HailoUniqueIDPtr id = std::dynamic_pointer_cast<HailoUniqueID>(obj);
        if (id->get_mode() == TRACKING_ID)
        {
            return id;
        }
    }
    return nullptr;
}

/**
* @brief Returns a boolean box is invalid cause it has nan value.
* 
* @param box HailoBBox
* @return boolean indicating if box has nan value.
*/
bool box_contains_nan(HailoBBox box)
{
    return (std::isnan(box.xmin()) && std::isnan(box.ymin()) && std::isnan(box.width()) && std::isnan(box.height()));
}

/**
 * @brief Returns a vector of HailoROIPtr to crop and resize.
 *        Specifically, this algorithm doesn't make any actual filter,
 *        it just returns all the available detections
 *
 * @param image The original picture (cv::Mat).
 * @param roi The main ROI of this picture.
 * @return std::vector<HailoROIPtr> vector of ROI's to crop and resize.
 */
std::vector<HailoROIPtr> all_detections(std::shared_ptr<HailoMat> image, HailoROIPtr roi)
{
    /**
     * For performance reasons, the cropper is limited to processing only a single detection per frame.
     * Other detections are ignored but will have an opportunity to be processed in subsequent frames.
     * However, if detections are always processed in the same order for every frame, 
     * and processing is restricted to the first detection in the list, 
     * the same detection may consistently be processed while others are perpetually ignored.
     * To address this, an aging algorithm based on track IDs is implemented. 
     * A map of track IDs to their "age" (i.e., the number of frames since they were last processed) is maintained.
     * This ensures fairness by prioritizing detections with the oldest track IDs for processing over time.
     */
    static std::unordered_map<int, int> track_ages; // Map to store track ID and its age
    std::vector<HailoROIPtr> crop_rois;

    // Get all detections.
    std::vector<HailoDetectionPtr> detections_ptrs = hailo_common::get_hailo_detections(roi);

    // Increment the age of all tracks
    for (auto &entry : track_ages) {
        entry.second++;
    }

    // Sort detections by track age (oldest first)
    std::sort(detections_ptrs.begin(), detections_ptrs.end(), [&](const HailoDetectionPtr &a, const HailoDetectionPtr &b) {
        auto tracking_obj_a = get_tracking_id(a);
        auto tracking_obj_b = get_tracking_id(b);

        if (tracking_obj_a && tracking_obj_b) {
            int track_id_a = tracking_obj_a->get_id();
            int track_id_b = tracking_obj_b->get_id();
            return track_ages[track_id_a] > track_ages[track_id_b];
        }
        return false; // If no tracking ID, do not prioritize
    });

    for (HailoDetectionPtr &detection : detections_ptrs)
    {
        if (!box_contains_nan(detection->get_bbox()))
        {
            crop_rois.emplace_back(detection);

            // Reset the age of the processed track
            auto tracking_obj = get_tracking_id(detection);
            if (tracking_obj) {
                int track_id = tracking_obj->get_id();
                track_ages[track_id] = 0;
            }

            // Limit to one detection
            break;
        }
    }

    // Remove old tracks that are no longer detected
    for (auto it = track_ages.begin(); it != track_ages.end();) {
        if (std::none_of(detections_ptrs.begin(), detections_ptrs.end(), [&](const HailoDetectionPtr &detection) {
                auto tracking_obj = get_tracking_id(detection);
                return tracking_obj && tracking_obj->get_id() == it->first;
            })) {
            it = track_ages.erase(it);
        } else {
            ++it;
        }
    }

    return crop_rois;
}
