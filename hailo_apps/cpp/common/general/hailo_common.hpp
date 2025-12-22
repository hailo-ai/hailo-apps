/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
/* __BEGIN_DECLS should be used at the beginning of your declarations,
   so that C++ compilers don't mangle their names.  Use __END_DECLS at
   the end of C declarations. */

#pragma once
#include "hailo_objects.hpp"
// #include <stdlib.h>
// #include <string>
// #include <cstring>

#undef __BEGIN_DECLS
#undef __END_DECLS
#ifdef __cplusplus
#define __BEGIN_DECLS \
    extern "C"        \
    {
#define __END_DECLS }
#else
#define __BEGIN_DECLS /* empty */
#define __END_DECLS   /* empty */
#endif

namespace hailo_common
{

    inline void add_object(HailoROIPtr roi, HailoObjectPtr obj)
    {
        roi->add_object(obj);
    }

    inline void add_objects(HailoROIPtr roi, std::vector<HailoObjectPtr> objects)
    {
        for (HailoObjectPtr obj : objects)
        {
            add_object(roi, obj);
        }
    }

    inline void add_classification(HailoROIPtr roi, std::string type, std::string label, float confidence, int class_id = NULL_CLASS_ID)
    {
        add_object(roi,
                   std::make_shared<HailoClassification>(type, class_id, label, confidence));
    }

    inline HailoDetectionPtr add_detection(HailoROIPtr roi, HailoBBox bbox, std::string label, float confidence, int class_id = NULL_CLASS_ID)
    {
        HailoDetectionPtr detection = std::make_shared<HailoDetection>(bbox, class_id, label, confidence);
        detection->set_scaling_bbox(roi->get_bbox());
        add_object(roi, detection);
        return detection;
    }

    inline void add_detections(HailoROIPtr roi, std::vector<HailoDetection> detections)
    {
        for (auto det : detections)
        {
            add_object(roi, std::make_shared<HailoDetection>(det));
        }
    }

    inline void add_detection_pointers(HailoROIPtr roi, std::vector<HailoDetectionPtr> detections)
    {
        for (auto det : detections)
        {
            det->set_scaling_bbox(roi->get_bbox());
            roi->add_object(det);
        }
    }

    inline void remove_objects(HailoROIPtr roi, std::vector<HailoObjectPtr> objects)
    {
        for (HailoObjectPtr obj : objects)
        {
            roi->remove_object(obj);
        }
    }

    inline void remove_detections(HailoROIPtr roi, std::vector<HailoDetectionPtr> objects)
    {
        for (HailoObjectPtr obj : objects)
        {
            roi->remove_object(obj);
        }
    }

    inline bool has_classifications(HailoROIPtr roi, std::string classification_type)
    {
        for (auto obj : roi->get_objects_typed(HAILO_CLASSIFICATION))
        {
            HailoClassificationPtr classification = std::dynamic_pointer_cast<HailoClassification>(obj);
            if (classification_type.compare(classification->get_classification_type()) == 0)
            {
                return true;
            }
        }
        return false;
    }

    inline void remove_classifications(HailoROIPtr roi, std::string classification_type)
    {
        std::vector<HailoObjectPtr> classifications;
        for (auto obj : roi->get_objects_typed(HAILO_CLASSIFICATION))
        {
            HailoClassificationPtr classification = std::dynamic_pointer_cast<HailoClassification>(obj);
            if (classification_type.compare(classification->get_classification_type()) == 0)
            {
                classifications.push_back(classification);
            }
        }
        remove_objects(roi, classifications);
    }

    inline std::vector<HailoDetectionPtr> get_hailo_detections(HailoROIPtr roi)
    {
        std::vector<HailoObjectPtr> objects = roi->get_objects_typed(HAILO_DETECTION);
        std::vector<HailoDetectionPtr> detections;

        for (auto obj : objects)
        {
            detections.emplace_back(std::dynamic_pointer_cast<HailoDetection>(obj));
        }
        return detections;
    }

    inline std::vector<HailoTileROIPtr> get_hailo_tiles(HailoROIPtr roi)
    {
        std::vector<HailoObjectPtr> objects = roi->get_objects_typed(HAILO_TILE);
        std::vector<HailoTileROIPtr> tiles;

        for (auto obj : objects)
        {
            tiles.emplace_back(std::dynamic_pointer_cast<HailoTileROI>(obj));
        }
        return tiles;
    }

    inline std::vector<HailoClassificationPtr> get_hailo_classifications(HailoROIPtr roi, std::string classification_type = "")
    {
        std::vector<HailoObjectPtr> objects = roi->get_objects_typed(HAILO_CLASSIFICATION);
        std::vector<HailoClassificationPtr> classifications;

        for (auto obj : objects)
        {
            HailoClassificationPtr classification = std::dynamic_pointer_cast<HailoClassification>(obj);
            if (classification_type.empty() || classification_type.compare(classification->get_classification_type()) == 0)
            {
                classifications.emplace_back(std::dynamic_pointer_cast<HailoClassification>(obj));
            }
        }
        return classifications;
    }

    inline std::vector<HailoUniqueIDPtr> get_hailo_unique_id(HailoROIPtr roi)
    {
        std::vector<HailoObjectPtr> objects = roi->get_objects_typed(HAILO_UNIQUE_ID);
        std::vector<HailoUniqueIDPtr> unique_ids;

        for (auto obj : objects)
        {
            unique_ids.emplace_back(std::dynamic_pointer_cast<HailoUniqueID>(obj));
        }
        return unique_ids;
    }

    inline std::vector<HailoUniqueIDPtr> get_hailo_unique_id_by_mode(HailoROIPtr roi, hailo_unique_id_mode_t mode)
    {
        std::vector<HailoObjectPtr> objects = roi->get_objects_typed(HAILO_UNIQUE_ID);
        std::vector<HailoUniqueIDPtr> unique_ids;

        for (auto obj : objects)
        {
            HailoUniqueIDPtr unique_id = std::dynamic_pointer_cast<HailoUniqueID>(obj);
            if(unique_id->get_mode() == mode)
            {
                unique_ids.emplace_back(unique_id);
            }
        }
        return unique_ids;
    }

    inline std::vector<HailoUniqueIDPtr> get_hailo_track_id(HailoROIPtr roi)
    {
        return get_hailo_unique_id_by_mode(roi, TRACKING_ID);
    }

    inline std::vector<HailoUniqueIDPtr> get_hailo_global_id(HailoROIPtr roi)
    {
        return get_hailo_unique_id_by_mode(roi, GLOBAL_ID);
    }

    inline std::vector<HailoLandmarksPtr> get_hailo_landmarks(HailoROIPtr roi)
    {
        std::vector<HailoObjectPtr> objects = roi->get_objects_typed(HAILO_LANDMARKS);
        std::vector<HailoLandmarksPtr> landmarks;

        for (auto obj : objects)
        {
            landmarks.emplace_back(std::dynamic_pointer_cast<HailoLandmarks>(obj));
        }
        return landmarks;
    }

    inline std::vector<HailoROIPtr> get_hailo_roi_instances(HailoROIPtr roi)
    {
        std::vector<HailoROIPtr> hailo_rois;

        for (auto obj : roi->get_objects())
        {
            HailoROIPtr hailo_roi = std::dynamic_pointer_cast<HailoROI>(obj);
            if (hailo_roi)
                hailo_rois.emplace_back(hailo_roi);
        }
        return hailo_rois;
    }

    /**
     * Flatten HailoBBox with parent HailoBBox.
     * re scales each bbox values (x,y,width,height min/max values) to match the parent scale.
     *
     * @param[in] bbox    HailoBBox, target bbox to flatten.
     * @param[in] parent_bbox  HailoBBox, parent bbox - to scale to.
     * @return void.
     */
    inline HailoBBox create_flattened_bbox(const HailoBBox &bbox, const HailoBBox &parent_bbox)
    {
        float xmin = parent_bbox.xmin() + bbox.xmin() * parent_bbox.width();
        float ymin = parent_bbox.ymin() + bbox.ymin() * parent_bbox.height();

        float width = bbox.width() * parent_bbox.width();
        float height = bbox.height() * parent_bbox.height();

        return HailoBBox(xmin, ymin, width, height);
    }

    /**
     * Flatten HailoROI subobjects and move them under parent HailoROI.
     * re scales each suboject bbox values to match the parent and move the under the ownership of the parent.
     *
     * @param[in] roi    HailoROIPtr, target roi to flatten.
     * @param[in] parent_roi  HailoROIPtr, parent roi - will take ownership of the subobjects of the target.
     * @param[in] filter_type    hailo_object_t, specific subobject type to flatten.
     * @return void.
     */
    inline void flatten_hailo_roi(HailoROIPtr roi, HailoROIPtr parent_roi, hailo_object_t filter_type)
    {
        std::vector<HailoObjectPtr> objects = roi->get_objects();
        for (uint index = 0; index < objects.size(); index++)
        {
            if (objects[index]->get_type() == filter_type)
            {
                HailoROIPtr sub_obj_roi = std::dynamic_pointer_cast<HailoROI>(objects[index]);
                sub_obj_roi->set_bbox(std::move(create_flattened_bbox(sub_obj_roi->get_bbox(), roi->get_bbox())));
                parent_roi->add_object(sub_obj_roi);
                roi->remove_object(index);
                objects.erase(objects.begin() + index);
                index--;
            }
        }
    }

    /**
     * Fixate Point(x,y) to the values of a given HailoBBox.
     * @param[in] hailo_roi    HailoROIPtr, target roi.
     * @param[in] point    HailoPoint, target point object.
     * @param[in] fixate_bbox  HailoBBox, target bbox - to fixate the point to.
     * @return New HailoPoint.
     */
    inline HailoPoint fixate_point_with_bbox(HailoROIPtr hailo_roi, HailoPoint &point, const HailoBBox &fixate_bbox)
    {
        HailoBBox current_bbox = hailo_roi->get_bbox();
        float xmin = current_bbox.xmin() + point.x() * current_bbox.width();
        float ymin = current_bbox.ymin() + point.y() * current_bbox.height();

        return HailoPoint((xmin-fixate_bbox.xmin())/fixate_bbox.width(), (ymin-fixate_bbox.ymin())/fixate_bbox.height());
    }

    /**
     * Fixate HailoROI landmarks subobject to the values of a given HailoBBox.
     * re place every landmark point to match new bbox, by changing the x and y values.
     *
     * @param[in] hailo_roi    HailoROIPtr, target roi.
     * @param[in] fixate_bbox  HailoBBox, target bbox - to fixate the landmarks to.
     * @return void.
     */
    inline void fixate_landmarks_with_bbox(HailoROIPtr hailo_roi, HailoBBox fixate_bbox)
    {
        std::vector<HailoLandmarksPtr> landmarks_ptrs = get_hailo_landmarks(hailo_roi);
        if (landmarks_ptrs.size() > 0)
        {
            HailoLandmarksPtr landmarks_obj = std::dynamic_pointer_cast<HailoLandmarks>(landmarks_ptrs[0]);
            std::vector<HailoPoint> decoded_points = landmarks_obj->get_points();
            std::vector<HailoPoint> new_points;
            new_points.reserve(decoded_points.size());

            for (size_t i = 0; i < decoded_points.size(); i++)
            {
                new_points.emplace_back(fixate_point_with_bbox(hailo_roi, decoded_points[i], fixate_bbox));
            }
            landmarks_obj->set_points(new_points);
        }
    }

}