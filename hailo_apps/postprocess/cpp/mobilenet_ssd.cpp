/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 **/

#include <regex>
#include "mobilenet_ssd.hpp"
#include "hailo_nms_decode.hpp"

static void mobilenet_ssd_base(HailoROIPtr roi, std::map<uint8_t, std::string> &labels_dict)
{
    if (!roi->has_tensors())
    {
        return;
    }

    std::vector<HailoTensorPtr> tensors = roi->get_tensors();
    // find the nms tensor
    for (auto tensor : tensors)
    {
        if (std::regex_search(tensor->name(), std::regex("nms")))
        {
            auto post = HailoNMSDecode(tensor, labels_dict);
            auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
            hailo_common::add_detections(roi, detections);
            break;
        }
    }
}

void mobilenet_ssd(HailoROIPtr roi)
{
    mobilenet_ssd_base(roi, common::coco_ninety_classes);
}

void mobilenet_ssd_merged(HailoROIPtr roi)
{
    mobilenet_ssd_base(roi, common::coco_ninety_classes);
}

void mobilenet_ssd_visdrone(HailoROIPtr roi)
{
    mobilenet_ssd_base(roi, common::coco_visdrone_classes);
}

void filter(HailoROIPtr roi)
{
    mobilenet_ssd(roi);
}
