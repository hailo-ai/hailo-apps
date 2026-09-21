/**
 * Copyright (c) 2021-2026 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
#include "common/nms.hpp"

namespace common
{

    float iou_calc(const HailoBBox &box_1, const HailoBBox &box_2)
    {
        // Calculate IOU between two detection boxes
        const float width_of_overlap_area = std::min(box_1.xmax(), box_2.xmax()) - std::max(box_1.xmin(), box_2.xmin());
        const float height_of_overlap_area = std::min(box_1.ymax(), box_2.ymax()) - std::max(box_1.ymin(), box_2.ymin());
        const float positive_width_of_overlap_area = std::max(width_of_overlap_area, 0.0f);
        const float positive_height_of_overlap_area = std::max(height_of_overlap_area, 0.0f);
        const float area_of_overlap = positive_width_of_overlap_area * positive_height_of_overlap_area;
        const float box_1_area = (box_1.ymax() - box_1.ymin()) * (box_1.xmax() - box_1.xmin());
        const float box_2_area = (box_2.ymax() - box_2.ymin()) * (box_2.xmax() - box_2.xmin());
        // The IOU is a ratio of how much the boxes overlap vs their size outside the overlap.
        // Boxes that are similar will have a higher overlap threshold.
        return area_of_overlap / (box_1_area + box_2_area - area_of_overlap);
    }

}
