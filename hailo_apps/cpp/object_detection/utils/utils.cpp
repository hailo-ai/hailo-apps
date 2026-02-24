#include "utils.hpp"
#include "toolbox.hpp"
#include "labels/coco_eighty.hpp"

using namespace hailo_utils;

std::vector<cv::Scalar> COLORS = {
    cv::Scalar(255,   0,   0),  // Red
    cv::Scalar(  0, 255,   0),  // Green
    cv::Scalar(  0,   0, 255),  // Blue
    cv::Scalar(255, 255,   0),  // Cyan
    cv::Scalar(255,   0, 255),  // Magenta
    cv::Scalar(  0, 255, 255),  // Yellow
    cv::Scalar(255, 128,   0),  // Orange
    cv::Scalar(128,   0, 128),  // Purple
    cv::Scalar(128, 128,   0),  // Olive
    cv::Scalar(128,   0, 255),  // Violet
    cv::Scalar(  0, 128, 255),  // Sky Blue
    cv::Scalar(255,   0, 128),  // Pink
    cv::Scalar(  0, 128,   0),  // Dark Green
    cv::Scalar(128, 128, 128),  // Gray
    cv::Scalar(255, 255, 255)   // White
};


void initialize_class_colors(std::unordered_map<int, cv::Scalar>& class_colors) {
    for (int cls = 0; cls <= 80; ++cls) {
        class_colors[cls] = COLORS[cls % COLORS.size()]; 
    }
}

cv::Rect get_bbox_coordinates(const hailo_bbox_float32_t& bbox, int frame_width, int frame_height) {
    int x1 = static_cast<int>(bbox.x_min * frame_width);
    int y1 = static_cast<int>(bbox.y_min * frame_height);
    int x2 = static_cast<int>(bbox.x_max * frame_width);
    int y2 = static_cast<int>(bbox.y_max * frame_height);
    return cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
}

void draw_label(cv::Mat& frame, const std::string& label, const cv::Point& top_left, const cv::Scalar& color) {
    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_TRIPLEX, 0.6, 1, &baseLine);
    int top = std::max(top_left.y, label_size.height);
    cv::rectangle(frame, cv::Point(top_left.x, top + baseLine), 
                  cv::Point(top_left.x + label_size.width, top - label_size.height), color, cv::FILLED);
    cv::putText(frame, label, cv::Point(top_left.x, top), cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
}

void draw_single_bbox(cv::Mat& frame, const NamedBbox& named_bbox, const cv::Scalar& color) {
    auto bbox_rect = get_bbox_coordinates(named_bbox.bbox, frame.cols, frame.rows);
    cv::rectangle(frame, bbox_rect, color, 2);

    const int cls_id = static_cast<int>(named_bbox.class_id);
    std::string score_str = std::to_string(named_bbox.bbox.score * 100).substr(0, 4) + "%";
    std::string label = common::coco_eighty[cls_id] + " " + score_str;
    draw_label(frame, label, bbox_rect.tl(), color);
}

void draw_bounding_boxes(cv::Mat& frame, const std::vector<NamedBbox>& bboxes) {
    std::unordered_map<int, cv::Scalar> class_colors;
    initialize_class_colors(class_colors);
    for (const auto& named_bbox : bboxes) {
        const auto& color = class_colors[named_bbox.class_id];
        draw_single_bbox(frame, named_bbox, color);
    }
}

void draw_bounding_boxes(cv::Mat &frame,
                         const std::vector<NamedBbox> &bboxes,
                         const VisualizationParams &vis)
{
    const size_t max_draw =
        (vis.max_boxes_to_draw > 0)
            ? std::min((size_t)vis.max_boxes_to_draw, bboxes.size())
            : bboxes.size();

    size_t drawn = 0;

    for (const auto &named_bbox : bboxes) {
        if (drawn >= max_draw)
            break;

        // Apply score threshold from visualization config
        if (named_bbox.bbox.score < vis.score_thresh)
            continue;

        const int class_id = static_cast<int>(named_bbox.class_id);
        const cv::Scalar color = COLORS[class_id % COLORS.size()];

        draw_single_bbox(frame, named_bbox, color);
        ++drawn;
    }
}


std::vector<NamedBbox> parse_nms_data(uint8_t* data, size_t max_class_count) {
    std::vector<NamedBbox> bboxes;
    size_t offset = 0;

    for (size_t class_id = 0; class_id < max_class_count; class_id++) {
        auto det_count = static_cast<uint32_t>(*reinterpret_cast<float32_t*>(data + offset));
        offset += sizeof(float32_t);

        for (size_t j = 0; j < det_count; j++) {
            hailo_bbox_float32_t bbox_data = *reinterpret_cast<hailo_bbox_float32_t*>(data + offset);
            offset += sizeof(hailo_bbox_float32_t);

            NamedBbox named_bbox;
            named_bbox.bbox = bbox_data;
            named_bbox.class_id = class_id + 1;
            bboxes.push_back(named_bbox);
        }
    }
    return bboxes;
}

