#include "utils.hpp"
#include "toolbox.hpp"
#include "labels/coco_eighty.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>

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


// ─────────────────────────────────────────────────────────────────────────────
// Anchor-free (raw tensor) decode — yolo26n/s/m style
// ─────────────────────────────────────────────────────────────────────────────

namespace {

static constexpr float ANCHOR_FREE_INPUT_SIZE = 640.0f;

// Grid height → stride mapping
static int stride_for_grid(int grid_h) {
    if (grid_h == 80) return 8;
    if (grid_h == 40) return 16;
    if (grid_h == 20) return 32;
    return static_cast<int>(ANCHOR_FREE_INPUT_SIZE / grid_h);
}

static inline float clamp_sigmoid(float x) {
    x = std::max(-88.0f, std::min(88.0f, x));
    return 1.0f / (1.0f + std::exp(-x));
}

struct RawDet {
    float x1, y1, x2, y2, score;
    int   class_id;
};

static float box_iou(const RawDet &a, const RawDet &b) {
    float ix1 = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
    float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
    float ua = (a.x2-a.x1)*(a.y2-a.y1) + (b.x2-b.x1)*(b.y2-b.y1) - inter;
    return inter / (ua + 1e-6f);
}

static std::vector<RawDet> nms(std::vector<RawDet> &dets, float iou_thr) {
    std::vector<size_t> idx(dets.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b){ return dets[a].score > dets[b].score; });

    std::vector<bool> suppressed(dets.size(), false);
    std::vector<RawDet> keep;
    for (size_t i : idx) {
        if (suppressed[i]) continue;
        keep.push_back(dets[i]);
        for (size_t j : idx)
            if (j != i && !suppressed[j] && box_iou(dets[i], dets[j]) > iou_thr)
                suppressed[j] = true;
    }
    return keep;
}

} // namespace

std::vector<NamedBbox> decode_anchor_free(
    const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &outputs,
    const VisualizationParams &vis)
{
    // Sort tensors by grid height: C==4 → bbox ltrb, else → class logits
    std::map<int, const float*> bbox_by_h, cls_by_h;
    std::map<int, int> cls_ch_by_h;

    for (const auto &[ptr, info] : outputs) {
        int H = static_cast<int>(info.shape.height);
        int C = static_cast<int>(info.shape.features);
        const float *fp = reinterpret_cast<const float*>(ptr);
        if (C == 4)
            bbox_by_h[H] = fp;
        else {
            cls_by_h[H]    = fp;
            cls_ch_by_h[H] = C;
        }
    }

    std::vector<RawDet> candidates;

    for (auto &[grid_h, bbox_ptr] : bbox_by_h) {
        auto cls_it = cls_by_h.find(grid_h);
        if (cls_it == cls_by_h.end()) continue;

        const float *cls_ptr = cls_it->second;
        int num_cls  = cls_ch_by_h[grid_h];
        int stride   = stride_for_grid(grid_h);
        int grid_w   = grid_h; // square grid

        for (int gy = 0; gy < grid_h; ++gy) {
            for (int gx = 0; gx < grid_w; ++gx) {
                int si = gy * grid_w + gx;

                // Find best class
                const float *cls = cls_ptr + si * num_cls;
                int best_cls = 0;
                float best_logit = cls[0];
                for (int c = 1; c < num_cls; ++c)
                    if (cls[c] > best_logit) { best_logit = cls[c]; best_cls = c; }

                float score = clamp_sigmoid(best_logit);
                if (score < vis.score_thresh) continue;

                // Decode ltrb (values are in stride units)
                const float *b = bbox_ptr + si * 4;
                float cx = (gx + 0.5f) * stride;
                float cy = (gy + 0.5f) * stride;
                float x1 = std::max(0.0f, cx - b[0] * stride) / ANCHOR_FREE_INPUT_SIZE;
                float y1 = std::max(0.0f, cy - b[1] * stride) / ANCHOR_FREE_INPUT_SIZE;
                float x2 = std::min(ANCHOR_FREE_INPUT_SIZE, cx + b[2] * stride) / ANCHOR_FREE_INPUT_SIZE;
                float y2 = std::min(ANCHOR_FREE_INPUT_SIZE, cy + b[3] * stride) / ANCHOR_FREE_INPUT_SIZE;

                candidates.push_back({x1, y1, x2, y2, score, best_cls});
            }
        }
    }

    static constexpr float IOU_THRESHOLD = 0.45f;
    auto kept = nms(candidates, IOU_THRESHOLD);

    int max_draw = (vis.max_boxes_to_draw > 0) ? vis.max_boxes_to_draw : INT_MAX;
    std::vector<NamedBbox> result;
    for (int i = 0; i < static_cast<int>(kept.size()) && i < max_draw; ++i) {
        const auto &d = kept[i];
        hailo_bbox_float32_t hb;
        hb.y_min = d.y1; hb.x_min = d.x1;
        hb.y_max = d.y2; hb.x_max = d.x2;
        hb.score  = d.score;
        NamedBbox nb; nb.bbox = hb; nb.class_id = static_cast<size_t>(d.class_id);
        result.push_back(nb);
    }
    return result;
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

