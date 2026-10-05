/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 *
 * Community fork: local overlay drawing logic.
 **/

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <glib.h>
#include "overlay.hpp"
#include "overlay_utils.hpp"
#include "hailo_common.hpp"
#include "sprite_cache.hpp"
#include "style_config.hpp"

// TAPPAS 5.3.0 decoupled OpenCV types from HailoMat draw API.
// Provide inline shims so the same source compiles on both 5.1.x and 5.3.x.
#if __has_include("hailomat_internal.hpp")
#include "hailomat_internal.hpp"   // provides get_cv_matrices()
static inline hailo_point_t to_h(const cv::Point &p) { return {p.x, p.y}; }
static inline hailo_rect_t to_h(const cv::Rect &r) { return {r.x, r.y, r.width, r.height}; }
static inline hailo_size_t to_h(const cv::Size &s) { return {s.width, s.height}; }
static inline hailo_scalar_t to_h(const cv::Scalar &s) { return {s[0], s[1], s[2], s[3]}; }
#define GET_CV_MAT(hmat) get_cv_matrices(hmat)[0]
#else
// TAPPAS 5.1.x — cv:: types pass directly to HailoMat methods
static inline const cv::Point& to_h(const cv::Point &p) { return p; }
static inline const cv::Rect& to_h(const cv::Rect &r) { return r; }
static inline const cv::Size& to_h(const cv::Size &s) { return s; }
static inline const cv::Scalar& to_h(const cv::Scalar &s) { return s; }
#define GET_CV_MAT(hmat) (hmat).get_matrices()[0]
#endif

#define SPACE " "
#define TEXT_CLS_FONT_SCALE_FACTOR (0.0025f)
#define MINIMUM_TEXT_CLS_FONT_SCALE (0.5f)
#define TEXT_DEFAULT_HEIGHT (0.1f)
#define TEXT_FONT_FACTOR (0.12f)
#define MINIMAL_BOX_WIDTH_FOR_TEXT (10)
#define LANDMARKS_COLOR (cv::Scalar(255, 0, 0))
#define NO_GLOBAL_ID_COLOR (cv::Scalar(255, 0, 0))
#define GLOBAL_ID_COLOR (cv::Scalar(0, 255, 0))
#define DEFAULT_DETECTION_COLOR (cv::Scalar(255, 255, 255))
#define DEFAULT_TILE_COLOR (2)
#define NULL_COLOR_ID ((size_t)NULL_CLASS_ID)
#define DEFAULT_COLOR (cv::Scalar(255, 255, 255))
#define RGB2Y(R, G, B) CLIP((0.257 * (R) + 0.504 * (G) + 0.098 * (B)) + 16)
#define RGB2U(R, G, B) CLIP((-0.148 * (R)-0.291 * (G) + 0.439 * (B)) + 128)
#define RGB2V(R, G, B) CLIP((0.439 * (R)-0.368 * (G)-0.071 * (B)) + 128)

#define DEPTH_MIN_DISTANCE 0.5
#define DEPTH_MAX_DISTANCE 3

#define STATS_RING_SIZE 30

static const std::vector<cv::Scalar> tile_layer_color_table = {
    cv::Scalar(0, 0, 255), cv::Scalar(200, 100, 120), cv::Scalar(255, 0, 0), cv::Scalar(120, 0, 0), cv::Scalar(0, 0, 120)};

static const std::vector<cv::Scalar> color_table = {
    cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 255),
    cv::Scalar(255, 0, 255), cv::Scalar(255, 170, 0), cv::Scalar(255, 0, 170), cv::Scalar(0, 255, 170), cv::Scalar(170, 255, 0),
    cv::Scalar(170, 0, 255), cv::Scalar(0, 170, 255), cv::Scalar(255, 85, 0), cv::Scalar(85, 255, 0), cv::Scalar(0, 255, 85),
    cv::Scalar(0, 85, 255), cv::Scalar(85, 0, 255), cv::Scalar(255, 0, 85), cv::Scalar(255, 255, 255)};

static cv::Scalar get_color(size_t color_id)
{
    cv::Scalar color;
    if (NULL_COLOR_ID == color_id)
        color = DEFAULT_COLOR;
    else
        color = indexToColor(color_id);

    return color;
}

cv::Scalar indexToColor(size_t index)
{
    return color_table[index % color_table.size()];
}

std::string confidence_to_string(float confidence)
{
    int confidence_percentage = (confidence * 100);

    return std::to_string(confidence_percentage) + "%";
}

static overlay_status_t draw_classification(HailoMat &mat, HailoROIPtr roi, std::string text, uint number_of_classifications, size_t color_id = NULL_COLOR_ID)
{
    auto bbox = hailo_common::create_flattened_bbox(roi->get_bbox(), roi->get_scaling_bbox());
    int roi_xmin = bbox.xmin() * mat.native_width();
    int roi_ymin = bbox.ymin() * mat.native_height();
    int roi_width = mat.native_width() * bbox.width();
    int roi_height = mat.native_height() * bbox.height();
    auto text_position = cv::Point(roi_xmin, roi_ymin + (TEXT_DEFAULT_HEIGHT * number_of_classifications * roi_height) + log(roi_height));
    double font_scale = TEXT_CLS_FONT_SCALE_FACTOR * roi_width;
    font_scale = (font_scale < MINIMUM_TEXT_CLS_FONT_SCALE) ? MINIMUM_TEXT_CLS_FONT_SCALE : font_scale;
    mat.draw_text(text, to_h(text_position), font_scale, to_h(get_color(color_id)));
    return OVERLAY_STATUS_OK;
}

static std::string get_classification_text(HailoClassificationPtr result, bool show_confidence = true)
{
    std::string text;
    std::string label = result->get_label();
    std::string confidence;
    if (show_confidence)
        confidence = SPACE + confidence_to_string(result->get_confidence());
    text = label + confidence;
    return text;
}

// Forward declaration (defined below, needed by draw_landmarks)
static void draw_sprite(cv::Mat &frame, const cv::Rect &bbox, const cv::Mat &sprite_bgra);

static overlay_status_t draw_landmarks(HailoMat &hmat, HailoLandmarksPtr landmarks, HailoROIPtr roi,
                                       float landmark_point_radius,
                                       SpriteCache *sprite_cache = nullptr,
                                       const std::unordered_map<int, std::string> *keypoint_sprites = nullptr)
{
    HailoBBox bbox = roi->get_bbox();
    int thickness;
    std::vector<std::pair<int, int>> pairs = landmarks->get_pairs();
    int R = 0;
    std::vector<HailoPoint> points = landmarks->get_points();
    if (landmarks->get_landmarks_type() == "centerpose")
    {
        R = roi->get_bbox().height() * hmat.native_height() / 60;
    }

    float threshold = landmarks->get_threshold();
    // If threshold is 0, use a sensible default so low-confidence joints are hidden
    if (threshold <= 0.0f)
        threshold = 0.5f;

    // Determine which keypoint indices have sprites (for skipping skeleton lines)
    bool has_kp_sprites = sprite_cache && keypoint_sprites && !keypoint_sprites->empty();

    for (auto &pair : pairs)
    {
        if ((points.at(pair.first).confidence() >= threshold) &&
            (points.at(pair.second).confidence() >= threshold))
        {
            // Skip skeleton lines that connect two sprite-replaced keypoints
            if (has_kp_sprites &&
                keypoint_sprites->count(pair.first) && keypoint_sprites->count(pair.second))
                continue;

            uint x1 = ((points.at(pair.first).x() * bbox.width()) + bbox.xmin()) * hmat.native_width();
            uint y1 = ((points.at(pair.first).y() * bbox.height()) + bbox.ymin()) * hmat.native_height();

            uint x2 = ((points.at(pair.second).x() * bbox.width()) + bbox.xmin()) * hmat.native_width();
            uint y2 = ((points.at(pair.second).y() * bbox.height()) + bbox.ymin()) * hmat.native_height();

            cv::Point joint1 = cv::Point(x1, y1);
            cv::Point joint2 = cv::Point(x2, y2);

            thickness = (bbox.width() < 0.05) ? 1 : 2;
            hmat.draw_line(to_h(joint1), to_h(joint2), to_h(get_color(4)), thickness, cv::LINE_4);
        }
    }

    // Compute sprite size proportional to detection bbox (bbox_height / 6)
    int sprite_sz = has_kp_sprites ? std::max(8, (int)(bbox.height() * hmat.native_height() / 6)) : 0;

    for (int idx = 0; idx < (int)points.size(); idx++)
    {
        auto &point = points[idx];
        if (point.confidence() < threshold) continue;

        uint x = ((point.x() * bbox.width()) + bbox.xmin()) * hmat.native_width();
        uint y = ((point.y() * bbox.height()) + bbox.ymin()) * hmat.native_height();

        // Check for keypoint sprite
        if (has_kp_sprites) {
            auto kp_it = keypoint_sprites->find(idx);
            if (kp_it != keypoint_sprites->end()) {
                const cv::Mat *sprite = sprite_cache->get_sprite(kp_it->second, sprite_sz, sprite_sz);
                if (sprite) {
                    // Center sprite on keypoint
                    cv::Rect sprite_rect(x - sprite->cols / 2, y - sprite->rows / 2,
                                         sprite->cols, sprite->rows);
                    draw_sprite(GET_CV_MAT(hmat), sprite_rect, *sprite);
                    continue;  // skip default dot
                }
            }
        }

        auto center = cv::Point(x, y);
        hmat.draw_ellipse(to_h(center), to_h(cv::Size(R, R)), 0, 0, 360, to_h(get_color(7)), landmark_point_radius);
    }
    return OVERLAY_STATUS_OK;
}

static cv::Rect get_rect(HailoMat &mat, HailoDetectionPtr detection, HailoROIPtr roi)
{
    HailoBBox roi_bbox = hailo_common::create_flattened_bbox(roi->get_bbox(), roi->get_scaling_bbox());
    auto detection_bbox = detection->get_bbox();

    auto bbox_min = cv::Point(((detection_bbox.xmin() * roi_bbox.width()) + roi_bbox.xmin()) * mat.native_width(),
                              ((detection_bbox.ymin() * roi_bbox.height()) + roi_bbox.ymin()) * mat.native_height());
    auto bbox_max = cv::Point(((detection_bbox.xmax() * roi_bbox.width()) + roi_bbox.xmin()) * mat.native_width(),
                              ((detection_bbox.ymax() * roi_bbox.height()) + roi_bbox.ymin()) * mat.native_height());
    return cv::Rect(bbox_min, bbox_max);
}

static std::string get_detection_text(HailoDetectionPtr detection, bool show_confidence = true)
{
    std::string text;
    std::string label = detection->get_label();
    std::string confidence = confidence_to_string(detection->get_confidence());
    if (!show_confidence)
        text = label;
    else if (!label.empty())
    {
        text = label + SPACE + confidence;
    }
    else
    {
        text = confidence;
    }
    return text;
}

static overlay_status_t draw_tile(HailoMat &mat, HailoTileROIPtr tile)
{
    auto bbox = tile->get_bbox();
    auto bbox_min = cv::Point(bbox.xmin() * mat.width(), bbox.ymin() * mat.height());
    auto bbox_max = cv::Point(bbox.xmax() * mat.width(), bbox.ymax() * mat.height());
    cv::Rect rect(bbox_min, bbox_max);
    cv::Scalar color;
    uint tile_layer = tile->get_layer();
    if (tile_layer < tile_layer_color_table.size())
        color = tile_layer_color_table[tile_layer];
    else
        color = get_color(DEFAULT_TILE_COLOR);

    mat.draw_rectangle(to_h(rect), to_h(color));

    return OVERLAY_STATUS_OK;
}

static overlay_status_t draw_id(HailoMat &mat, HailoUniqueIDPtr &hailo_id, HailoROIPtr roi)
{
    std::string id_text = std::to_string(hailo_id->get_id());

    auto bbox = roi->get_bbox();
    auto bbox_min = cv::Point(bbox.xmin() * mat.native_width(), bbox.ymin() * mat.native_height());
    auto bbox_max = cv::Point(bbox.xmax() * mat.native_width(), bbox.ymax() * mat.native_height());
    auto bbox_width = bbox_max.x - bbox_min.x;
    auto color = get_color(NULL_CLASS_ID);

    double font_scale = TEXT_FONT_FACTOR * log(bbox_width);
    auto text_position = cv::Point(bbox_min.x + log(bbox_width), bbox_max.y - log(bbox_width));
    mat.draw_text(id_text, to_h(text_position), font_scale, to_h(color));
    return OVERLAY_STATUS_OK;
}

template <typename T>
void calc_destination_roi_and_resize_mask(cv::Mat &destinationROI, cv::Mat &image_planes, HailoROIPtr roi, HailoMaskPtr mask, cv::Mat &resized_mask_data, T data_ptr, int cv_type)
{
    if (mask->get_height() == 0 || mask->get_width() == 0) {
        return;
    }

    HailoBBox bbox = roi->get_bbox();
    int roi_xmin = bbox.xmin() * image_planes.cols;
    int roi_ymin = bbox.ymin() * image_planes.rows;
    int roi_width = image_planes.cols * bbox.width();
    int roi_height = image_planes.rows * bbox.height();

    roi_xmin = std::clamp(roi_xmin, 0, image_planes.cols);
    roi_ymin = std::clamp(roi_ymin, 0, image_planes.rows);
    roi_width = std::clamp(roi_width, 0, image_planes.cols - roi_xmin);
    roi_height = std::clamp(roi_height, 0, image_planes.rows - roi_ymin);

    cv::Mat mat_data = cv::Mat(mask->get_height(), mask->get_width(), cv_type, (uint8_t *)data_ptr.data());
    cv::resize(mat_data, resized_mask_data, cv::Size(roi_width, roi_height), 0, 0, cv::INTER_LINEAR);

    cv::Rect roi_rect(cv::Point(roi_xmin, roi_ymin), cv::Size(roi_width, roi_height));
    destinationROI = image_planes(roi_rect);
}

static overlay_status_t draw_depth_mask(cv::Mat &image_planes, HailoDepthMaskPtr mask, HailoROIPtr roi, const uint mask_overlay_n_threads)
{
    cv::Mat resized_mask_data;
    cv::Mat destinationROI;
    calc_destination_roi_and_resize_mask(destinationROI, image_planes, roi, mask, resized_mask_data, mask->get_data(), CV_32F);

    float min = DEPTH_MIN_DISTANCE;
    float max = DEPTH_MAX_DISTANCE;

    double min_val;
    double max_val;
    cv::Point min_loc;
    cv::Point max_loc;

    cv::minMaxLoc(resized_mask_data, &min_val, &max_val, &min_loc, &max_loc);

    if (max < max_val)
        max = max_val;
    if (min > min_val)
        min = min_val;

    resized_mask_data = (resized_mask_data - min) / (max - min);

    if (mask_overlay_n_threads > 0)
        cv::setNumThreads(mask_overlay_n_threads);

    cv::parallel_for_(cv::Range(0, destinationROI.rows * destinationROI.cols), ParallelPixelDepthMask(destinationROI.data, resized_mask_data.data, mask->get_transparency(), image_planes.cols, destinationROI.cols));

    return OVERLAY_STATUS_OK;
}

static overlay_status_t
draw_class_mask(cv::Mat &image_planes, HailoClassMaskPtr mask, HailoROIPtr roi, const uint mask_overlay_n_threads)
{
    cv::Mat resized_mask_data;
    cv::Mat destinationROI;
    calc_destination_roi_and_resize_mask(destinationROI, image_planes, roi, mask, resized_mask_data, mask->get_data(), CV_8UC1);

    if (mask_overlay_n_threads > 0)
        cv::setNumThreads(mask_overlay_n_threads);

    cv::parallel_for_(cv::Range(0, destinationROI.rows * destinationROI.cols), ParallelPixelClassMask(destinationROI.data, resized_mask_data.data, mask->get_transparency(), image_planes.cols, destinationROI.cols));

    return OVERLAY_STATUS_OK;
}

static overlay_status_t draw_conf_class_mask(cv::Mat &image_planes, HailoConfClassMaskPtr mask, HailoROIPtr roi, const uint mask_overlay_n_threads)
{
    cv::Mat resized_mask_data;
    cv::Mat destinationROI;
    calc_destination_roi_and_resize_mask(destinationROI, image_planes, roi, mask, resized_mask_data, mask->get_data(), CV_32F);

    cv::Scalar mask_color = indexToColor(mask->get_class_id());

    if (mask_overlay_n_threads > 0)
        cv::setNumThreads(mask_overlay_n_threads);

    cv::parallel_for_(cv::Range(0, destinationROI.rows * destinationROI.cols), ParallelPixelClassConfMask(destinationROI.data, resized_mask_data.data, mask->get_transparency(), image_planes.cols, destinationROI.cols, mask_color));

    return OVERLAY_STATUS_OK;
}

static bool try_get_custom_color(HailoDetectionPtr det, cv::Scalar &out)
{
    for (auto &obj : det->get_objects()) {
        if (obj->get_type() != HAILO_CLASSIFICATION) continue;
        auto cls = std::dynamic_pointer_cast<HailoClassification>(obj);
        if (!cls || cls->get_classification_type() != "overlay_color") continue;
        // Fast path: packed 0xRRGGBB in class_id
        int cid = cls->get_class_id();
        if (cid > 0) {
            out = cv::Scalar((cid >> 16) & 0xFF, (cid >> 8) & 0xFF, cid & 0xFF);
            return true;
        }
        // Fallback: parse "R,G,B" from label
        int r, g, b;
        if (sscanf(cls->get_label().c_str(), "%d,%d,%d", &r, &g, &b) == 3) {
            out = cv::Scalar(r, g, b);
            return true;
        }
    }
    return false;
}

static std::string try_get_sprite_key(HailoDetectionPtr det)
{
    for (auto &obj : det->get_objects()) {
        if (obj->get_type() != HAILO_CLASSIFICATION) continue;
        auto cls = std::dynamic_pointer_cast<HailoClassification>(obj);
        if (cls && cls->get_classification_type() == "overlay_sprite")
            return cls->get_label();
    }
    return "";
}

static void draw_sprite(cv::Mat &frame, const cv::Rect &bbox, const cv::Mat &sprite_bgra)
{
    if (sprite_bgra.empty() || sprite_bgra.channels() != 4) return;

    // Clip bbox to frame bounds
    int x0 = std::max(bbox.x, 0);
    int y0 = std::max(bbox.y, 0);
    int x1 = std::min(bbox.x + sprite_bgra.cols, frame.cols);
    int y1 = std::min(bbox.y + sprite_bgra.rows, frame.rows);
    if (x0 >= x1 || y0 >= y1) return;

    int sx = x0 - bbox.x;
    int sy = y0 - bbox.y;
    int w = x1 - x0;
    int h = y1 - y0;

    for (int row = 0; row < h; row++) {
        const cv::Vec4b *sptr = sprite_bgra.ptr<cv::Vec4b>(sy + row) + sx;
        uchar *dptr = frame.ptr<uchar>(y0 + row) + x0 * frame.channels();
        int channels = frame.channels();
        for (int col = 0; col < w; col++) {
            uchar alpha = sptr[col][3];
            if (alpha == 0) {
                dptr += channels;
                continue;
            }
            if (alpha == 255) {
                dptr[0] = sptr[col][2]; // B->R (BGR to RGB)
                dptr[1] = sptr[col][1]; // G
                dptr[2] = sptr[col][0]; // R->B
            } else {
                uint a = alpha;
                uint inv_a = 255 - a;
                dptr[0] = (sptr[col][2] * a + dptr[0] * inv_a) / 255;
                dptr[1] = (sptr[col][1] * a + dptr[1] * inv_a) / 255;
                dptr[2] = (sptr[col][0] * a + dptr[2] * inv_a) / 255;
            }
            dptr += channels;
        }
    }
}

overlay_status_t draw_all(HailoMat &hmat, HailoROIPtr roi, const OverlayParams &params)
{
    overlay_status_t ret = OVERLAY_STATUS_UNINITIALIZED;
    uint number_of_classifications = 0;
    cv::Mat &mat = GET_CV_MAT(hmat);
    for (auto obj : roi->get_objects())
    {
        switch (obj->get_type())
        {
        case HAILO_DETECTION:
        {
            HailoDetectionPtr detection = std::dynamic_pointer_cast<HailoDetection>(obj);

            // Confidence filter
            if (detection->get_confidence() < params.min_confidence)
                continue;

            // Label filter
            const std::string &label = detection->get_label();
            if (params.show_labels_set && params.show_labels_set->find(label) == params.show_labels_set->end())
                continue;
            if (params.hide_labels_set && params.hide_labels_set->find(label) != params.hide_labels_set->end())
                continue;

            // Style config lookup
            const StyleEntry *style = nullptr;
            if (params.style_config) {
                auto *sc = static_cast<StyleConfig*>(params.style_config);
                style = sc->lookup(label, detection->get_class_id());
            }

            cv::Scalar color = NO_GLOBAL_ID_COLOR;
            std::string text = "";
            if (params.local_gallery)
            {
                auto global_ids = hailo_common::get_hailo_global_id(detection);
                if (global_ids.size() > 1)
                    std::cerr << "ERROR: more than one global id in roi" << std::endl;
                if (global_ids.size() == 1)
                    color = GLOBAL_ID_COLOR;
            }
            else
            {
                color = get_color((size_t)detection->get_class_id());
                text = get_detection_text(detection, params.show_confidence);
            }

            // Color priority: overlay_color metadata > style config > default
            if (style && style->color[0] >= 0)
                color = style->color;
            if (params.use_custom_colors) {
                cv::Scalar custom;
                if (try_get_custom_color(detection, custom))
                    color = custom;
            }

            auto rect = get_rect(hmat, detection, roi);

            // Per-class visibility overrides from style config
            bool draw_bbox = params.show_bbox;
            bool draw_label = params.show_labels_text;
            bool draw_lm = params.show_landmarks;
            if (style) {
                if (style->show_bbox >= 0) draw_bbox = (style->show_bbox == 1);
                if (style->show_label >= 0) draw_label = (style->show_label == 1);
                if (style->show_landmarks >= 0) draw_lm = (style->show_landmarks == 1);
            }

            // Check if bbox sprite will replace bbox+text
            bool bbox_replaced_by_sprite = false;
            if (params.sprite_replace_bbox && params.sprite_cache) {
                std::string skey = try_get_sprite_key(detection);
                if (skey.empty() && style && !style->sprite_key.empty())
                    skey = style->sprite_key;
                if (!skey.empty()) {
                    auto *cache = static_cast<SpriteCache*>(params.sprite_cache);
                    if (cache->get_sprite(skey, rect.width, rect.height))
                        bbox_replaced_by_sprite = true;
                }
            }

            // Bbox
            if (draw_bbox && !bbox_replaced_by_sprite)
                hmat.draw_rectangle(to_h(rect), to_h(color));

            // Detection text
            if (draw_label && !bbox_replaced_by_sprite && !text.empty()) {
                auto text_position = cv::Point(rect.x - log(rect.width), rect.y - log(rect.width));
                float font_scale = (params.text_font_scale > 0)
                    ? params.text_font_scale
                    : TEXT_FONT_FACTOR * log(rect.width);

                cv::Scalar text_col = color;
                if (style && style->text_color[0] >= 0)
                    text_col = style->text_color;

                if (params.text_background) {
                    int baseline = 0;
                    cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
                        font_scale, 1, &baseline);
                    cv::Rect bg_rect(text_position.x, text_position.y - text_size.height,
                        text_size.width, text_size.height + baseline);
                    cv::rectangle(GET_CV_MAT(hmat), bg_rect, cv::Scalar(0, 0, 0), cv::FILLED);
                }

                hmat.draw_text(text, to_h(text_position), font_scale, to_h(text_col));
            }

            // Recurse into sub-objects (with landmark/keypoint sprite overrides)
            {
                const std::unordered_map<int, std::string> *kp_sprites = nullptr;
                if (style && !style->keypoint_sprites.empty())
                    kp_sprites = &style->keypoint_sprites;

                if (draw_lm == params.show_landmarks && !kp_sprites) {
                    ret = draw_all(hmat, detection, params);
                } else {
                    OverlayParams sub_params = params;
                    sub_params.show_landmarks = draw_lm;
                    sub_params.keypoint_sprites = kp_sprites;
                    ret = draw_all(hmat, detection, sub_params);
                }
            }

            // Sprite rendering on bbox
            if (params.sprite_cache) {
                std::string sprite_key = try_get_sprite_key(detection);
                // Fallback to style config sprite_key
                if (sprite_key.empty() && style && !style->sprite_key.empty())
                    sprite_key = style->sprite_key;
                if (!sprite_key.empty()) {
                    auto *cache = static_cast<SpriteCache*>(params.sprite_cache);
                    const cv::Mat *sprite = cache->get_sprite(sprite_key, rect.width, rect.height);
                    if (sprite)
                        draw_sprite(mat, rect, *sprite);
                }
            }
            break;
        }
        case HAILO_CLASSIFICATION:
        {
            HailoClassificationPtr classification = std::dynamic_pointer_cast<HailoClassification>(obj);
            const std::string &cls_type = classification->get_classification_type();
            // Skip metadata classifications not meant for display
            if (cls_type == "overlay_color" || cls_type == "overlay_sprite" ||
                cls_type == "overlay_arrow" || cls_type == "overlay_text")
                break;

            number_of_classifications++;
            if (cls_type == "tracking")
            {
                std::string text = get_classification_text(classification, false);
                if (text == "lost")
                    ret = draw_classification(hmat, roi, text, number_of_classifications, 0);
                else if (text == "new")
                    ret = draw_classification(hmat, roi, text, number_of_classifications, 1);
                else if (text == "tracked")
                    ret = draw_classification(hmat, roi, text, number_of_classifications, 2);
            }
            else
            {
                std::string text = get_classification_text(classification, params.show_confidence);
                ret = draw_classification(hmat, roi, text, number_of_classifications);
            }
            break;
        }
        case HAILO_LANDMARKS:
        {
            if (params.show_landmarks) {
                HailoLandmarksPtr landmarks = std::dynamic_pointer_cast<HailoLandmarks>(obj);
                auto *kp_sprites = static_cast<const std::unordered_map<int, std::string>*>(params.keypoint_sprites);
                SpriteCache *cache = params.sprite_cache ? static_cast<SpriteCache*>(params.sprite_cache) : nullptr;
                draw_landmarks(hmat, landmarks, roi, params.landmark_point_radius, cache, kp_sprites);
            }
            break;
        }
        case HAILO_TILE:
        {
            HailoTileROIPtr tile = std::dynamic_pointer_cast<HailoTileROI>(obj);
            draw_tile(hmat, tile);
            draw_all(hmat, tile, params);
            break;
        }
        case HAILO_UNIQUE_ID:
        {
            if (params.show_tracking_id) {
                HailoUniqueIDPtr id = std::dynamic_pointer_cast<HailoUniqueID>(obj);
                if ((params.local_gallery && id->get_mode() == GLOBAL_ID) || (!params.local_gallery && id->get_mode() == TRACKING_ID))
                    draw_id(hmat, id, roi);
            }
            break;
        }
        case HAILO_DEPTH_MASK:
        {
            HailoDepthMaskPtr mask = std::dynamic_pointer_cast<HailoDepthMask>(obj);
            draw_depth_mask(mat, mask, roi, params.mask_overlay_n_threads);
            break;
        }
        case HAILO_CLASS_MASK:
        {
            HailoClassMaskPtr mask = std::dynamic_pointer_cast<HailoClassMask>(obj);
            draw_class_mask(mat, mask, roi, params.mask_overlay_n_threads);
            break;
        }
        case HAILO_CONF_CLASS_MASK:
        {
            HailoConfClassMaskPtr mask = std::dynamic_pointer_cast<HailoConfClassMask>(obj);
            draw_conf_class_mask(mat, mask, roi, params.mask_overlay_n_threads);
            break;
        }
        default:
            break;
        }
    }
    ret = OVERLAY_STATUS_OK;
    return ret;
}

void draw_stats_overlay(HailoMat &hmat, HailoROIPtr roi,
                        uint64_t *timestamps, int &index, int &count)
{
    // Record timestamp
    uint64_t now = g_get_monotonic_time();
    timestamps[index] = now;
    index = (index + 1) % STATS_RING_SIZE;
    if (count < STATS_RING_SIZE)
        count++;

    // Calculate FPS
    float fps = 0.0f;
    if (count > 1) {
        int oldest = (index - count + STATS_RING_SIZE) % STATS_RING_SIZE;
        int newest = (index - 1 + STATS_RING_SIZE) % STATS_RING_SIZE;
        uint64_t elapsed = timestamps[newest] - timestamps[oldest];
        if (elapsed > 0)
            fps = (count - 1) * 1000000.0f / elapsed;
    }

    // Count detections
    size_t num_objects = hailo_common::get_hailo_detections(roi).size();

    // Format text
    char buf[64];
    snprintf(buf, sizeof(buf), "FPS: %.0f | Objects: %zu", fps, num_objects);
    std::string text(buf);

    // Draw background + text at top-left
    double font_scale = 0.6;
    int font_thickness = 1;
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
        font_scale, font_thickness, &baseline);
    cv::Point text_pos(10, 10 + text_size.height);
    cv::Rect bg_rect(8, 8, text_size.width + 4, text_size.height + baseline + 4);
    cv::rectangle(GET_CV_MAT(hmat), bg_rect, cv::Scalar(0, 0, 0), cv::FILLED);
    hmat.draw_text(text, to_h(text_pos), font_scale, to_h(cv::Scalar(255, 255, 255)));
}

void face_blur(HailoMat &hmat, HailoROIPtr roi)
{
    for (auto detection : hailo_common::get_hailo_detections(roi))
    {
        if (detection->get_label() == "face")
        {
            HailoBBox roi_bbox = hailo_common::create_flattened_bbox(roi->get_bbox(), roi->get_scaling_bbox());
            auto detection_bbox = detection->get_bbox();
            auto xmin = std::clamp<int>(((detection_bbox.xmin() * roi_bbox.width()) + roi_bbox.xmin()) * hmat.native_width(), 0, hmat.native_width());
            auto ymin = std::clamp<int>(((detection_bbox.ymin() * roi_bbox.height()) + roi_bbox.ymin()) * hmat.native_height(), 0, hmat.native_height());
            auto xmax = std::clamp<int>(((detection_bbox.xmax() * roi_bbox.width()) + roi_bbox.xmin()) * hmat.native_width(), 0, hmat.native_width());
            auto ymax = std::clamp<int>(((detection_bbox.ymax() * roi_bbox.height()) + roi_bbox.ymin()) * hmat.native_height(), 0, hmat.native_height());
            auto rect = cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax));
            hmat.blur(to_h(rect), to_h(cv::Size(13, 13)));

            roi->remove_objects_typed(HAILO_LANDMARKS);
        }
        else
        {
            face_blur(hmat, detection);
        }
    }
}

static bool parse_kv_float(const std::string &label, const std::string &key, float &out) {
    std::string needle = key + ":";
    auto pos = label.find(needle);
    if (pos == std::string::npos) return false;
    try {
        out = std::stof(label.substr(pos + needle.size()));
    } catch (...) { return false; }
    return true;
}

static bool parse_kv_string(const std::string &label, const std::string &key, std::string &out) {
    std::string needle = key + ":";
    auto pos = label.find(needle);
    if (pos == std::string::npos) return false;
    size_t start = pos + needle.size();
    size_t end = label.find(',', start);
    out = label.substr(start, end == std::string::npos ? std::string::npos : end - start);
    return true;
}

void draw_hud_overlay(HailoMat &hmat, HailoROIPtr roi)
{
    cv::Mat &frame = GET_CV_MAT(hmat);
    int w = frame.cols;
    int h = frame.rows;

    for (auto &obj : roi->get_objects()) {
        if (obj->get_type() != HAILO_CLASSIFICATION) continue;
        auto cls = std::dynamic_pointer_cast<HailoClassification>(obj);
        if (!cls) continue;

        const std::string &cls_type = cls->get_classification_type();
        const std::string &label = cls->get_label();

        if (cls_type == "overlay_arrow") {
            float x = 0.5f, y = 0.5f, angle = 0.0f, len = 0.05f;
            float r = 0, g = 255, b = 0, t = 2;

            parse_kv_float(label, "x", x);
            parse_kv_float(label, "y", y);
            parse_kv_float(label, "angle", angle);
            parse_kv_float(label, "len", len);
            parse_kv_float(label, "r", r);
            parse_kv_float(label, "g", g);
            parse_kv_float(label, "b", b);
            parse_kv_float(label, "t", t);

            int px = (int)(x * w);
            int py = (int)(y * h);
            float rad = angle * (float)M_PI / 180.0f;
            int ex = px + (int)(len * h * cosf(rad));
            int ey = py - (int)(len * h * sinf(rad)); // screen y is inverted

            cv::arrowedLine(frame, cv::Point(px, py), cv::Point(ex, ey),
                            cv::Scalar((int)r, (int)g, (int)b),
                            (int)t, cv::LINE_AA, 0, 0.3);

        } else if (cls_type == "overlay_text") {
            float x = 0.5f, y = 0.5f;
            float r = 255, g = 255, b = 255, scale = 0.6f, bg = 0;
            std::string text;

            parse_kv_float(label, "x", x);
            parse_kv_float(label, "y", y);
            parse_kv_string(label, "text", text);
            parse_kv_float(label, "r", r);
            parse_kv_float(label, "g", g);
            parse_kv_float(label, "b", b);
            parse_kv_float(label, "scale", scale);
            parse_kv_float(label, "bg", bg);

            if (text.empty()) continue;

            cv::Point pos((int)(x * w), (int)(y * h));
            cv::Scalar color((int)r, (int)g, (int)b);

            if (bg > 0.5f) {
                int baseline = 0;
                cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
                    scale, 1, &baseline);
                cv::Rect bg_rect(pos.x - 2, pos.y - text_size.height - 2,
                    text_size.width + 4, text_size.height + baseline + 4);
                // Clip to frame
                bg_rect &= cv::Rect(0, 0, w, h);
                cv::rectangle(frame, bg_rect, cv::Scalar(0, 0, 0), cv::FILLED);
            }

            cv::putText(frame, text, pos, cv::FONT_HERSHEY_SIMPLEX,
                        scale, color, 1, cv::LINE_AA);
        }
    }
}
