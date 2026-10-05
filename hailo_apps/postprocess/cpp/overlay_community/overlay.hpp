/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 *
 * Community fork: local overlay drawing logic.
 **/
#pragma once

#include <map>
#include <vector>
#include <string>
#include <unordered_set>
#include "hailo_objects.hpp"
#include "hailomat.hpp"

typedef enum
{
    OVERLAY_STATUS_UNINITIALIZED = -1,
    OVERLAY_STATUS_OK,

} overlay_status_t;

struct OverlayParams {
    float landmark_point_radius;
    bool show_confidence;
    bool local_gallery;
    uint mask_overlay_n_threads;
    // Phase 1: Visibility controls
    bool show_bbox;
    bool show_labels_text;
    bool show_landmarks;
    bool show_tracking_id;
    float min_confidence;
    bool text_background;
    float text_font_scale;      // 0 = auto
    bool stats_overlay;
    const std::unordered_set<std::string> *show_labels_set;  // nullptr = show all
    const std::unordered_set<std::string> *hide_labels_set;  // nullptr = hide none
    // Phase 2: Custom colors
    bool use_custom_colors;
    // Phase 3: Sprite/stamp and style config (opaque pointers, nullptr = disabled)
    void *sprite_cache;         // SpriteCache*
    void *style_config;         // StyleConfig*
    bool sprite_replace_bbox;   // when true, bbox sprite replaces bbox+text
    const void *keypoint_sprites;   // const std::unordered_map<int,std::string>* from current style, nullptr = none
};

__BEGIN_DECLS
overlay_status_t draw_all(HailoMat &hmat, HailoROIPtr roi, const OverlayParams &params);
void face_blur(HailoMat &mat, HailoROIPtr roi);
void draw_stats_overlay(HailoMat &hmat, HailoROIPtr roi,
                        uint64_t *timestamps, int &index, int &count);
void draw_hud_overlay(HailoMat &hmat, HailoROIPtr roi);

cv::Scalar indexToColor(size_t index);

__END_DECLS
