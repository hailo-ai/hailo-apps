/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 *
 * Community fork: hailooverlay_community - a locally modifiable overlay element.
 **/
#pragma once

#include <gst/base/gstbasetransform.h>
#include <vector>
#include "hailo_objects.hpp"

G_BEGIN_DECLS

#define GST_TYPE_HAILO_OVERLAY_COMMUNITY (gst_hailooverlay_community_get_type())
#define GST_HAILO_OVERLAY_COMMUNITY(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_HAILO_OVERLAY_COMMUNITY, GstHailoOverlayCommunity))
#define GST_HAILO_OVERLAY_COMMUNITY_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_HAILO_OVERLAY_COMMUNITY, GstHailoOverlayCommunityClass))
#define GST_IS_HAILO_OVERLAY_COMMUNITY(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_HAILO_OVERLAY_COMMUNITY))
#define GST_IS_HAILO_OVERLAY_COMMUNITY_CLASS(obj) (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_HAILO_OVERLAY_COMMUNITY))

typedef struct _GstHailoOverlayCommunity GstHailoOverlayCommunity;
typedef struct _GstHailoOverlayCommunityClass GstHailoOverlayCommunityClass;

struct _GstHailoOverlayCommunity
{
    GstBaseTransform base_hailooverlay;
    // Existing
    gint line_thickness;
    gint font_thickness;
    gfloat landmark_point_radius;
    gboolean face_blur;
    gboolean show_confidence;
    gboolean local_gallery;
    guint mask_overlay_n_threads;
    // Phase 1: Visibility controls
    gboolean show_bbox;
    gboolean show_labels_text;
    gboolean show_landmarks;
    gboolean show_tracking_id;
    gfloat min_confidence;
    gchar *show_labels_str;
    gchar *hide_labels_str;
    gboolean text_background;
    gfloat text_font_scale;
    gboolean stats_overlay;
    // Phase 2: Custom colors
    gboolean use_custom_colors;
    // Phase 3: Config file paths and sprite options
    gchar *sprite_config_path;
    gchar *style_config_path;
    gboolean sprite_replace_bbox;
    // Internal state (not GObject properties)
    void *show_labels_set;      // std::unordered_set<std::string>*
    void *hide_labels_set;      // std::unordered_set<std::string>*
    void *sprite_cache;         // SpriteCache*
    void *style_config;         // StyleConfig*
    uint64_t stats_timestamps[30];
    int stats_index;
    int stats_count;
    // Phase 4: HUD overlay
    gboolean hud_overlay;
};

struct _GstHailoOverlayCommunityClass
{
    GstBaseTransformClass base_hailooverlay_class;
};

GType gst_hailooverlay_community_get_type(void);

G_END_DECLS
