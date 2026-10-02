/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 *
 * hailooverlay_community - overlay element with confidence-based landmark filtering.
 **/
#include <gst/gst.h>
#include <gst/video/video.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <sstream>
#include <unordered_set>
#include "gsthailooverlay_community.hpp"
#include "image.hpp"
#include "overlay.hpp"
#include "gst_hailo_meta.hpp"
#include "sprite_cache.hpp"
#include "style_config.hpp"

GST_DEBUG_CATEGORY_STATIC(gst_hailooverlay_community_debug_category);
#define GST_CAT_DEFAULT gst_hailooverlay_community_debug_category

/* prototypes */

static void gst_hailooverlay_community_set_property(GObject *object,
                                          guint property_id, const GValue *value, GParamSpec *pspec);
static void gst_hailooverlay_community_get_property(GObject *object,
                                          guint property_id, GValue *value, GParamSpec *pspec);
static void gst_hailooverlay_community_dispose(GObject *object);
static void gst_hailooverlay_community_finalize(GObject *object);

static gboolean gst_hailooverlay_community_start(GstBaseTransform *trans);
static gboolean gst_hailooverlay_community_stop(GstBaseTransform *trans);
static GstFlowReturn gst_hailooverlay_community_transform_ip(GstBaseTransform *trans,
                                                   GstBuffer *buffer);

/* helpers */

static void parse_label_set(const gchar *csv, std::unordered_set<std::string> *out)
{
    out->clear();
    if (!csv || csv[0] == '\0') return;
    std::istringstream ss(csv);
    std::string token;
    while (std::getline(ss, token, ',')) {
        auto start = token.find_first_not_of(" ");
        auto end = token.find_last_not_of(" ");
        if (start != std::string::npos)
            out->insert(token.substr(start, end - start + 1));
    }
}

/* class initialization */

G_DEFINE_TYPE_WITH_CODE(GstHailoOverlayCommunity, gst_hailooverlay_community, GST_TYPE_BASE_TRANSFORM,
                        GST_DEBUG_CATEGORY_INIT(gst_hailooverlay_community_debug_category, "hailooverlay_community", 0,
                                                "debug category for hailooverlay_community element"));

enum
{
    PROP_0,
    PROP_LINE_THICKNESS,
    PROP_FONT_THICKNESS,
    PROP_LANDMARK_POINT_RADIUS,
    PROP_FACE_BLUR,
    PROP_SHOW_CONF,
    PROP_MASK_OVERLAY_N_THREADS,
    PROP_LOCAL_GALLERY,
    PROP_SHOW_BBOX,
    PROP_SHOW_LABELS_TEXT,
    PROP_SHOW_LANDMARKS,
    PROP_SHOW_TRACKING_ID,
    PROP_MIN_CONFIDENCE,
    PROP_SHOW_LABELS,
    PROP_HIDE_LABELS,
    PROP_TEXT_BACKGROUND,
    PROP_TEXT_FONT_SCALE,
    PROP_STATS_OVERLAY,
    PROP_USE_CUSTOM_COLORS,
    PROP_SPRITE_CONFIG,
    PROP_STYLE_CONFIG,
    PROP_SPRITE_REPLACE_BBOX,
    PROP_HUD_OVERLAY,
};

static void
gst_hailooverlay_community_class_init(GstHailoOverlayCommunityClass *klass)
{
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
    GstBaseTransformClass *base_transform_class =
        GST_BASE_TRANSFORM_CLASS(klass);

    const char *description = "Draws post-processing results for networks inferred by hailonet elements."
                              "\n\t\t\t   "
                              "Draws classes contained by HailoROI objects attached to incoming frames.";
    // Overlay drawing (sprite blending, HUD overlay, text backgrounds) assumes
    // RGB-like pixel layout (>=3 interleaved bytes/pixel). NV12 (1 channel Y
    // plane) and YUY2 (2 bytes/pixel packed) are not interleaved RGB and
    // break those code paths (out-of-bounds writes on NV12, wrong colors /
    // position drift on YUY2), so only RGB/RGBA are advertised.
    gst_element_class_add_pad_template(GST_ELEMENT_CLASS(klass),
                                       gst_pad_template_new("src", GST_PAD_SRC, GST_PAD_ALWAYS,
                                                            gst_caps_from_string(GST_VIDEO_CAPS_MAKE("{ RGB, RGBA }"))));
    gst_element_class_add_pad_template(GST_ELEMENT_CLASS(klass),
                                       gst_pad_template_new("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
                                                            gst_caps_from_string(GST_VIDEO_CAPS_MAKE("{ RGB, RGBA }"))));

    gst_element_class_set_static_metadata(GST_ELEMENT_CLASS(klass),
                                          "hailooverlay_community - overlay element",
                                          "Hailo/Tools",
                                          description,
                                          "hailo.ai <contact@hailo.ai>");

    gobject_class->set_property = gst_hailooverlay_community_set_property;
    gobject_class->get_property = gst_hailooverlay_community_get_property;

    /* Existing properties */
    g_object_class_install_property(gobject_class, PROP_LINE_THICKNESS,
                                    g_param_spec_int("line-thickness", "line-thickness", "The thickness when drawing lines. Default 1.", 0, G_MAXINT, 1,
                                                     (GParamFlags)(GST_PARAM_MUTABLE_READY | G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class, PROP_FONT_THICKNESS,
                                    g_param_spec_int("font-thickness", "font-thickness", "The thickness when drawing text. Default 1.", 0, G_MAXINT, 1,
                                                     (GParamFlags)(GST_PARAM_MUTABLE_READY | G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class, PROP_FACE_BLUR,
                                    g_param_spec_boolean("face-blur", "face-blur", "Whether to blur faces", false,
                                                         (GParamFlags)(GST_PARAM_CONTROLLABLE | G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class, PROP_SHOW_CONF,
                                    g_param_spec_boolean("show-confidence", "show-confidence", "Whether to display confidence on detections, classifications etc...", true,
                                                         (GParamFlags)(GST_PARAM_CONTROLLABLE | G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class, PROP_MASK_OVERLAY_N_THREADS,
                                    g_param_spec_uint("mask-overlay-n-threads", "mask-overlay-n-threads", "Number of threads to use for parallel mask drawing. Default 0 (Will use the default value OpenCV initializes - effected by the system capabilities).", 0, G_MAXUINT, 0,
                                                      (GParamFlags)(GST_PARAM_MUTABLE_READY | G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class, PROP_LOCAL_GALLERY,
                                    g_param_spec_boolean("local-gallery", "local-gallery", "Whether to display Identified and UnIdentified ROI's taken from the local gallery, as well as the Global ID they receive.", false,
                                                         (GParamFlags)(GST_PARAM_CONTROLLABLE | G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class, PROP_LANDMARK_POINT_RADIUS,
                                    g_param_spec_float("landmark-point-radius", "landmark-point-radius", "The radius of the points when drawing landmarks. Default 3.", 0, G_MAXFLOAT, 3,
                                                       (GParamFlags)(GST_PARAM_MUTABLE_READY | G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    /* Phase 1: Visibility controls */
    g_object_class_install_property(gobject_class, PROP_SHOW_BBOX,
                                    g_param_spec_boolean("show-bbox", "show-bbox", "Whether to draw bounding boxes.", true,
                                                         (GParamFlags)(GST_PARAM_CONTROLLABLE | G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class, PROP_SHOW_LABELS_TEXT,
                                    g_param_spec_boolean("show-labels-text", "show-labels-text", "Whether to draw detection label text.", true,
                                                         (GParamFlags)(GST_PARAM_CONTROLLABLE | G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class, PROP_SHOW_LANDMARKS,
                                    g_param_spec_boolean("show-landmarks", "show-landmarks", "Whether to draw landmarks/skeletons.", true,
                                                         (GParamFlags)(GST_PARAM_CONTROLLABLE | G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class, PROP_SHOW_TRACKING_ID,
                                    g_param_spec_boolean("show-tracking-id", "show-tracking-id", "Whether to draw tracking IDs.", true,
                                                         (GParamFlags)(GST_PARAM_CONTROLLABLE | G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class, PROP_MIN_CONFIDENCE,
                                    g_param_spec_float("min-confidence", "min-confidence", "Minimum confidence threshold to display a detection. Default 0.0 (show all).", 0.0f, 1.0f, 0.0f,
                                                       (GParamFlags)(GST_PARAM_CONTROLLABLE | G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class, PROP_SHOW_LABELS,
                                    g_param_spec_string("show-labels", "show-labels", "Comma-separated list of labels to show (empty = show all).", "",
                                                        (GParamFlags)(GST_PARAM_MUTABLE_PLAYING | G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class, PROP_HIDE_LABELS,
                                    g_param_spec_string("hide-labels", "hide-labels", "Comma-separated list of labels to hide (empty = hide none).", "",
                                                        (GParamFlags)(GST_PARAM_MUTABLE_PLAYING | G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class, PROP_TEXT_BACKGROUND,
                                    g_param_spec_boolean("text-background", "text-background", "Whether to draw a dark background behind label text.", false,
                                                         (GParamFlags)(GST_PARAM_CONTROLLABLE | G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class, PROP_TEXT_FONT_SCALE,
                                    g_param_spec_float("text-font-scale", "text-font-scale", "Fixed font scale for text. 0 = auto (default).", 0.0f, 10.0f, 0.0f,
                                                       (GParamFlags)(GST_PARAM_CONTROLLABLE | G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class, PROP_STATS_OVERLAY,
                                    g_param_spec_boolean("stats-overlay", "stats-overlay", "Whether to display FPS and object count overlay.", false,
                                                         (GParamFlags)(GST_PARAM_CONTROLLABLE | G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    /* Phase 2: Custom colors */
    g_object_class_install_property(gobject_class, PROP_USE_CUSTOM_COLORS,
                                    g_param_spec_boolean("use-custom-colors", "use-custom-colors", "Whether to use custom colors from overlay_color classification metadata.", false,
                                                         (GParamFlags)(GST_PARAM_CONTROLLABLE | G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    /* Phase 3: Config files */
    g_object_class_install_property(gobject_class, PROP_SPRITE_CONFIG,
                                    g_param_spec_string("sprite-config", "sprite-config", "Path to YAML file mapping sprite keys to PNG image paths.", "",
                                                        (GParamFlags)(GST_PARAM_MUTABLE_READY | G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class, PROP_STYLE_CONFIG,
                                    g_param_spec_string("style-config", "style-config", "Path to YAML file with per-class style overrides (color, visibility, sprite).", "",
                                                        (GParamFlags)(GST_PARAM_MUTABLE_READY | G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class, PROP_SPRITE_REPLACE_BBOX,
                                    g_param_spec_boolean("sprite-replace-bbox", "sprite-replace-bbox", "When true, a bbox sprite replaces the bounding box and label text instead of overlaying on top.", false,
                                                         (GParamFlags)(GST_PARAM_CONTROLLABLE | G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    /* Phase 4: HUD overlay */
    g_object_class_install_property(gobject_class, PROP_HUD_OVERLAY,
                                    g_param_spec_boolean("hud-overlay", "hud-overlay", "Whether to draw HUD elements (arrows, text) from overlay_arrow and overlay_text metadata.", false,
                                                         (GParamFlags)(GST_PARAM_CONTROLLABLE | G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    gobject_class->dispose = gst_hailooverlay_community_dispose;
    gobject_class->finalize = gst_hailooverlay_community_finalize;
    base_transform_class->start = GST_DEBUG_FUNCPTR(gst_hailooverlay_community_start);
    base_transform_class->stop = GST_DEBUG_FUNCPTR(gst_hailooverlay_community_stop);
    base_transform_class->transform_ip =
        GST_DEBUG_FUNCPTR(gst_hailooverlay_community_transform_ip);
}

static void
gst_hailooverlay_community_init(GstHailoOverlayCommunity *hailooverlay)
{
    hailooverlay->line_thickness = 1;
    hailooverlay->font_thickness = 1;
    hailooverlay->face_blur = false;
    hailooverlay->show_confidence = true;
    hailooverlay->local_gallery = false;
    hailooverlay->landmark_point_radius = 3;
    hailooverlay->mask_overlay_n_threads = 0;
    // Phase 1
    hailooverlay->show_bbox = true;
    hailooverlay->show_labels_text = true;
    hailooverlay->show_landmarks = true;
    hailooverlay->show_tracking_id = true;
    hailooverlay->min_confidence = 0.0f;
    hailooverlay->show_labels_str = g_strdup("");
    hailooverlay->hide_labels_str = g_strdup("");
    hailooverlay->text_background = false;
    hailooverlay->text_font_scale = 0.0f;
    hailooverlay->stats_overlay = false;
    // Phase 2
    hailooverlay->use_custom_colors = false;
    // Phase 3
    hailooverlay->sprite_config_path = g_strdup("");
    hailooverlay->style_config_path = g_strdup("");
    hailooverlay->sprite_replace_bbox = false;
    // Phase 4
    hailooverlay->hud_overlay = false;
    // Internal state
    hailooverlay->show_labels_set = new std::unordered_set<std::string>();
    hailooverlay->hide_labels_set = new std::unordered_set<std::string>();
    hailooverlay->sprite_cache = nullptr;
    hailooverlay->style_config = nullptr;
    hailooverlay->stats_index = 0;
    hailooverlay->stats_count = 0;
    memset(hailooverlay->stats_timestamps, 0, sizeof(hailooverlay->stats_timestamps));
}

void gst_hailooverlay_community_set_property(GObject *object, guint property_id,
                                   const GValue *value, GParamSpec *pspec)
{
    GstHailoOverlayCommunity *hailooverlay = GST_HAILO_OVERLAY_COMMUNITY(object);

    GST_DEBUG_OBJECT(hailooverlay, "set_property");

    switch (property_id)
    {
    case PROP_LINE_THICKNESS:
        hailooverlay->line_thickness = g_value_get_int(value);
        break;
    case PROP_FONT_THICKNESS:
        hailooverlay->font_thickness = g_value_get_int(value);
        break;
    case PROP_LANDMARK_POINT_RADIUS:
        hailooverlay->landmark_point_radius = g_value_get_float(value);
        break;
    case PROP_FACE_BLUR:
        hailooverlay->face_blur = g_value_get_boolean(value);
        break;
    case PROP_SHOW_CONF:
        hailooverlay->show_confidence = g_value_get_boolean(value);
        break;
    case PROP_MASK_OVERLAY_N_THREADS:
        hailooverlay->mask_overlay_n_threads = g_value_get_uint(value);
        break;
    case PROP_LOCAL_GALLERY:
        hailooverlay->local_gallery = g_value_get_boolean(value);
        break;
    case PROP_SHOW_BBOX:
        hailooverlay->show_bbox = g_value_get_boolean(value);
        break;
    case PROP_SHOW_LABELS_TEXT:
        hailooverlay->show_labels_text = g_value_get_boolean(value);
        break;
    case PROP_SHOW_LANDMARKS:
        hailooverlay->show_landmarks = g_value_get_boolean(value);
        break;
    case PROP_SHOW_TRACKING_ID:
        hailooverlay->show_tracking_id = g_value_get_boolean(value);
        break;
    case PROP_MIN_CONFIDENCE:
        hailooverlay->min_confidence = g_value_get_float(value);
        break;
    case PROP_SHOW_LABELS:
        g_free(hailooverlay->show_labels_str);
        hailooverlay->show_labels_str = g_value_dup_string(value);
        parse_label_set(hailooverlay->show_labels_str,
                        static_cast<std::unordered_set<std::string>*>(hailooverlay->show_labels_set));
        break;
    case PROP_HIDE_LABELS:
        g_free(hailooverlay->hide_labels_str);
        hailooverlay->hide_labels_str = g_value_dup_string(value);
        parse_label_set(hailooverlay->hide_labels_str,
                        static_cast<std::unordered_set<std::string>*>(hailooverlay->hide_labels_set));
        break;
    case PROP_TEXT_BACKGROUND:
        hailooverlay->text_background = g_value_get_boolean(value);
        break;
    case PROP_TEXT_FONT_SCALE:
        hailooverlay->text_font_scale = g_value_get_float(value);
        break;
    case PROP_STATS_OVERLAY:
        hailooverlay->stats_overlay = g_value_get_boolean(value);
        break;
    case PROP_USE_CUSTOM_COLORS:
        hailooverlay->use_custom_colors = g_value_get_boolean(value);
        break;
    case PROP_SPRITE_CONFIG:
    {
        g_free(hailooverlay->sprite_config_path);
        hailooverlay->sprite_config_path = g_value_dup_string(value);
        delete static_cast<SpriteCache*>(hailooverlay->sprite_cache);
        hailooverlay->sprite_cache = nullptr;
        if (hailooverlay->sprite_config_path && hailooverlay->sprite_config_path[0] != '\0') {
            auto *cache = new SpriteCache();
            try {
                cache->load_config(hailooverlay->sprite_config_path);
                hailooverlay->sprite_cache = cache;
            } catch (const std::exception &e) {
                GST_ERROR_OBJECT(hailooverlay, "Failed to load sprite-config '%s': %s",
                                  hailooverlay->sprite_config_path, e.what());
                delete cache;
                hailooverlay->sprite_cache = nullptr;
            }
        }
        break;
    }
    case PROP_STYLE_CONFIG:
    {
        g_free(hailooverlay->style_config_path);
        hailooverlay->style_config_path = g_value_dup_string(value);
        delete static_cast<StyleConfig*>(hailooverlay->style_config);
        hailooverlay->style_config = nullptr;
        if (hailooverlay->style_config_path && hailooverlay->style_config_path[0] != '\0') {
            auto *cfg = new StyleConfig();
            try {
                cfg->load(hailooverlay->style_config_path);
                hailooverlay->style_config = cfg;
            } catch (const std::exception &e) {
                GST_ERROR_OBJECT(hailooverlay, "Failed to load style-config '%s': %s",
                                  hailooverlay->style_config_path, e.what());
                delete cfg;
                hailooverlay->style_config = nullptr;
            }
        }
        break;
    }
    case PROP_SPRITE_REPLACE_BBOX:
        hailooverlay->sprite_replace_bbox = g_value_get_boolean(value);
        break;
    case PROP_HUD_OVERLAY:
        hailooverlay->hud_overlay = g_value_get_boolean(value);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
        break;
    }
}

void gst_hailooverlay_community_get_property(GObject *object, guint property_id,
                                   GValue *value, GParamSpec *pspec)
{
    GstHailoOverlayCommunity *hailooverlay = GST_HAILO_OVERLAY_COMMUNITY(object);

    GST_DEBUG_OBJECT(hailooverlay, "get_property");

    switch (property_id)
    {
    case PROP_LINE_THICKNESS:
        g_value_set_int(value, hailooverlay->line_thickness);
        break;
    case PROP_FONT_THICKNESS:
        g_value_set_int(value, hailooverlay->font_thickness);
        break;
    case PROP_LANDMARK_POINT_RADIUS:
        g_value_set_float(value, hailooverlay->landmark_point_radius);
        break;
    case PROP_FACE_BLUR:
        g_value_set_boolean(value, hailooverlay->face_blur);
        break;
    case PROP_SHOW_CONF:
        g_value_set_boolean(value, hailooverlay->show_confidence);
        break;
    case PROP_LOCAL_GALLERY:
        g_value_set_boolean(value, hailooverlay->local_gallery);
        break;
    case PROP_MASK_OVERLAY_N_THREADS:
        g_value_set_uint(value, hailooverlay->mask_overlay_n_threads);
        break;
    case PROP_SHOW_BBOX:
        g_value_set_boolean(value, hailooverlay->show_bbox);
        break;
    case PROP_SHOW_LABELS_TEXT:
        g_value_set_boolean(value, hailooverlay->show_labels_text);
        break;
    case PROP_SHOW_LANDMARKS:
        g_value_set_boolean(value, hailooverlay->show_landmarks);
        break;
    case PROP_SHOW_TRACKING_ID:
        g_value_set_boolean(value, hailooverlay->show_tracking_id);
        break;
    case PROP_MIN_CONFIDENCE:
        g_value_set_float(value, hailooverlay->min_confidence);
        break;
    case PROP_SHOW_LABELS:
        g_value_set_string(value, hailooverlay->show_labels_str);
        break;
    case PROP_HIDE_LABELS:
        g_value_set_string(value, hailooverlay->hide_labels_str);
        break;
    case PROP_TEXT_BACKGROUND:
        g_value_set_boolean(value, hailooverlay->text_background);
        break;
    case PROP_TEXT_FONT_SCALE:
        g_value_set_float(value, hailooverlay->text_font_scale);
        break;
    case PROP_STATS_OVERLAY:
        g_value_set_boolean(value, hailooverlay->stats_overlay);
        break;
    case PROP_USE_CUSTOM_COLORS:
        g_value_set_boolean(value, hailooverlay->use_custom_colors);
        break;
    case PROP_SPRITE_CONFIG:
        g_value_set_string(value, hailooverlay->sprite_config_path);
        break;
    case PROP_STYLE_CONFIG:
        g_value_set_string(value, hailooverlay->style_config_path);
        break;
    case PROP_SPRITE_REPLACE_BBOX:
        g_value_set_boolean(value, hailooverlay->sprite_replace_bbox);
        break;
    case PROP_HUD_OVERLAY:
        g_value_set_boolean(value, hailooverlay->hud_overlay);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
        break;
    }
}

void gst_hailooverlay_community_dispose(GObject *object)
{
    GstHailoOverlayCommunity *hailooverlay = GST_HAILO_OVERLAY_COMMUNITY(object);
    GST_DEBUG_OBJECT(hailooverlay, "dispose");

    G_OBJECT_CLASS(gst_hailooverlay_community_parent_class)->dispose(object);
}

void gst_hailooverlay_community_finalize(GObject *object)
{
    GstHailoOverlayCommunity *hailooverlay = GST_HAILO_OVERLAY_COMMUNITY(object);
    GST_DEBUG_OBJECT(hailooverlay, "finalize");

    g_free(hailooverlay->show_labels_str);
    g_free(hailooverlay->hide_labels_str);
    g_free(hailooverlay->sprite_config_path);
    g_free(hailooverlay->style_config_path);
    delete static_cast<std::unordered_set<std::string>*>(hailooverlay->show_labels_set);
    delete static_cast<std::unordered_set<std::string>*>(hailooverlay->hide_labels_set);
    delete static_cast<SpriteCache*>(hailooverlay->sprite_cache);
    delete static_cast<StyleConfig*>(hailooverlay->style_config);

    G_OBJECT_CLASS(gst_hailooverlay_community_parent_class)->finalize(object);
}

static gboolean
gst_hailooverlay_community_start(GstBaseTransform *trans)
{
    GstHailoOverlayCommunity *hailooverlay = GST_HAILO_OVERLAY_COMMUNITY(trans);
    GST_DEBUG_OBJECT(hailooverlay, "start");

    return TRUE;
}

static gboolean
gst_hailooverlay_community_stop(GstBaseTransform *trans)
{
    GstHailoOverlayCommunity *hailooverlay = GST_HAILO_OVERLAY_COMMUNITY(trans);
    GST_DEBUG_OBJECT(hailooverlay, "stop");

    return TRUE;
}

static GstFlowReturn
gst_hailooverlay_community_transform_ip(GstBaseTransform *trans,
                              GstBuffer *buffer)
{
    overlay_status_t ret = OVERLAY_STATUS_UNINITIALIZED;
    GstFlowReturn status = GST_FLOW_ERROR;
    GstHailoOverlayCommunity *hailooverlay = GST_HAILO_OVERLAY_COMMUNITY(trans);
    GstCaps *caps;
    cv::Mat mat;
    HailoROIPtr hailo_roi;
    GST_DEBUG_OBJECT(hailooverlay, "transform_ip");

    caps = gst_pad_get_current_caps(trans->sinkpad);

    GstVideoInfo *info = gst_video_info_new();
    gst_video_info_from_caps(info, caps);

    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_READWRITE);

    std::shared_ptr<HailoMat> hmat = get_mat_by_format(buffer, info, hailooverlay->line_thickness, hailooverlay->font_thickness);
    gst_video_info_free(info);

    hailo_roi = get_hailo_main_roi(buffer, true);

    if (hmat)
    {
        if (hailooverlay->face_blur)
        {
            face_blur(*hmat.get(), hailo_roi);
        }

        // Build OverlayParams
        OverlayParams params;
        params.landmark_point_radius = hailooverlay->landmark_point_radius;
        params.show_confidence = hailooverlay->show_confidence;
        params.local_gallery = hailooverlay->local_gallery;
        params.mask_overlay_n_threads = hailooverlay->mask_overlay_n_threads;
        params.show_bbox = hailooverlay->show_bbox;
        params.show_labels_text = hailooverlay->show_labels_text;
        params.show_landmarks = hailooverlay->show_landmarks;
        params.show_tracking_id = hailooverlay->show_tracking_id;
        params.min_confidence = hailooverlay->min_confidence;
        params.use_custom_colors = hailooverlay->use_custom_colors;
        params.text_background = hailooverlay->text_background;
        params.text_font_scale = hailooverlay->text_font_scale;
        params.stats_overlay = hailooverlay->stats_overlay;
        auto *show_set = static_cast<std::unordered_set<std::string>*>(hailooverlay->show_labels_set);
        auto *hide_set = static_cast<std::unordered_set<std::string>*>(hailooverlay->hide_labels_set);
        params.show_labels_set = show_set->empty() ? nullptr : show_set;
        params.hide_labels_set = hide_set->empty() ? nullptr : hide_set;
        params.sprite_cache = hailooverlay->sprite_cache;
        params.style_config = hailooverlay->style_config;
        params.sprite_replace_bbox = hailooverlay->sprite_replace_bbox;
        params.keypoint_sprites = nullptr;

        ret = draw_all(*hmat.get(), hailo_roi, params);

        if (hailooverlay->stats_overlay)
        {
            draw_stats_overlay(*hmat.get(), hailo_roi,
                               hailooverlay->stats_timestamps,
                               hailooverlay->stats_index,
                               hailooverlay->stats_count);
        }

        if (hailooverlay->hud_overlay)
        {
            draw_hud_overlay(*hmat.get(), hailo_roi);
        }
    }
    if (ret != OVERLAY_STATUS_OK)
    {
        status = GST_FLOW_ERROR;
        goto cleanup;
    }
    status = GST_FLOW_OK;
cleanup:
    gst_caps_unref(caps);
    gst_buffer_unmap(buffer, &map);
    return status;
}

/* Plugin registration */
static gboolean
plugin_init(GstPlugin *plugin)
{
    return gst_element_register(plugin, "hailooverlay_community", GST_RANK_NONE, GST_TYPE_HAILO_OVERLAY_COMMUNITY);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    hailooverlay_community,
    "Overlay element with confidence-based landmark filtering",
    plugin_init,
    "1.0",
    "LGPL",
    "hailo-apps-infra",
    "https://github.com/hailo-ai/hailo-apps-infra")
