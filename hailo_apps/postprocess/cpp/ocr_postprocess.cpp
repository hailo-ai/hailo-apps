#include "ocr_postprocess.hpp"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <iomanip>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <cctype>

// RapidJSON for optional config
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/schema.h"

#if __GNUC__ > 8
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

// ---------------------------
// JSON helpers
// ---------------------------
static bool validate_json_with_schema(FILE *fp, const char *schema)
{
    char buffer[1 << 12];
    rapidjson::FileReadStream is(fp, buffer, sizeof(buffer));
    rapidjson::Document sd;
    sd.Parse(schema);
    if (sd.HasParseError())
        return false;
    rapidjson::SchemaDocument sdoc(sd);
    rapidjson::SchemaValidator validator(sdoc);
    fseek(fp, 0, SEEK_SET);
    rapidjson::FileReadStream is2(fp, buffer, sizeof(buffer));
    rapidjson::Document d;
    d.ParseStream(is2);
    if (d.HasParseError())
        return false;
    return d.Accept(validator);
}

static void load_default_charset(OcrParams &p)
{
    p.charset.emplace_back("blank");
    for (char c = '0'; c <= '9'; ++c)
        p.charset.emplace_back(1, c);
    p.charset.emplace_back(":");
    p.charset.emplace_back(";");
    p.charset.emplace_back("<");
    p.charset.emplace_back("=");
    p.charset.emplace_back(">");
    p.charset.emplace_back("?");
    p.charset.emplace_back("@");
    for (char c = 'A'; c <= 'Z'; ++c)
        p.charset.emplace_back(1, c);
    p.charset.emplace_back("[");
    p.charset.emplace_back("\\");
    p.charset.emplace_back("]");
    p.charset.emplace_back("^");
    p.charset.emplace_back("_");
    p.charset.emplace_back("`");
    for (char c = 'a'; c <= 'z'; ++c)
        p.charset.emplace_back(1, c);
    p.charset.emplace_back("{");
    p.charset.emplace_back("|");
    p.charset.emplace_back("}");
    p.charset.emplace_back("~");
    p.charset.emplace_back("!");
    p.charset.emplace_back("\"");
    p.charset.emplace_back("#");
    p.charset.emplace_back("$");
    p.charset.emplace_back("%");
    p.charset.emplace_back("&");
    p.charset.emplace_back("'");
    p.charset.emplace_back("(");
    p.charset.emplace_back(")");
    p.charset.emplace_back("*");
    p.charset.emplace_back("+");
    p.charset.emplace_back(",");
    p.charset.emplace_back("-");
    p.charset.emplace_back(".");
    p.charset.emplace_back("/");
    p.charset.emplace_back(" ");
    p.charset.emplace_back(" "); // Second space to match Python reference charset
}

static void load_charset_from_file(OcrParams &p)
{
    if (p.charset_path.empty())
    {
        load_default_charset(p);
        return;
    }
    std::ifstream in(p.charset_path);
    if (!in.is_open())
        throw std::runtime_error("Failed to open charset file: " + p.charset_path);
    std::string line;
    while (std::getline(in, line))
        p.charset.push_back(line);
    if (p.charset.empty())
        load_default_charset(p);
}

// ---------------------------
// Spell Correction Functions
// ---------------------------

/**
 * @brief Calculate Levenshtein edit distance between two strings
 */
static int levenshtein_distance(const std::string &s1, const std::string &s2)
{
    const size_t len1 = s1.size();
    const size_t len2 = s2.size();

    if (len1 == 0)
        return len2;
    if (len2 == 0)
        return len1;

    std::vector<std::vector<int>> dp(len1 + 1, std::vector<int>(len2 + 1));

    for (size_t i = 0; i <= len1; ++i)
        dp[i][0] = i;
    for (size_t j = 0; j <= len2; ++j)
        dp[0][j] = j;

    for (size_t i = 1; i <= len1; ++i)
    {
        for (size_t j = 1; j <= len2; ++j)
        {
            int cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
            dp[i][j] = std::min({dp[i - 1][j] + 1,          // deletion
                                 dp[i][j - 1] + 1,          // insertion
                                 dp[i - 1][j - 1] + cost}); // substitution
        }
    }

    return dp[len1][len2];
}

/**
 * @brief Load frequency dictionary from file
 * Format: word frequency_count (space-separated)
 */
static void load_frequency_dictionary(OcrParams &p, const std::string &config_dir = "")
{
    if (p.frequency_dict_path.empty())
    {
        return; // No dictionary specified, spell correction disabled
    }

    // Resolve relative paths relative to config file directory
    std::string dict_path = p.frequency_dict_path;
    if (!config_dir.empty() && !fs::path(dict_path).is_absolute())
    {
        dict_path = (fs::path(config_dir) / dict_path).string();
    }

    std::ifstream in(dict_path);
    if (!in.is_open())
    {
        return;
    }

    std::string line;
    while (std::getline(in, line))
    {
        std::istringstream iss(line);
        std::string word;
        uint64_t frequency;

        if (iss >> word >> frequency)
        {
            // Convert to lowercase for case-insensitive matching
            std::string word_lower = word;
            std::transform(word_lower.begin(), word_lower.end(), word_lower.begin(), ::tolower);
            p.frequency_dict[word_lower] = frequency;
        }
    }
}

/**
 * @brief Find best matching word in dictionary using edit distance
 * Returns the word with highest frequency among words within max_edit_distance
 */
static std::string find_best_match(const std::string &word, const OcrParams &p)
{
    if (p.frequency_dict.empty())
    {
        return word; // No dictionary loaded
    }

    std::string word_lower = word;
    std::transform(word_lower.begin(), word_lower.end(), word_lower.begin(), ::tolower);

    // Check if exact match exists
    auto it = p.frequency_dict.find(word_lower);
    if (it != p.frequency_dict.end())
    {
        return word; // Exact match found, return original (preserve case)
    }

    // Find best match within edit distance
    std::string best_word = word;
    uint64_t best_frequency = 0;
    int best_distance = p.max_edit_distance + 1;

    for (const auto &entry : p.frequency_dict)
    {
        int dist = levenshtein_distance(word_lower, entry.first);
        if (dist <= p.max_edit_distance)
        {
            // Prefer closer matches, or if same distance, prefer higher frequency
            if (dist < best_distance || (dist == best_distance && entry.second > best_frequency))
            {
                best_distance = dist;
                best_frequency = entry.second;
                best_word = entry.first; // Use lowercase from dictionary
            }
        }
    }

    // If no good match found, return original word
    if (best_distance > p.max_edit_distance)
    {
        return word;
    }

    return best_word;
}

/**
 * @brief Correct text using frequency dictionary (similar to SymSpell lookup_compound)
 * Splits text into words and corrects each word individually
 */
static std::string correct_text(const std::string &text, const OcrParams &p)
{
    if (p.frequency_dict.empty() || text.empty())
    {
        return text;
    }

    std::istringstream iss(text);
    std::ostringstream oss;
    std::string word;
    bool first = true;

    while (iss >> word)
    {
        if (!first)
            oss << " ";
        first = false;

        // Remove punctuation for matching, but preserve it in output
        std::string word_clean = word;
        std::string prefix = "";
        std::string suffix = "";

        // Extract leading/trailing punctuation
        while (!word_clean.empty() && !std::isalnum(word_clean.front()))
        {
            prefix += word_clean.front();
            word_clean.erase(0, 1);
        }
        while (!word_clean.empty() && !std::isalnum(word_clean.back()))
        {
            suffix = word_clean.back() + suffix;
            word_clean.pop_back();
        }

        if (!word_clean.empty())
        {
            std::string corrected = find_best_match(word_clean, p);
            oss << prefix << corrected << suffix;
        }
        else
        {
            oss << word; // No alphanumeric characters, keep as-is
        }
    }

    return oss.str();
}

// ---------------------------
// init / free_resources
// ---------------------------
OcrParams *init(const std::string config_path, const std::string /*function_name*/)
{
    auto *params = new OcrParams();
    if (fs::exists(config_path))
    {
        const char *schema = R""""({
          "$schema": "http://json-schema.org/draft-04/schema#",
          "type": "object",
          "properties": {
            "det_bin_thresh":     { "type": "number" },
            "det_box_thresh":     { "type": "number" },
            "det_unclip_ratio":   { "type": "number" },
            "det_max_candidates": { "type": "integer" },
            "det_min_box_size":   { "type": "number" },
            "det_output_name":    { "type": "string" },
            "det_map_h":          { "type": "integer" },
            "det_map_w":          { "type": "integer" },
            "letterbox_fix":      { "type": "boolean" },
            "rec_output_name":    { "type": "string" },
            "charset_path":       { "type": "string" },
            "blank_index":        { "type": "integer" },
            "logits_are_softmax": { "type": "boolean" },
            "time_major":         { "type": "boolean" },
            "text_conf_smooth":   { "type": "number" },
            "attach_caption_box": { "type": "boolean" },
            "frequency_dict_path": { "type": "string" },
            "max_edit_distance":  { "type": "integer" }
          }
        })"""";
        FILE *fp = fopen(config_path.c_str(), "r");
        if (!fp)
            throw std::runtime_error("JSON config file cannot be opened");
        bool ok = validate_json_with_schema(fp, schema);
        if (!ok)
        {
            fclose(fp);
            throw std::runtime_error("JSON config doesn't match schema");
        }
        fseek(fp, 0, SEEK_SET);
        char buffer[1 << 14];
        rapidjson::FileReadStream frs(fp, buffer, sizeof(buffer));
        rapidjson::Document d;
        d.ParseStream(frs);
        fclose(fp);
        auto getf = [&](const char *k, float &dst)
        { if (d.HasMember(k)) dst = d[k].GetFloat(); };
        auto geti = [&](const char *k, int &dst)
        { if (d.HasMember(k)) dst = d[k].GetInt(); };
        auto getb = [&](const char *k, bool &dst)
        { if (d.HasMember(k)) dst = d[k].GetBool(); };
        auto gets = [&](const char *k, std::string &dst)
        { if (d.HasMember(k)) dst = d[k].GetString(); };
        getf("det_bin_thresh", params->det_bin_thresh);
        getf("det_box_thresh", params->det_box_thresh);
        getf("det_unclip_ratio", params->det_unclip_ratio);
        geti("det_max_candidates", params->det_max_candidates);
        getf("det_min_box_size", params->det_min_box_size);
        gets("det_output_name", params->det_output_name);
        geti("det_map_h", params->det_map_h);
        geti("det_map_w", params->det_map_w);
        getb("letterbox_fix", params->letterbox_fix);
        gets("rec_output_name", params->rec_output_name);
        gets("charset_path", params->charset_path);
        geti("blank_index", params->blank_index);
        getb("logits_are_softmax", params->logits_are_softmax);
        getb("time_major", params->time_major);
        getf("text_conf_smooth", params->text_conf_smooth);
        getb("attach_caption_box", params->attach_caption_box);
        gets("frequency_dict_path", params->frequency_dict_path);
        geti("max_edit_distance", params->max_edit_distance);
    }
    load_charset_from_file(*params);
    // Get config file directory for resolving relative paths
    std::string config_dir;
    if (!config_path.empty() && fs::exists(config_path))
    {
        config_dir = fs::path(config_path).parent_path().string();
    }
    load_frequency_dictionary(*params, config_dir);
    return params;
}

void free_resources(void *params_void_ptr)
{
    auto *p = reinterpret_cast<OcrParams *>(params_void_ptr);
    delete p;
}

// ---------------------------
// Tensor helpers
// ---------------------------
static cv::Mat tensor_to_probmap_u8_as_float(HailoTensorPtr t, int H, int W)
{
    const uint8_t *u8 = reinterpret_cast<const uint8_t *>(t->data());
    if (!u8)
        throw std::runtime_error("Detector tensor has null data()");
    cv::Mat prob(H, W, CV_8UC1);
    std::memcpy(prob.data, u8, (size_t)H * (size_t)W * sizeof(uint8_t));
    cv::Mat out;
    prob.convertTo(out, CV_32F, 1.0 / 255.0);
    return out;
}

static HailoTensorPtr get_tensor_by_name_or_fallback(const HailoROIPtr &roi, const std::string &desired)
{
    HailoTensorPtr chosen;
    for (auto &t : roi->get_tensors())
    {
        if (t->name() == desired)
        {
            chosen = t;
            break;
        }
    }
    if (!chosen)
    {
        auto tensors = roi->get_tensors();
        if (tensors.empty())
            throw std::runtime_error("ROI has no tensors");
        chosen = tensors.front();
    }
    return chosen;
}

// ---------------------------
// Helper functions
// ---------------------------

static void softmax1d(std::vector<float> &v)
{
    float m = *std::max_element(v.begin(), v.end());
    double sum = 0.0;
    for (float &x : v)
        sum += std::exp(double(x - m));
    for (float &x : v)
        x = float(std::exp(double(x - m)) / sum);
}

static inline int odd_at_least(int v) { return (v % 2 == 0) ? v + 1 : v; }

static void merge_horizontal_boxes(std::vector<cv::Rect> &rects,
                                   int max_gap_px,
                                   float min_y_overlap_ratio)
{
    if (rects.size() <= 1)
        return;
    std::sort(rects.begin(), rects.end(),
              [](const cv::Rect &a, const cv::Rect &b)
              { return a.x < b.x; });
    auto y_overlap_ratio = [](const cv::Rect &a, const cv::Rect &b)
    {
        const int top = std::max(a.y, b.y);
        const int bot = std::min(a.y + a.height, b.y + b.height);
        const int inter = std::max(0, bot - top);
        const int min_h = std::max(1, std::min(a.height, b.height));
        return static_cast<float>(inter) / static_cast<float>(min_h);
    };
    std::vector<cv::Rect> merged;
    merged.reserve(rects.size());
    cv::Rect run = rects[0];
    for (size_t i = 1; i < rects.size(); ++i)
    {
        const cv::Rect &next = rects[i];
        const int gap = next.x - (run.x + run.width);
        const float yov = y_overlap_ratio(run, next);
        if (gap <= max_gap_px && yov >= min_y_overlap_ratio)
        {
            run |= next;
        }
        else
        {
            merged.push_back(run);
            run = next;
        }
    }
    merged.push_back(run);
    rects.swap(merged);
}

static float region_score_rect(const cv::Mat &prob, const cv::Rect &r)
{
    const cv::Rect R = r & cv::Rect(0, 0, prob.cols, prob.rows);
    if (R.empty())
        return 0.f;
    return (float)cv::mean(prob(R))[0];
}

// ---------------------------
// OCR Detection Post-process
// ---------------------------
extern "C" void paddleocr_det(HailoROIPtr roi, void *params_void_ptr)
{
    if (!roi->has_tensors())
    {
        return;
    }
    auto *p = reinterpret_cast<OcrParams *>(params_void_ptr);

    // === SECTION 1: Tensor Setup ===
    HailoTensorPtr t = get_tensor_by_name_or_fallback(roi, p->det_output_name);
    const auto &sh = t->shape();
    int H = 0, W = 0;
    if (sh.size() == 4)
    {
        if (sh[1] == 1)
        {
            H = int(sh[2]);
            W = int(sh[3]);
        }
        else if (sh[3] == 1)
        {
            H = int(sh[1]);
            W = int(sh[2]);
        }
        else
        {
            H = int(sh[2]);
            W = int(sh[3]);
        }
    }
    else if (sh.size() == 3)
    {
        if (sh[2] == 1)
        {
            H = int(sh[0]);
            W = int(sh[1]);
        }
        else if (sh[0] == 1)
        {
            H = int(sh[1]);
            W = int(sh[2]);
        }
        else
        {
            std::vector<int> v{int(sh[0]), int(sh[1]), int(sh[2])};
            std::sort(v.begin(), v.end());
            H = v[1];
            W = v[2];
        }
    }
    else if (sh.size() == 2)
    {
        H = int(sh[0]);
        W = int(sh[1]);
    }
    else
    {
        H = p->det_map_h;
        W = p->det_map_w;
    }
    if (W <= 4 && H > 16)
        std::swap(H, W);

    // === SECTION 2: Probability Map Analysis ===
    cv::Mat prob = tensor_to_probmap_u8_as_float(t, H, W);
    // Optimized: Only count non-zero pixels, skip minMaxLoc (not used)
    int above_default = cv::countNonZero(prob > p->det_bin_thresh);
    const int total_pixels = H * W;
    float fg_ratio_default = (total_pixels > 0) ? float(above_default) / float(total_pixels) : 0.0f;

    // === SECTION 3: Adaptive Thresholding ===
    float bin_thr = p->det_bin_thresh;
    if (fg_ratio_default < 0.003f)
        bin_thr = std::max(0.15f, p->det_bin_thresh * 0.8f);
    else if (fg_ratio_default > 0.08f)
        bin_thr = std::min(0.75f, p->det_bin_thresh * 1.2f);

    cv::Mat bin;
    cv::threshold(prob, bin, bin_thr, 1.0, cv::THRESH_BINARY);
    bin.convertTo(bin, CV_8U, 255.0);

    // === SECTION 4: Morphological Operations ===
    float kscale = (fg_ratio_default < 0.01f) ? 1.0f : (fg_ratio_default > 0.06f ? 1.5f : 1.2f);
    int kx = odd_at_least(std::max(3, int(std::round(W * 0.012f * kscale))));
    int ky = odd_at_least(std::max(1, int(std::round(H * 0.006f * kscale))));
    cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kx, ky));
    cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, k, {-1, -1}, 1);

    // === SECTION 5: Contour Detection ===
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Optimized: Early filtering - filter out tiny contours before expensive operations
    const int MIN_CONTOUR_AREA = 20; // Filter very small contours early
    std::vector<cv::Rect> rects;
    rects.reserve(contours.size());
    for (auto &c : contours)
    {
        if (c.empty())
            continue;
        cv::Rect r = cv::boundingRect(c);
        // Early filter: skip very small rectangles before expensive operations
        if (r.width > 0 && r.height > 0 && (r.width * r.height) >= MIN_CONTOUR_AREA)
        {
            rects.push_back(r);
        }
    }

    if (rects.empty())
    {
        return;
    }

    // === SECTION 6: Box Merging ===
    std::vector<int> hs;
    hs.reserve(rects.size());
    for (auto &r : rects)
        hs.push_back(r.height);
    std::nth_element(hs.begin(), hs.begin() + hs.size() / 2, hs.end());
    const int median_h = hs[hs.size() / 2];
    int GAP_PX = std::max(3, std::min(int(W * 0.02f), int(std::round(median_h * 1.0f))));
    float YOV = 0.45f;
    merge_horizontal_boxes(rects, GAP_PX, YOV);

    // === SECTION 7: Box Inflation ===
    auto clamp_rect_xywh = [&](cv::Rect r) -> cv::Rect
    {
        r.x = std::max(0, std::min(r.x, W - 1));
        r.y = std::max(0, std::min(r.y, H - 1));
        r.width = std::max(1, std::min(r.width, W - r.x));
        r.height = std::max(1, std::min(r.height, H - r.y));
        return r;
    };

    const int PAD_X0 = std::max(2, int(std::round(median_h * 0.6f)));
    const int PAD_Y0 = std::max(1, int(std::round(median_h * 0.35f)));
    // Optimized: Reduced iterations from 2 to 1 for better performance
    // Box inflation is still effective with 1 iteration, and reduces computation by ~50%
    const int GROW_ITERS = 1;
    const float GROW_X_PER_H = 0.15f;
    const float GROW_Y_PER_H = 0.12f;

    // Track statistics for summary
    int dropped_minH = 0, dropped_area = 0, dropped_ar = 0, dropped_score = 0;

    for (size_t i = 0; i < rects.size(); ++i)
    {
        cv::Rect r = rects[i];

        // Base inflate
        r = clamp_rect_xywh(cv::Rect(r.x - PAD_X0, r.y - PAD_Y0,
                                     r.width + 2 * PAD_X0, r.height + 2 * PAD_Y0));

        // Extra vertical thickening for very wide lines
        float ar_after_base = float(r.width) / std::max(1, r.height);
        if (ar_after_base > 10.0f)
        {
            int addY = std::max(PAD_Y0, int(std::round(r.height * 0.5f)));
            r = clamp_rect_xywh(cv::Rect(r.x, r.y - addY / 2,
                                         r.width, r.height + addY));
        }

        // Iterative grows
        for (int it = 0; it < GROW_ITERS; ++it)
        {
            int gx = std::max(1, int(std::round(std::max(2.0f, r.height * GROW_X_PER_H))));
            int gy = std::max(1, int(std::round(std::max(1.0f, r.height * GROW_Y_PER_H))));
            r = clamp_rect_xywh(cv::Rect(r.x - gx, r.y - gy,
                                         r.width + 2 * gx, r.height + 2 * gy));
        }

        rects[i] = r;
    }

    // === SECTION 8: Filtering & Final Detections ===
    HailoBBox roi_box = hailo_common::create_flattened_bbox(roi->get_bbox(), roi->get_scaling_bbox());
    const float sx = roi_box.width() / float(W);
    const float sy = roi_box.height() / float(H);

    const int MIN_H_PX = std::max(3, int(std::round(H * 0.010f)));
    const float AR_MIN = 0.6f;
    const float AR_MAX = 80.0f;
    const float MIN_AREA_PX = std::max(80.0f, float(median_h * median_h * 0.4f));
    const float SCORE_BASE = p->det_box_thresh;

    std::vector<HailoDetection> outs;
    outs.reserve(rects.size());

    // Optimized: Filter by cheap criteria first, then calculate expensive score only for candidates
    for (size_t i = 0; i < rects.size(); ++i)
    {
        const auto &r = rects[i];

        // Fast filters first (no expensive operations)
        int hpx = r.height;
        if (hpx < MIN_H_PX)
        {
            dropped_minH++;
            continue;
        }

        float area = float(r.width) * float(r.height);
        if (area < MIN_AREA_PX)
        {
            dropped_area++;
            continue;
        }

        float ar = float(r.width) / std::max(1, r.height);
        if (ar < AR_MIN || ar > AR_MAX)
        {
            dropped_ar++;
            continue;
        }

        // Only calculate expensive score after passing fast filters
        float score = region_score_rect(prob, r);
        float score_min = (ar > 16.0f) ? std::max(0.45f, SCORE_BASE - 0.15f) : SCORE_BASE;
        if (score < score_min)
        {
            dropped_score++;
            continue;
        }

        // All filters passed, create detection
        float xmin = r.x * sx + roi_box.xmin();
        float ymin = r.y * sy + roi_box.ymin();
        float w = r.width * sx;
        float h = r.height * sy;

        outs.emplace_back(HailoBBox(xmin, ymin, w, h), std::string("text_region"), score);

        if ((int)outs.size() >= p->det_max_candidates)
            break;
    }

    // === SECTION 9: Fallback (if all filtered) ===
    if (outs.empty() && !rects.empty())
    {
        std::vector<size_t> idx(rects.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b)
                  { return rects[a].width > rects[b].width; });
        const size_t keepN = std::min<size_t>(2, idx.size());
        for (size_t k = 0; k < keepN; ++k)
        {
            cv::Rect r = rects[idx[k]];
            int gx = std::max(1, int(std::round(r.height * 0.1f)));
            int gy = std::max(1, int(std::round(r.height * 0.1f)));
            r = clamp_rect_xywh(cv::Rect(r.x - gx, r.y - gy, r.width + 2 * gx, r.height + 2 * gy));
            float xmin = r.x * sx + roi_box.xmin();
            float ymin = r.y * sy + roi_box.ymin();
            float w = r.width * sx;
            float h = r.height * sy;
            float score = region_score_rect(prob, r);
            outs.emplace_back(HailoBBox(xmin, ymin, w, h), std::string("text_region"), score);
        }
    }

    // === SECTION 10: Summary ===
    if (!outs.empty())
    {
        hailo_common::add_detections(roi, outs);
        if (p->letterbox_fix)
            roi->clear_scaling_bbox();
    }
}

// ---------------------------
// OCR Recognition Post-process
// ---------------------------
extern "C" void paddleocr_recognize(HailoROIPtr roi, void *params_void_ptr)
{
    if (!roi->has_tensors())
    {
        return;
    }
    auto *p = reinterpret_cast<OcrParams *>(params_void_ptr);

    HailoTensorPtr t = get_tensor_by_name_or_fallback(roi, p->rec_output_name);
    const auto &shape = t->shape();
    if (shape.size() != 3)
        throw std::runtime_error("Unexpected recognizer rank (expected 3)");

    const uint8_t *u8 = reinterpret_cast<const uint8_t *>(t->data());
    if (!u8)
        throw std::runtime_error("Recognizer tensor not UINT8");
    const size_t N = shape[0];
    const size_t D1 = shape[1];
    const size_t D2 = shape[2];
    if (N != 1)
        throw std::runtime_error("Recognizer expects N=1");
    if (D1 == 0 || D2 == 0)
        throw std::runtime_error("Invalid tensor dimensions");

    // Determine layout: Python code expects BxLxC format where L=time steps, C=charset size
    // time_major=false means layout is NTC (batch, time, channels): [N, T, C]
    // time_major=true means layout is NCT (batch, channels, time): [N, C, T]
    size_t C, T;
    bool layout_is_NCT;
    if (p->time_major)
    {
        // NCT layout: [N, C, T]
        C = D1; // Channels (charset size) is first
        T = D2; // Time steps is second
        layout_is_NCT = true;
    }
    else
    {
        // NTC layout: [N, T, C] - this is the standard format
        // When time_major=false, D1 is always T (time steps) and D2 is always C (charset size)
        T = D1; // Time steps is first dimension
        C = D2; // Charset size is second dimension
        layout_is_NCT = false;
    }

    // Build probs[T][C]
    std::vector<std::vector<float>> probs(T, std::vector<float>(C));
    const uint8_t *base = u8;
    if (layout_is_NCT)
    {
        for (size_t c = 0; c < C; ++c)
        {
            for (size_t t0 = 0; t0 < T; ++t0)
            {
                float v = base[c * T + t0] * (1.0f / 255.0f);
                probs[t0][c] = v;
            }
        }
    }
    else
    {
        for (size_t t0 = 0; t0 < T; ++t0)
        {
            for (size_t c = 0; c < C; ++c)
            {
                float v = base[t0 * C + c] * (1.0f / 255.0f);
                probs[t0][c] = v;
            }
        }
    }

    if (!p->logits_are_softmax)
    {
        for (size_t t0 = 0; t0 < T; ++t0)
            softmax1d(probs[t0]);
    }

    // Python-style CTC decode matching ocr_eval_postprocess()
    // Step 1: Get argmax indices and max probabilities for each time step
    std::vector<int> text_index(T);
    std::vector<float> text_prob(T);

    for (size_t t0 = 0; t0 < T; ++t0)
    {
        auto &row = probs[t0];
        auto it = std::max_element(row.begin(), row.end());
        text_index[t0] = int(std::distance(row.begin(), it));
        text_prob[t0] = *it;
    }

    // Step 2: Create selection mask (Python: selection[1:] = text_index[1:] != text_index[:-1])
    // This removes consecutive duplicates (CTC collapse)
    std::vector<bool> selection(T, true);
    for (size_t i = 1; i < T; ++i)
    {
        selection[i] = (text_index[i] != text_index[i - 1]);
    }

    // Step 3: Filter out blank tokens (ignored_tokens = [0])
    for (size_t i = 0; i < T; ++i)
    {
        if (text_index[i] == p->blank_index)
        {
            selection[i] = false;
        }
    }

    // Step 4: Build output text and confidence from selected tokens
    std::string out_text;
    out_text.reserve(T);
    std::vector<float> conf_list;
    conf_list.reserve(T);

    for (size_t i = 0; i < T; ++i)
    {
        if (selection[i])
        {
            int idx = text_index[i];
            if (idx >= 0 && idx < (int)p->charset.size())
            {
                out_text += p->charset[idx];
                conf_list.push_back(text_prob[i]);
            }
            else
            {
                out_text += "?";
                conf_list.push_back(0.0f);
            }
        }
    }

    // Step 5: Compute mean confidence (matching Python: np.mean(conf_list))
    float conf = 0.0f;
    if (conf_list.empty())
    {
        conf_list.push_back(0.0f);
    }
    float conf_sum = 0.0f;
    for (float c : conf_list)
    {
        conf_sum += c;
    }
    conf = conf_sum / (float)conf_list.size();

    // Attach classification to detection - simple and clean approach
    if (out_text.empty() || out_text == " ")
    {
        return;
    }

    // Apply spell correction
    std::string corrected_text = correct_text(out_text, *p);
    std::string text_to_attach = (!corrected_text.empty() && corrected_text != out_text) ? corrected_text : out_text;

    // Find detection - cropper returns ROI with detection inside
    HailoDetectionPtr target_detection = nullptr;
    auto detection_roi = std::dynamic_pointer_cast<HailoDetection>(roi);
    if (detection_roi)
    {
        target_detection = detection_roi;
    }
    else
    {
        auto detections = hailo_common::get_hailo_detections(roi);
        if (!detections.empty())
        {
            target_detection = detections[0];
        }
    }

    if (target_detection)
    {
        // Remove existing classifications
        auto existing = target_detection->get_objects_typed(HAILO_CLASSIFICATION);
        for (auto &cls : existing)
        {
            target_detection->remove_object(cls);
        }

        // Add new classification with recognized text
        auto classification = std::make_shared<HailoClassification>("text_region", text_to_attach, conf);
        target_detection->add_object(classification);
    }
}

// ---------------------------
// Cropper Functions
// ---------------------------
extern "C" void crop_text_regions_filter(HailoROIPtr roi, void *params_void_ptr)
{
    std::vector<HailoDetectionPtr> detections = hailo_common::get_hailo_detections(roi);
    std::vector<HailoDetectionPtr> text_detections;
    for (auto detection : detections)
    {
        text_detections.push_back(detection);
    }
    roi->remove_objects_typed(HAILO_DETECTION);
    for (auto text_detection : text_detections)
    {
        roi->add_object(text_detection);
    }
}

extern "C" std::vector<HailoROIPtr> crop_text_regions(std::shared_ptr<HailoMat> image,
                                                      HailoROIPtr roi,
                                                      bool use_letterbox,
                                                      bool no_scaling_bbox,
                                                      bool internal_offset,
                                                      const std::string &resize_method)
{
    std::vector<HailoROIPtr> out_rois;
    std::vector<HailoDetectionPtr> detections = hailo_common::get_hailo_detections(roi);

    if (!image || !roi)
    {
        return out_rois;
    }

    const int img_w = image->width();
    const int img_h = image->height();
    // MAX_TEXT_REGIONS: Limits regions processed per frame to avoid overload
    // Note: Recognition batch_size is 8, but we limit to 2 per frame for better performance
    // Regions that already have classifications are skipped (don't send for recognition)
    constexpr int MAX_TEXT_REGIONS = 8;
    constexpr float MIN_CONFIDENCE = 0.12f; // Increased from 0.09 to avoid processing low-quality detections
    constexpr float MIN_W_PX = 4.0f;
    constexpr float MIN_H_PX = 2.0f;

    // Padding for OCR cropping (adds context around text for better recognition)
    // Similar to Python implementation which uses boundingRect that naturally includes some padding
    constexpr float PAD_X_RATIO = 0.05f; // 5% padding on left/right
    constexpr float PAD_Y_RATIO = 0.10f; // 10% padding on top/bottom

    auto clamp01 = [](float v)
    { return std::max(0.0f, std::min(1.0f, v)); };

    // Return individual detection ROIs with padded boxes for cropping
    // The parent ROI keeps original boxes for display
    // Filtering logic (all done in C++ to reduce Python processing):
    // 1. Skip detections that already have classifications (already recognized - don't send for recognition)
    //    This prevents reprocessing tracked detections that were already recognized in previous frames
    // 2. Filter by confidence threshold
    // 3. Filter by label and size
    // 4. Limit total number of regions processed per frame (MAX_TEXT_REGIONS = 8 matches batch_size)
    int count = 0;
    for (auto &detection : detections)
    {
        if (!detection)
            continue;
        if (detection->get_label() != "text_region")
            continue;

        // FIRST: Skip detections that already have classifications (already recognized)
        // This prevents reprocessing tracked detections that were already recognized
        auto existing_classifications = detection->get_objects_typed(HAILO_CLASSIFICATION);
        if (!existing_classifications.empty())
        {
            continue; // Detection already has OCR result, skip it
        }

        // SECOND: Filter by confidence threshold to reduce processing load
        if (detection->get_confidence() < MIN_CONFIDENCE)
        {
            continue; // Skip low-confidence detections
        }

        // THIRD: Check count limit AFTER we know we're going to process this detection
        if (count >= MAX_TEXT_REGIONS)
        {
            break; // Reached limit, stop processing more detections this frame
        }

        HailoBBox orig_bbox = detection->get_bbox();
        float nw = orig_bbox.width();
        float nh = orig_bbox.height();
        float w_px = nw * img_w;
        float h_px = nh * img_h;

        // Filter by minimum size
        if (w_px < MIN_W_PX || h_px < MIN_H_PX)
            continue;

        // Create padded bbox for cropping (OCR needs context around text)
        float pad_x = nw * PAD_X_RATIO;
        float pad_y = nh * PAD_Y_RATIO;
        float padded_xmin = clamp01(orig_bbox.xmin() - pad_x);
        float padded_ymin = clamp01(orig_bbox.ymin() - pad_y);
        float padded_xmax = clamp01(orig_bbox.xmax() + pad_x);
        float padded_ymax = clamp01(orig_bbox.ymax() + pad_y);
        float padded_w = padded_xmax - padded_xmin;
        float padded_h = padded_ymax - padded_ymin;

        // IMPORTANT: Return the ORIGINAL detection object directly
        // This ensures that when recognition attaches classifications to it,
        // those classifications will be on the same detection object that exists
        // in the parent ROI, so they'll be preserved after aggregation.
        // The framework's cropper will handle the bbox for cropping internally
        // via the ROI's scaling bbox or by using the detection's bbox temporarily.
        // We set a scaling bbox on the ROI to tell the cropper to use padded bbox.
        HailoBBox padded_bbox(padded_xmin, padded_ymin, padded_w, padded_h);

        // Create a ROI wrapper with the padded bbox for cropping
        // The detection inside will be the original one, so classifications attach to it
        HailoROIPtr crop_roi = std::make_shared<HailoROI>(padded_bbox);
        crop_roi->add_object(detection);

        // Return the ROI containing the original detection
        // When aggregator merges, it will preserve the classifications on the detection
        out_rois.push_back(crop_roi);
        ++count;
    }

    return out_rois;
}
