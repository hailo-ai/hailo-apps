/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 *
 * Sprite/stamp cache for hailooverlay_community.
 * Loads PNG images and caches resized copies bucketed to 8px increments.
 **/
#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <map>
#include <yaml-cpp/yaml.h>

class SpriteCache {
public:
    void load_config(const std::string &yaml_path) {
        key_to_path_.clear();
        cache_.clear();
        source_cache_.clear();

        YAML::Node root = YAML::LoadFile(yaml_path);
        if (!root["sprites"]) return;

        for (auto it = root["sprites"].begin(); it != root["sprites"].end(); ++it) {
            std::string key = it->first.as<std::string>();
            std::string path = it->second.as<std::string>();
            key_to_path_[key] = path;
        }
    }

    const cv::Mat *get_sprite(const std::string &key, int target_w, int target_h) {
        auto path_it = key_to_path_.find(key);
        if (path_it == key_to_path_.end()) return nullptr;
        if (target_w <= 0 || target_h <= 0) return nullptr;

        uint32_t bkey = bucket_key(target_w, target_h);
        auto &size_map = cache_[key];
        auto cached_it = size_map.find(bkey);
        if (cached_it != size_map.end())
            return &cached_it->second;

        // Load source if not yet loaded
        const cv::Mat *src = load_source(key, path_it->second);
        if (!src || src->empty()) return nullptr;

        // Compute letterbox fit
        int bw = round_up_8(target_w);
        int bh = round_up_8(target_h);
        float scale = std::min((float)bw / src->cols, (float)bh / src->rows);
        int sw = (int)(src->cols * scale);
        int sh = (int)(src->rows * scale);
        if (sw <= 0 || sh <= 0) return nullptr;

        cv::Mat resized;
        cv::resize(*src, resized, cv::Size(sw, sh), 0, 0, cv::INTER_AREA);

        // Center in letterbox
        cv::Mat letterbox(bh, bw, CV_8UC4, cv::Scalar(0, 0, 0, 0));
        int ox = (bw - sw) / 2;
        int oy = (bh - sh) / 2;
        resized.copyTo(letterbox(cv::Rect(ox, oy, sw, sh)));

        auto result = size_map.emplace(bkey, std::move(letterbox));
        return &result.first->second;
    }

private:
    static int round_up_8(int v) {
        return ((v + 7) / 8) * 8;
    }

    static uint32_t bucket_key(int w, int h) {
        return ((uint32_t)round_up_8(w) << 16) | (uint32_t)round_up_8(h);
    }

    const cv::Mat *load_source(const std::string &key, const std::string &path) {
        auto it = source_cache_.find(key);
        if (it != source_cache_.end())
            return &it->second;

        cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
        if (img.empty()) return nullptr;

        // Convert to 4-channel BGRA if needed
        if (img.channels() == 3) {
            cv::cvtColor(img, img, cv::COLOR_BGR2BGRA);
        } else if (img.channels() == 1) {
            cv::cvtColor(img, img, cv::COLOR_GRAY2BGRA);
        }
        // img.channels() == 4 (BGRA) is the expected case for PNGs with alpha

        auto result = source_cache_.emplace(key, std::move(img));
        return &result.first->second;
    }

    std::unordered_map<std::string, std::string> key_to_path_;
    std::unordered_map<std::string, cv::Mat> source_cache_;
    std::unordered_map<std::string, std::map<uint32_t, cv::Mat>> cache_;
};
