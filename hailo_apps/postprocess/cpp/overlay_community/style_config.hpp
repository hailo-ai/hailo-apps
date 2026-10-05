/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 *
 * Per-class style configuration for hailooverlay_community.
 * Loaded from a YAML file, applies color/visibility overrides per detection label or class_id.
 **/
#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <yaml-cpp/yaml.h>

struct StyleEntry {
    cv::Scalar color = cv::Scalar(-1, -1, -1);
    cv::Scalar text_color = cv::Scalar(-1, -1, -1);
    int line_thickness = -1;    // -1 = use global default
    int show_bbox = -1;         // -1 = inherit, 0 = false, 1 = true
    int show_label = -1;
    int show_landmarks = -1;
    std::string sprite_key;     // empty = no sprite (drawn on bbox)
    // Per-keypoint sprite overrides: keypoint index -> sprite key
    // When set, the sprite replaces the keypoint dot at that index.
    std::unordered_map<int, std::string> keypoint_sprites;
};

class StyleConfig {
public:
    void load(const std::string &yaml_path) {
        by_label_.clear();
        by_class_id_.clear();

        YAML::Node root = YAML::LoadFile(yaml_path);
        if (!root["styles"]) return;

        for (auto it = root["styles"].begin(); it != root["styles"].end(); ++it) {
            std::string key = it->first.as<std::string>();
            YAML::Node val = it->second;
            StyleEntry entry;

            if (val["color"] && val["color"].IsSequence() && val["color"].size() == 3)
                entry.color = cv::Scalar(val["color"][0].as<int>(), val["color"][1].as<int>(), val["color"][2].as<int>());
            if (val["text_color"] && val["text_color"].IsSequence() && val["text_color"].size() == 3)
                entry.text_color = cv::Scalar(val["text_color"][0].as<int>(), val["text_color"][1].as<int>(), val["text_color"][2].as<int>());
            if (val["line_thickness"])
                entry.line_thickness = val["line_thickness"].as<int>();
            if (val["show_bbox"])
                entry.show_bbox = val["show_bbox"].as<bool>() ? 1 : 0;
            if (val["show_label"])
                entry.show_label = val["show_label"].as<bool>() ? 1 : 0;
            if (val["show_landmarks"])
                entry.show_landmarks = val["show_landmarks"].as<bool>() ? 1 : 0;
            if (val["sprite_key"])
                entry.sprite_key = val["sprite_key"].as<std::string>();
            if (val["keypoint_sprites"] && val["keypoint_sprites"].IsMap()) {
                for (auto kp = val["keypoint_sprites"].begin(); kp != val["keypoint_sprites"].end(); ++kp) {
                    int idx = kp->first.as<int>();
                    std::string skey = kp->second.as<std::string>();
                    entry.keypoint_sprites[idx] = skey;
                }
            }

            // If key is purely numeric, also store by class_id
            bool is_numeric = !key.empty();
            for (char c : key) {
                if (!isdigit(c) && c != '-') { is_numeric = false; break; }
            }
            if (is_numeric) {
                by_class_id_[std::stoi(key)] = entry;
            }
            by_label_[key] = entry;
        }
    }

    const StyleEntry *lookup(const std::string &label, int class_id) const {
        auto it = by_label_.find(label);
        if (it != by_label_.end()) return &it->second;
        auto cit = by_class_id_.find(class_id);
        if (cit != by_class_id_.end()) return &cit->second;
        return nullptr;
    }

private:
    std::unordered_map<std::string, StyleEntry> by_label_;
    std::unordered_map<int, StyleEntry> by_class_id_;
};
