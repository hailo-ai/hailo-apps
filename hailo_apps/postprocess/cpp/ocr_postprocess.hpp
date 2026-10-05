/**  
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.  
**/  
#pragma once  
#include "hailo_objects.hpp"  
#include "hailo_common.hpp" 
#include "hailomat.hpp"

  
#include <string>  
#include <vector>
#include <unordered_map>  
  
// OCR Parameters structure  
struct OcrParams {  
    // Detection parameters  
    float det_bin_thresh = 0.3f;  
    float det_box_thresh = 0.15f;  
    float det_unclip_ratio = 3.0f;  
    int det_max_candidates = 100;  
    float det_min_box_size = 1.0f;  
    std::string det_output_name = "";  
    int det_map_h = 960;  
    int det_map_w = 544;  
    bool letterbox_fix = true;  
  
    // Recognition parameters  
    std::string rec_output_name = "";  
    std::string charset_path = "";  
    std::vector<std::string> charset;  
    int blank_index = 0;  
    bool logits_are_softmax = false;  
    bool time_major = false;  
    float text_conf_smooth = 0.9f;  
    bool attach_caption_box = false;  
    
    // Spell correction parameters
    std::string frequency_dict_path = "";  
    int max_edit_distance = 2;  
    std::unordered_map<std::string, uint64_t> frequency_dict;  // word -> frequency
};  
  
// Function declarations following the post-processing pattern  
__BEGIN_DECLS  
  
// Initialization and cleanup functions  
OcrParams *init(const std::string config_path, const std::string function_name);  
void free_resources(void *params_void_ptr);  
  
// Main post-processing functions  
void paddleocr_det(HailoROIPtr roi, void *params_void_ptr);  
void paddleocr_recognize(HailoROIPtr roi, void *params_void_ptr);  
  
// Cropping function for text regions  
std::vector<HailoROIPtr> crop_text_regions(std::shared_ptr<HailoMat> image,
                                           HailoROIPtr roi,
                                           bool use_letterbox,
                                           bool no_scaling_bbox,
                                           bool internal_offset,
                                           const std::string &resize_method); 

void crop_text_regions_filter(HailoROIPtr roi, void *params_void_ptr);  

  
__END_DECLS

