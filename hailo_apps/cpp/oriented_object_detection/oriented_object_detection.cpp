/**
 * Copyright (c) 2020-2025 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/
/**
 * @file oriented_object_detection.cpp
 * Oriented Object Detection example using YOLO11 OBB
 **/

#include "toolbox.hpp"
using namespace hailo_utils;

#include "hailo_infer.hpp"
#include "obb_utils.hpp"
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <numeric>
namespace fs = std::filesystem;

/////////// Constants ///////////
constexpr size_t MAX_QUEUE_SIZE = 60;
constexpr int IMG_SIZE = 640;
constexpr int CLS_NUM = 15;  // DOTAv1 dataset
constexpr float SCORE_THRESHOLD = 0.35f;
constexpr float NMS_IOU_THRESHOLD = 0.25f;
constexpr bool ENABLE_VISUALIZATION = true;  // Set to false to skip visualization and improve throughput

std::shared_ptr<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>> preprocessed_batch_queue =
    std::make_shared<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>>(MAX_QUEUE_SIZE);

std::shared_ptr<BoundedTSQueue<InferenceResult>> results_queue =
    std::make_shared<BoundedTSQueue<InferenceResult>>(MAX_QUEUE_SIZE);

// Task-specific preprocessing callback
void preprocess_callback(const std::vector<cv::Mat>& org_frames, 
                        std::vector<cv::Mat>& preprocessed_frames, 
                        uint32_t target_width, uint32_t target_height) {
    for (size_t i = 0; i < org_frames.size(); ++i) {
        const auto& frame = org_frames[i];
        
        // Calculate scale to maintain aspect ratio (letterbox)
        float scale = std::min(
            static_cast<float>(target_width) / frame.cols,
            static_cast<float>(target_height) / frame.rows
        );
        
        int new_w = static_cast<int>(frame.cols * scale);
        int new_h = static_cast<int>(frame.rows * scale);
        
        // Resize maintaining aspect ratio
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
        
        // Convert BGR to RGB (neural network expects RGB)
        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        
        // Create letterbox with gray padding (114, 114, 114)
        cv::Mat letterbox(target_height, target_width, rgb.type(), cv::Scalar(114, 114, 114));
        
        // Center the resized image
        int top = (target_height - new_h) / 2;
        int left = (target_width - new_w) / 2;
        rgb.copyTo(letterbox(cv::Rect(left, top, new_w, new_h)));
        preprocessed_frames.push_back(letterbox);
    }
}

// Task-specific postprocessing callback
void postprocess_callback(cv::Mat& frame_to_draw, 
                         const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>>& output_data_and_infos) {
    
    // CRITICAL FIX: Copy the output data immediately to avoid use-after-free
    // The pointers in output_data_and_infos may become invalid if processing takes too long
    std::vector<std::pair<std::vector<uint8_t>, hailo_vstream_info_t>> output_data_copies;
    output_data_copies.reserve(output_data_and_infos.size());
    
    for (const auto& [ptr, info] : output_data_and_infos) {
        size_t element_count = info.shape.height * info.shape.width * info.shape.features;
        // We requested FLOAT32 format, so data is always float32 (4 bytes per element)
        // even though info.format.type may still show the original HEF format
        size_t data_size = element_count * sizeof(float);
        
        std::vector<uint8_t> data_copy(ptr, ptr + data_size);
        output_data_copies.emplace_back(std::move(data_copy), info);
    }
    
    // Convert back to pointer format for obb_postprocess
    std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> safe_output_data;
    safe_output_data.reserve(output_data_copies.size());
    for (auto& [data_vec, info] : output_data_copies) {
        safe_output_data.emplace_back(data_vec.data(), info);
    }
    
    // 1. Run obb_postprocess with safe copied data
    cv::Mat postprocess_output;
    postprocess_output = obb_postprocess(
        safe_output_data,
        IMG_SIZE,
        CLS_NUM
    );
    
    // 2. Extract OBB detections
    std::vector<OBBDetection> detections = extract_obb_detections(
        postprocess_output,
        frame_to_draw.rows,
        frame_to_draw.cols,
        CLS_NUM,
        IMG_SIZE,
        SCORE_THRESHOLD
    );
    
    // 3. Apply rotated NMS and filter detections
    std::vector<size_t> keep_indices = rotated_nms(detections, NMS_IOU_THRESHOLD);
    std::vector<OBBDetection> kept_detections;
    for (size_t idx : keep_indices) {
        kept_detections.push_back(detections[idx]);
    }
    
    // 4. Draw on frame (only if visualization is enabled)
    if (ENABLE_VISUALIZATION) {
        draw_obb_detections(frame_to_draw, kept_detections);
    }
}

int main(int argc, char** argv)
{
    const std::string APP_NAME = "oriented_object_detection";

    std::chrono::duration<double> inference_time;
    std::chrono::time_point<std::chrono::system_clock> t_start = std::chrono::high_resolution_clock::now();
    double org_height, org_width;
    cv::VideoCapture capture;
    size_t frame_count;
    InputType input_type;

    CommandLineArgs args = parse_command_line_arguments(argc, argv);

    post_parse_args(APP_NAME, args, argc, argv);
    HailoInfer model(args.net, args.batch_size, HAILO_FORMAT_TYPE_UINT8, HAILO_FORMAT_TYPE_FLOAT32);
    input_type = determine_input_type(args.input,
                                    std::ref(capture),
                                    std::ref(org_height),
                                    std::ref(org_width),
                                    std::ref(frame_count),
                                    std::ref(args.batch_size),
                                    std::ref(args.camera_resolution));
    
    auto preprocess_thread = std::async(run_preprocess,
                                        std::ref(args.input),
                                        std::ref(args.net),
                                        std::ref(model),
                                        std::ref(input_type),
                                        std::ref(capture),
                                        std::ref(args.batch_size),
                                        std::ref(args.framerate),
                                        preprocessed_batch_queue,
                                        preprocess_callback);
                                        

    ModelInputQueuesMap input_queues = {
        { model.get_infer_model()->get_input_names().at(0), preprocessed_batch_queue }
    };
    auto inference_thread = std::async(run_inference_async,
                                    std::ref(model),
                                    std::ref(inference_time),
                                    std::ref(input_queues),
                                    results_queue);

    auto output_parser_thread = std::async(run_post_process,
                                std::ref(input_type),
                                std::ref(org_height),
                                std::ref(org_width),
                                std::ref(frame_count),
                                std::ref(capture),
                                std::ref(args.framerate),
                                std::ref(args.batch_size),
                                std::ref(args.save_stream_output),
                                std::ref(args.output_dir),
                                std::ref(args.output_resolution),
                                results_queue,
                                postprocess_callback);

    hailo_status status = wait_and_check_threads(
        preprocess_thread,    "Preprocess",
        inference_thread,     "Inference",
        output_parser_thread, "Postprocess "
    );
    
    if (HAILO_SUCCESS != status) {
        return status;
    }

    std::chrono::time_point<std::chrono::system_clock> t_end = std::chrono::high_resolution_clock::now();
    print_inference_statistics(inference_time, args.net, frame_count, t_end - t_start);

    return HAILO_SUCCESS;
}
