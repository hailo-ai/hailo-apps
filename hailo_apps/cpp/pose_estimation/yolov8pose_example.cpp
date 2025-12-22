/**
 * Copyright (c) 2020-2025 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/
/**
 * @file async_infer_basic_example.cpp
 * This example demonstrates the Async Infer API usage with a specific model.
 **/
#include "toolbox.hpp"
//using namespace hailo_utils;
#include <iomanip>
#include <cstring>
#include "hailo_infer.hpp"
#include "hailo/hailort.hpp"
#include "common.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "general/hailo_objects.hpp"
#include "yolov8pose_postprocess.hpp"

#include <iostream>
#include <chrono>
#include <mutex>
#include <future>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace hailo_utils;

namespace fs = std::filesystem;



/////////// Constants ///////////
constexpr size_t MAX_QUEUE_SIZE = 60;

std::shared_ptr<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>> preprocessed_batch_queue =
    std::make_shared<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>>(MAX_QUEUE_SIZE);

std::shared_ptr<BoundedTSQueue<InferenceResult>> results_queue =
    std::make_shared<BoundedTSQueue<InferenceResult>>(MAX_QUEUE_SIZE);


// Task-specific preprocessing callback
void preprocess_callback(const std::vector<cv::Mat>& org_frames,
                         std::vector<cv::Mat>& preprocessed_frames,
                         uint32_t target_width, uint32_t target_height)
{
    preprocessed_frames.clear();
    preprocessed_frames.reserve(org_frames.size());

    for (const auto &src_bgr : org_frames) {
        // Skip invalid frames but keep vector alignment (optional: push empty)
        if (src_bgr.empty()) {
            preprocessed_frames.emplace_back();
            continue;
        }
        cv::Mat rgb;
        // 1) Convert to RGB
        if (src_bgr.channels() == 3) {
            cv::cvtColor(src_bgr, rgb, cv::COLOR_BGR2RGB);
        } else if (src_bgr.channels() == 4) {
            // If someone passed BGRA, drop alpha
            cv::cvtColor(src_bgr, rgb, cv::COLOR_BGRA2RGB);
        } else if (src_bgr.channels() == 1) {
            // If grayscale sneaks in, promote to 3 channels
            cv::cvtColor(src_bgr, rgb, cv::COLOR_GRAY2RGB);
        } else {
            // Fallback: force 3 channels by duplicating/merging
            std::vector<cv::Mat> ch(3, src_bgr);
            cv::merge(ch, rgb);
            cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB); // ensure RGB order
        }
        // 2) Resize to target
        if (rgb.cols != static_cast<int>(target_width) || rgb.rows != static_cast<int>(target_height)) {
            cv::resize(rgb, rgb, cv::Size(static_cast<int>(target_width),
                                          static_cast<int>(target_height)),
                       0.0, 0.0, cv::INTER_AREA);
        }
        // 3) Ensure contiguous buffer
        if (!rgb.isContinuous()) {
            rgb = rgb.clone();
        }
        // 4) Push to output vector
        preprocessed_frames.push_back(std::move(rgb));
    }
}
void postprocess_callback(
    cv::Mat &frame_to_draw,
    const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &output_data_and_infos)
{
    const int org_width  = frame_to_draw.cols;
    const int org_height = frame_to_draw.rows;

    // 1) ROI
    HailoROIPtr roi = std::make_shared<HailoROI>(HailoBBox(0.0f, 0.0f, 1.0f, 1.0f));

    // 2) Sort like the old pipeline:
    //    small→large by area (20→40→80), then features 64→1→51
    auto tensors = output_data_and_infos;

    // 3) Add tensors ONCE (HailoTensor ctor takes hailo_tensor_metadata_t)
    for (const auto &p : tensors) {
        roi->add_tensor(std::make_shared<HailoTensor>(p.first, p.second));
    }
    // 4) Pose (requires dtype-aware decoder in yolov8pose_postprocess)
    auto keypoints_and_pairs = yolov8(roi);

    // 5) Detections
    auto detections = hailo_common::get_hailo_detections(roi);
    for (const auto &det : detections) {
        if (!det || det->get_confidence() <= 0.0f) continue;
        const HailoBBox b = det->get_bbox();
        const cv::Point2f p1(b.xmin()*org_width, b.ymin()*org_height);
        const cv::Point2f p2(b.xmax()*org_width, b.ymax()*org_height);
        cv::rectangle(frame_to_draw, p1, p2, cv::Scalar(0,0,255), 1);

        std::ostringstream oss;
        oss << det->get_label() << " " << std::fixed << std::setprecision(2)
            << (det->get_confidence()*100.0f) << "%";
        cv::putText(frame_to_draw, oss.str(), p1 + cv::Point2f(0,-3),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,0,255), 1, cv::LINE_AA);
    }

    for (PairPairs &p : keypoints_and_pairs.second) {
        float x1 = p.pt1.first * float(org_width);
        float y1 = p.pt1.second * float(org_height);
        float x2 = p.pt2.first * float(org_width);
        float y2 = p.pt2.second * float(org_height);
        cv::line(frame_to_draw, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 255), 3);
    }

    for (auto &keypoint : keypoints_and_pairs.first) {
        float x = keypoint.xs * float(org_width);
        float y = keypoint.ys * float(org_height);
        cv::circle(frame_to_draw, cv::Point(x, y), 3, cv::Scalar(255, 0, 0), -1);
    }
}


int main(int argc, char** argv)
{
    const std::string APP_NAME = "pose_estimation";
    std::chrono::duration<double> inference_time;
    std::chrono::time_point<std::chrono::system_clock> t_start = std::chrono::high_resolution_clock::now();
    double org_height, org_width;
    cv::VideoCapture capture;
    size_t frame_count;
    InputType input_type;

    CommandLineArgs args = parse_command_line_arguments(argc, argv);
    post_parse_args(APP_NAME, args, argc, argv);
    HailoInfer model(args.net, args.batch_size);
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