/**
 * Copyright 2021 (C) Hailo Technologies Ltd.
 * All rights reserved.
 *
 * Hailo Technologies Ltd. ("Hailo") disclaims any warranties, including, but not limited to,
 * the implied warranties of merchantability and fitness for a particular purpose.
 * This software is provided on an "AS IS" basis, and Hailo has no obligation to provide maintenance,
 * support, updates, enhancements, or modifications.
 *
 * You may use this software in the development of any project.
 * You shall not reproduce, modify or distribute this software without prior written permission.
 **/
/**
 * @ file semseg
 * This example demonstrates using virtual streams over c++
 **/

#include "toolbox.hpp"
#include "hailo_infer.hpp"
#include "hailo/hailort.hpp"

#include <iostream>
#include <chrono>
#include <mutex>
#include <future>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "cityscape_labels.hpp"
using namespace hailo_utils;


/////////// Constants ///////////
constexpr size_t MAX_QUEUE_SIZE = 60;

std::shared_ptr<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>> preprocessed_batch_queue =
    std::make_shared<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>>(MAX_QUEUE_SIZE);

std::shared_ptr<BoundedTSQueue<InferenceResult>> results_queue =
    std::make_shared<BoundedTSQueue<InferenceResult>>(MAX_QUEUE_SIZE);


/**
 * @brief Preprocess frames for inference.
*
 * Input can be PNG, JPEG, or video frames (OpenCV Mat).
 *
 * @param org_frames         Input frames (OpenCV Mats).
 * @param preprocessed_frames Output vector of processed frames.
 * @param target_width       Desired frame width.
 * @param target_height      Desired frame height.
 */
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
            cv::resize(rgb, rgb, cv::Size(static_cast<int>(target_width), static_cast<int>(target_height)),
                       0.0, 0.0, cv::INTER_LINEAR);
        }
        // 3) Ensure contiguous buffer
        if (!rgb.isContinuous()) {
            rgb = rgb.clone();
        }
        // 4) Push to output vector
        preprocessed_frames.push_back(std::move(rgb));
    }
}


/**
 * @brief Semantic segmentation post-process.
 * Post-process H×W×1 UINT8 label map → Cityscapes BGR mask (CV_8UC3) with blur+scale (1.6,+10), overwriting frame_to_draw.
 * @param frame_to_draw Image to annotate in-place.
 * @param output_data_and_infos Vector of pairs {data_ptr, vstream_info}.
 */
void postprocess_callback(
    cv::Mat &frame_to_draw,
    const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &outs)
{
    if (outs.size() != 1) return;

    const auto &buf   = outs[0].first;
    const auto &info  = outs[0].second;

    if (!buf) return;
    const int H = static_cast<int>(info.shape.height);
    const int W = static_cast<int>(info.shape.width);
    const int C = static_cast<int>(info.shape.features);
    if (H <= 0 || W <= 0 || C != 1) return; // supports HxWx1 label map only

    const uint8_t *labels = buf;

    // Build flat BGR lookup table once (pre-scaled ×1.6 +10, clamped to uint8)
    static uint8_t lut[256][3];
    static bool lut_ready = false;
    if (!lut_ready) {
        CityScapeLabels pal;
        for (int i = 0; i < 256; ++i) {
            cv::Vec3f c = pal.id_2_color(static_cast<uint8_t>(i));
            lut[i][0] = cv::saturate_cast<uchar>(c[0] * 1.6f + 10.f);
            lut[i][1] = cv::saturate_cast<uchar>(c[1] * 1.6f + 10.f);
            lut[i][2] = cv::saturate_cast<uchar>(c[2] * 1.6f + 10.f);
        }
        lut_ready = true;
    }

    cv::Mat colored(H, W, CV_8UC3);
    uint8_t *dst = colored.ptr<uint8_t>(0);
    for (int i = 0; i < H * W; ++i) {
        const uint8_t *e = lut[labels[i]];
        dst[i*3+0] = e[0];
        dst[i*3+1] = e[1];
        dst[i*3+2] = e[2];
    }
    cv::resize(colored, frame_to_draw, frame_to_draw.size(), 0, 0, cv::INTER_NEAREST);
}


int main(int argc, char** argv)
{
    try{
        const std::string APP_NAME = "semantic_segmentation";
        std::chrono::duration<double> inference_time;
        auto t_start = Clock::now();
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
                                    std::ref(args.no_display),
                                    std::ref(args.output_dir),
                                    std::ref(args.output_resolution),
                                    results_queue,
                                    input_queues,
                                    postprocess_callback);

        hailo_status status = wait_and_check_threads(
            preprocess_thread,    "Preprocess",
            inference_thread,     "Inference",
            output_parser_thread, "Postprocess "
        );
        if (HAILO_SUCCESS != status) {
            return status;
        }
        
        auto t_end = Clock::now();
        print_inference_statistics(inference_time, args.net, static_cast<double>(frame_count), t_end - t_start);

        return HAILO_SUCCESS;
    }
    catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return HAILO_INTERNAL_FAILURE;
    }
}