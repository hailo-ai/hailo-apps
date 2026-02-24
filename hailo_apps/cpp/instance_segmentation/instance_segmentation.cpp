/**
 * Copyright (c) 2020-2025 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/
/**
 * @file async_infer_basic_example.cpp
 * This example demonstrates the Async Infer API usage with a specific model.
 **/
#include "../common/toolbox.hpp"
#include "../common/hailo_infer.hpp"
#include "instance_seg_postprocess.hpp"

#include <cstdlib>
#include <cstring>

using namespace hailo_utils;
namespace fs = std::filesystem;

/////////// Constants ///////////
constexpr size_t MAX_QUEUE_SIZE = 60;
constexpr size_t WITH_NMS_OUTPUTS = 1;
constexpr size_t YOLOV8_OUTPUTS   = 10;

std::shared_ptr<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>> preprocessed_batch_queue =
    std::make_shared<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>>(MAX_QUEUE_SIZE);

std::shared_ptr<BoundedTSQueue<InferenceResult>> results_queue =
    std::make_shared<BoundedTSQueue<InferenceResult>>(MAX_QUEUE_SIZE);

enum class SegType {
    WithNms,
    YoloV8
};

SegType detect_seg_model_type(HailoInfer &model)
{
    size_t outputs = model.get_output_vstream_infos_size();

    std::cout << "Detected instance segmentation model with " << outputs << " outputs.\n";
    if (outputs == WITH_NMS_OUTPUTS)
        return SegType::WithNms;

    if (outputs == YOLOV8_OUTPUTS)
        return SegType::YoloV8;

    throw std::runtime_error(
        "Unsupported instance segmentation HEF. "
        "To see the supported models, run: --list-nets");
}

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

void postprocess_callback(cv::Mat &frame_to_draw,
                          const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &outputs,
                          const int model_w,
                          const int model_h,
                          SegType seg_model_type,
                          const VisualizationParams &vis_param)
{
    const int org_w = frame_to_draw.cols;
    const int org_h = frame_to_draw.rows;


    if (seg_model_type == SegType::WithNms) {
        const uint8_t *src_ptr = outputs.front().first;
        // Draw overlay+boxes directly in original resolution
        draw_detections_and_mask(src_ptr,
                                 model_w, model_h,
                                 org_w, org_h,
                                 frame_to_draw,
                                 vis_param);
        return;
    }

    if (seg_model_type == SegType::YoloV8) {
        auto roi = build_roi_from_outputs(outputs);
        std::vector<cv::Mat> masks = filter(roi,
                                            org_h, org_w,
                                            model_h, model_w,
                                            vis_param.score_thresh);

        auto dets = get_detections_from_roi(roi);
        draw_masks_and_boxes(frame_to_draw, dets, masks, vis_param);
        return;
    }
}


int main(int argc, char** argv)
{
    const std::string APP_NAME = "instance_segmentation";
    std::chrono::duration<double> inference_time;
    std::chrono::time_point<std::chrono::system_clock> t_start = std::chrono::high_resolution_clock::now();
    double org_height, org_width;
    cv::VideoCapture capture;
    size_t frame_count;
    InputType input_type;

    CommandLineArgs args = parse_command_line_arguments(argc, argv);
    post_parse_args(APP_NAME, args, argc, argv);
    HailoInfer model(args.net, args.batch_size);

    SegType seg_model_type = detect_seg_model_type(model);

    // Load Visualization config params
    VisualizationParams vis_param;
    try {
        vis_param = load_visualization_params("visualization_config.json");
    } catch (const std::exception &e) {
        std::cerr << "ERROR: failed to load visualization_config.json: "
                << e.what() << "\n";
        return EXIT_FAILURE;
    }

    validate_visualization_params(vis_param, AppVisMode::instance_seg);

    input_type = determine_input_type(std::ref(args.input),
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
                                    
    PostprocessCallback post_cb =
        std::bind(postprocess_callback,
                std::placeholders::_1,   // cv::Mat&
                std::placeholders::_2,   // outputs
                model.get_model_shape().width,
                model.get_model_shape().height,
                seg_model_type,
                std::cref(vis_param));

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
                                post_cb);

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