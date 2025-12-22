/**
 * Copyright (c) 2020-2025 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/
/**
 * @file async_infer_basic_example.cpp
 * This example demonstrates the Async Infer API usage with a specific model.
 **/
#include "../common/toolbox.hpp"
using namespace hailo_utils;

#include "../common/hailo_infer.hpp"
#include "instance_seg_postprocess.hpp"


#include <cstring>

namespace fs = std::filesystem;

/////////// Constants ///////////
constexpr size_t MAX_QUEUE_SIZE = 60;

std::shared_ptr<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>> preprocessed_batch_queue =
    std::make_shared<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>>(MAX_QUEUE_SIZE);

std::shared_ptr<BoundedTSQueue<InferenceResult>> results_queue =
    std::make_shared<BoundedTSQueue<InferenceResult>>(MAX_QUEUE_SIZE);


void postprocess_callback(cv::Mat &frame_to_draw,
                          const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &outputs,
                          const int model_w,
                          const int model_h,
                          const bool hef_has_nms_and_mask)
{
        if (hef_has_nms_and_mask) {
            // ===== packed NMS + byte masks on device =====
            const uint8_t *src_ptr = outputs.front().first;
            LetterboxMap map{};
            cv::Mat model_space = make_model_space_canvas(frame_to_draw, model_w, model_h, map);
            draw_detections_and_mask(src_ptr, model_w, model_h, model_space);
            map_model_to_frame(model_space, map, frame_to_draw);
        } else {
            // ===== raw heads NMS + mask reconstruction =====
            auto roi = build_roi_from_outputs(outputs);
            std::vector<cv::Mat> masks = filter(roi, model_w, model_h);
            auto dets = get_detections_from_roi(roi);
            LetterboxMap map{};
            cv::Mat model_space = make_model_space_canvas(frame_to_draw, model_w, model_h, map);
            draw_masks_and_boxes(model_space, dets, masks, /*alpha=*/0.7f, /*thresh=*/0.5f);
            map_model_to_frame(model_space, map, frame_to_draw);
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
    bool hef_with_nms_and_mask = model.get_output_vstream_infos_size() == 1;
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
                                        preprocess_frames);

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
                hef_with_nms_and_mask); 


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