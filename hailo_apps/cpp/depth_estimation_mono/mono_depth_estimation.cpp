
#include "toolbox.hpp"
#include "hailo_infer.hpp"
#include "hailo/hailort.hpp"

#include <iostream>
#include <chrono>
#include <mutex>
#include <future>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
using namespace hailo_utils;


/////////// Constants ///////////
constexpr size_t MAX_QUEUE_SIZE = 60;

std::shared_ptr<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>> preprocessed_batch_queue =
    std::make_shared<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>>(MAX_QUEUE_SIZE);

std::shared_ptr<BoundedTSQueue<InferenceResult>> results_queue =
    std::make_shared<BoundedTSQueue<InferenceResult>>(MAX_QUEUE_SIZE);

/**
 * @brief Post-process a classifier output and overlay the top-1 result.
 *
 * Expects a single classifier tensor (probabilities already in float32).
 * Picks the output with the largest feature count if multiple exist,
 * finds argmax, applies a confidence threshold, prints, and overlays text.
 *
 * @param frame_to_draw Image to annotate in-place.
 * @param output_data_and_infos Vector of pairs {data_ptr, vstream_info}.
 */
void postprocess_callback(
    cv::Mat &frame_to_draw,
    const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &out)
{
    if (out.empty()) return;
    const auto &info = out[0].second;
    if (info.shape.features != 1) return;

    const int H = (int)info.shape.height, W = (int)info.shape.width;
    float *ptr = reinterpret_cast<float*>(out[0].first);

    cv::Mat in(H, W, CV_32F, ptr), z;
    cv::exp(-in, z);                   // z = exp(-in)
    z = 1.0f / (1.0f + z);             // sigmoid
    z = 1.0f / (z * 10.0f + 0.009f);   // scale

    double mn, mx; cv::minMaxIdx(z, &mn, &mx);
    if (mx <= mn) mx = mn + 1e-6;      // avoid /0

    cv::Mat u8; z.convertTo(u8, CV_8U, 255.0/(mx-mn), -mn*255.0/(mx-mn));
    cv::applyColorMap(u8, u8, cv::COLORMAP_PLASMA);
    cv::resize(u8, frame_to_draw, frame_to_draw.size(), 0, 0, cv::INTER_LINEAR);
}


int main(int argc, char** argv)
{
    const std::string APP_NAME = "depth_estimation_mono";
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
                                        preprocess_frames);

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
