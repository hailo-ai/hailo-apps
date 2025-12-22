
#include "toolbox.hpp"
#include "hailo_infer.hpp"
#include <iostream>
#include <chrono>
#include <mutex>
#include <future>
#include <iostream>
#include <chrono>
#include <mutex>
#include <future>
#include <filesystem>

#include <opencv2/highgui.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp>


#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
using namespace hailo_utils;


/////////// Constants ///////////
constexpr size_t MAX_QUEUE_SIZE = 60;

std::shared_ptr<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>> preprocessed_batch_queue_left =
    std::make_shared<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>>(MAX_QUEUE_SIZE);

std::shared_ptr<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>> preprocessed_batch_queue_right =
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
    const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &output_data_and_infos)
{
    if (output_data_and_infos.empty()) return;

    auto *data       = output_data_and_infos[0].first;
    const auto &info = output_data_and_infos[0].second;
    const auto &s    = info.shape; // height, width, features

    // Directly wrap output buffer into cv::Mat
    frame_to_draw = cv::Mat(s.height, s.width, CV_8U, data).clone();
}

struct StereoArgs {
    std::string net;
    std::string left;
    std::string right;
    bool save_stream_output;
    size_t batch_size;
    double framerate;
    std::string output_resolution;
    std::string output_dir;
    std::string camera_resolution;
};


static inline StereoArgs parse_stereo_args(int argc, char **argv) {

    std::string batch_str = getCmdOptionWithShortFlag(argc, argv, "--batch-size", "-b");
    std::string fps_str = getCmdOptionWithShortFlag(argc, argv, "--framerate", "-f");

    // Convert to proper types with defaults
    size_t batch_size = batch_str.empty() ? static_cast<size_t>(1) : static_cast<size_t>(std::stoul(batch_str));
    double framerate = fps_str.empty() ? 30.0 : std::stod(fps_str);
    std::string out_res_str = parse_output_resolution_arg(argc, argv);

    return {
        getCmdOptionWithShortFlag(argc, argv, "--net",   "-n"),
        getCmdOptionWithShortFlag(argc, argv, "--left",  "-l"),
        getCmdOptionWithShortFlag(argc, argv, "--right", "-r"),
        has_flag(argc, argv, "-s") || has_flag(argc, argv, "--save-stream-output"),
        batch_size,
        framerate,
        out_res_str,
        getCmdOptionWithShortFlag(argc, argv, "--output-dir", "-o"),
        getCmdOptionWithShortFlag(argc, argv, "--camera-resolution", "-cr"),
    };
}

int main(int argc, char** argv)
{
    const std::string APP_NAME = "depth_estimation_stereo";
    std::chrono::duration<double> inference_time;
    std::chrono::time_point<std::chrono::system_clock> t_start = std::chrono::high_resolution_clock::now();
    double org_height, org_width;
    cv::VideoCapture capture;
    size_t frame_count;
    InputType input_type;


    StereoArgs args = parse_stereo_args(argc, argv);

        if (hailo_utils::has_flag(argc, argv, "--list-nets")) {
        hailo_utils::list_networks(APP_NAME);
        return 0;
    }

    args.net = hailo_utils::resolve_net_arg(APP_NAME, args.net);
    HailoInfer model(args.net, args.batch_size);
    input_type = determine_input_type(args.left,
                                    std::ref(capture),
                                    std::ref(org_height),
                                    std::ref(org_width),
                                    std::ref(frame_count),
                                    std::ref(args.batch_size),
                                    std::ref(args.camera_resolution));

    auto preprocess_thread_input1 = std::async(run_preprocess,
                                        args.left,
                                        std::ref(args.net),
                                        std::ref(model),
                                        std::ref(input_type),
                                        std::ref(capture),
                                        std::ref(args.batch_size),
                                        std::ref(args.framerate),
                                        preprocessed_batch_queue_left,
                                        preprocess_frames);

    auto preprocess_thread_input2 = std::async(run_preprocess,
                                        args.right,
                                        std::ref(args.net),
                                        std::ref(model),
                                        std::ref(input_type),
                                        std::ref(capture),
                                        std::ref(args.batch_size),
                                        std::ref(args.framerate),
                                        preprocessed_batch_queue_right,
                                        preprocess_frames);

    ModelInputQueuesMap input_queues = {
        { model.get_infer_model()->get_input_names().at(0), preprocessed_batch_queue_left },
        { model.get_infer_model()->get_input_names().at(1), preprocessed_batch_queue_right }
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
        preprocess_thread_input1, "Preprocess_left_input",
        inference_thread,         "Inference",
        output_parser_thread,     "Postprocess",
        &preprocess_thread_input2, "Preprocess_right_input"

    );

    if (HAILO_SUCCESS != status) {
        return status;
    }

    std::chrono::time_point<std::chrono::system_clock> t_end = std::chrono::high_resolution_clock::now();
    print_inference_statistics(inference_time, args.net, frame_count, t_end - t_start);

    return HAILO_SUCCESS;
}
