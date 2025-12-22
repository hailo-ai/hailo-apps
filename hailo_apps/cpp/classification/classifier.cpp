/**
 * Copyright (c) 2020-2025 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/
/**
 * @file async_infer_basic_example.cpp
 * This example demonstrates the Async Infer API usage with a specific model.
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
#include "imagenet_labels.hpp"
using namespace hailo_utils;


/////////// Constants ///////////
constexpr size_t MAX_QUEUE_SIZE = 60;

std::shared_ptr<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>> preprocessed_batch_queue =
    std::make_shared<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>>(MAX_QUEUE_SIZE);

std::shared_ptr<BoundedTSQueue<InferenceResult>> results_queue =
    std::make_shared<BoundedTSQueue<InferenceResult>>(MAX_QUEUE_SIZE);

static bool APPLY_SOFTMAX = false;

/**
 * @brief Find the index of the maximum element in a vector.
 *
 * @tparam T Value type of the vector.
 * @param v Input vector.
 * @return Index of the maximum element.
 */
template <typename T>
static int argmax_vec(const std::vector<T> &v)
{
    return static_cast<int>(std::distance(v.begin(), std::max_element(v.begin(), v.end())));
}

/**
 * @brief Compute softmax probabilities from a vector of logits.
 *
 * @tparam T Value type of the vector.
 * @param v Input vector (raw scores or logits).
 * @return Vector of normalized probabilities summing to 1.0.
 */
template <typename T>
static std::vector<float> softmax_vec(const std::vector<T> &v)
{
    std::vector<float> out;
    out.reserve(v.size());
    float m = -INFINITY;
    for (auto &x : v) m = std::max<float>(m, static_cast<float>(x));
    float sum = 0.0f;
    for (auto &x : v) sum += std::exp(static_cast<float>(x) - m);
    for (auto &x : v) out.push_back(std::exp(static_cast<float>(x) - m) / sum);
    return out;
}

/**
 * @brief Classify logits or probabilities and format result string.
 *
 * @param logits Vector of raw logits or probabilities.
 * @param threshold Minimum confidence to accept; otherwise returns "N\A".
 * @return Formatted string: "<label> (<confidence>%)".
 */
static std::string classify_and_format(const std::vector<float> &logits,
                                       float threshold = 0.20f)
{
    static ImageNetLabels labels;
    std::vector<float> probs = APPLY_SOFTMAX
        ? softmax_vec(logits)
        : std::vector<float>(logits.begin(), logits.end());

    int idx = APPLY_SOFTMAX ? argmax_vec(probs) : argmax_vec(logits);
    float conf = probs.empty() ? 0.0f : probs[idx];

    if (conf < threshold) return std::string("N\\A");

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << conf * 100.0f;
    return labels.imagenet_labelstring(idx) + " (" + oss.str() + "%)";
}

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
    if (output_data_and_infos.empty()) {
        std::cout << "-W- postprocess_callback: no outputs\n";
        return;
    }

    const auto *data_ptr = output_data_and_infos[0].first;
    const auto &info     = output_data_and_infos[0].second;
    const size_t num_classes = static_cast<size_t>(info.shape.features);

    std::vector<float> probs;
    probs.reserve(num_classes);
    const float *f = reinterpret_cast<const float*>(data_ptr);
    for (size_t i = 0; i < num_classes; ++i) {
        probs.push_back(f[i]);
    }

    const std::string result = classify_and_format(probs, /*threshold=*/0.30f);
    // Overlay result on the frame (if provided)
    if (!frame_to_draw.empty()) {
        const int font = cv::FONT_HERSHEY_SIMPLEX;
        const double scale = 0.5;
        const int thickness = 1.3;
        int baseline = 0;
        (void)baseline;
        cv::Size textSize = cv::getTextSize(result, font, scale, thickness, &baseline);
        cv::Point origin(12, 16 + textSize.height);
        cv::putText(frame_to_draw, result, origin + cv::Point(1,1),
                    font, scale, cv::Scalar(0,0,0), thickness + 2, cv::LINE_AA); // shadow
        cv::putText(frame_to_draw, result, origin,
                    font, scale, cv::Scalar(255,255,255), thickness, cv::LINE_AA);
    }
}


int main(int argc, char** argv)
{
    const std::string APP_NAME = "classifier";
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

    // Determine model name from HEF path or plain name
    const std::string model_name = fs::path(args.net).stem().string();
    // Query JSON metadata to see if softmax should be applied
    APPLY_SOFTMAX = (hailo_utils::get_network_meta_value("classifier", model_name, "apply_softmax") == "true");

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
