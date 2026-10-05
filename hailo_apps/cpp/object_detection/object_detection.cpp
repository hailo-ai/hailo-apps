#include "toolbox.hpp"
#include "hailo_infer.hpp"
#include "hailo/hailort.hpp"
#include "utils.hpp"

using namespace hailo_utils;
using Clock = std::chrono::steady_clock;

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

// Task-specific postprocessing callback
void postprocess_callback(
    cv::Mat &frame_to_draw,
    const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &output_data_and_infos,
    const VisualizationParams &vis)
{
    std::vector<NamedBbox> bboxes;
    if (output_data_and_infos.size() == 1) {
        // NMS-embedded model (e.g. yolov8m): single packed output
        bboxes = parse_nms_data(output_data_and_infos[0].first, 80);
    } else {
        // Raw anchor-free model (e.g. yolo26n/s/m): 6 FLOAT32 tensors
        bboxes = decode_anchor_free(output_data_and_infos, vis);
    }
    draw_bounding_boxes(frame_to_draw, bboxes, vis);
}


int main(int argc, char** argv)
{
    try {
        const std::string APP_NAME = "object_detection";
        std::chrono::duration<double> inference_time;
        auto t_start = Clock::now();

        double org_height, org_width;
        cv::VideoCapture capture;
        size_t frame_count;
        InputType input_type;

        CommandLineArgs args = parse_command_line_arguments(argc, argv);
        post_parse_args(APP_NAME, args, argc, argv);

        // Inspect the HEF file (no VDevice needed) to count outputs.
        // Raw anchor-free models (yolo26n/s/m) have 6 outputs and need FLOAT32
        // so HailoRT dequantizes the tensors before we receive them.
        // NMS-embedded models have 1 output and work fine with AUTO.
        auto hef_inspect = hailort::Hef::create(args.net);
        if (!hef_inspect) throw std::runtime_error("Failed to parse HEF: " + args.net);
        size_t num_outputs = hef_inspect->get_output_vstream_infos().release().size();
        bool is_raw_tensor = num_outputs > 1;

        HailoInfer model(args.net, args.batch_size,
                         HAILO_FORMAT_TYPE_AUTO,
                         is_raw_tensor ? HAILO_FORMAT_TYPE_FLOAT32 : HAILO_FORMAT_TYPE_AUTO);

        // Load visualization config params
        VisualizationParams vis_param = load_visualization_params("visualization_config.yaml");
        validate_visualization_params(vis_param, AppVisMode::object_detection);

        auto post_cb = std::bind(postprocess_callback,
                                 std::placeholders::_1,
                                 std::placeholders::_2,
                                 std::cref(vis_param));

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
                                    post_cb);

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