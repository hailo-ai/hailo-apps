/**
 * Copyright (c) 2020-2025 Hailo Technologies Ltd.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/
/**
 * @file onnxrt_hailo_pipeline.cpp
 * Instance segmentation with Hailo -> ONNXRuntime postprocessing,
 * updated to the new queues map API.
 **/
#include "toolbox.hpp"
using namespace hailo_utils;

#include "hailo_infer.hpp"
#include "instance_seg_postprocess.hpp"
#include "onnx_decode.hpp"

/////////// Constants ///////////
constexpr size_t MAX_QUEUE_SIZE = 60;

std::shared_ptr<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>> preprocessed_batch_queue =
    std::make_shared<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>>(MAX_QUEUE_SIZE);

std::shared_ptr<BoundedTSQueue<InferenceResult>> results_queue =
    std::make_shared<BoundedTSQueue<InferenceResult>>(MAX_QUEUE_SIZE);

struct InstanceSegArgs : public CommandLineArgs {
    std::string decode_onnx;
};

InstanceSegArgs parse_instance_seg_args(int argc, char** argv)
{
    InstanceSegArgs a;
    a.net   = getCmdOptionWithShortFlag(argc, argv, "--net", "-n");
    a.input = getCmdOptionWithShortFlag(argc, argv, "--input", "-i");

    std::string bs = getCmdOptionWithShortFlag(argc, argv, "--batch-size", "-b");
    a.batch_size   = bs.empty() ? 1 : std::stoul(bs);

    std::string fps = getCmdOptionWithShortFlag(argc, argv, "--framerate", "-f");
    a.framerate     = fps.empty() ? 30.0 : std::stod(fps);
    a.save_stream_output = has_flag(argc, argv, "-s") || has_flag(argc, argv, "--save-stream-output");
    a.no_display         = has_flag(argc, argv, "--no-display");
    a.output_dir        = getCmdOptionWithShortFlag(argc, argv, "--output-dir", "-o");
    a.camera_resolution = getCmdOptionWithShortFlag(argc, argv, "--camera-resolution", "-cr");
    
    std::string out_res_str = parse_output_resolution_arg(argc, argv);
    a.output_resolution = out_res_str;
    a.decode_onnx       = getCmdOptionWithShortFlag(argc, argv, "--onnx", "-x");

    return a;
}


int main(int argc, char** argv)
{
    try{
        const std::string APP_NAME = "onnxrt_hailo_pipeline";
        std::chrono::duration<double> inference_time;
        auto t_start = Clock::now();
        double org_height, org_width;
        cv::VideoCapture capture;
        size_t frame_count;
        InputType input_type;

        InstanceSegArgs args = parse_instance_seg_args(argc, argv);
        if (args.decode_onnx.empty() &&
            !has_flag(argc, argv, "--list-models") &&
            !has_flag(argc, argv, "--list-inputs")) {
            std::cerr << "ERROR: Missing ONNX postprocessing model. "
                      << "Provide --onnx <path> (or -x <path>) pointing to a decode-only "
                      << "ONNX model compatible with the selected HEF's outputs.\n"
                      << "Example: ./build/onnxrt_hailo_pipeline --net /path/to/model.hef "
                      << "--onnx /path/to/postprocess.onnx --input /path/to/video.mp4\n";
            return HAILO_INVALID_ARGUMENT;
        }
        post_parse_args(APP_NAME, args, argc, argv);

        HailoInfer model(args.net, args.batch_size);
        const int model_w = model.get_model_shape().width;
        const int model_h = model.get_model_shape().height;

        InstanceSegDecodeONNX decoder(args.decode_onnx);
        input_type = determine_input_type(args.input,
                                        std::ref(capture),
                                        std::ref(org_height),
                                        std::ref(org_width),
                                        std::ref(frame_count),
                                        std::ref(args.batch_size),
                                        std::ref(args.camera_resolution));

        // Letterbox preprocess: scale to fit (preserve aspect ratio), pad right/bottom.
        // Must match compute_letterbox() in onnx_decode.cpp so coordinate inversion is correct.
        auto letterbox_preprocess = [](const std::vector<cv::Mat>& org_frames,
                                       std::vector<cv::Mat>& out_frames,
                                       uint32_t target_w, uint32_t target_h) {
            out_frames.clear();
            out_frames.reserve(org_frames.size());
            for (const auto& src : org_frames) {
                if (src.empty()) { out_frames.emplace_back(); continue; }
                cv::Mat rgb;
                switch (src.channels()) {
                    case 4:  cv::cvtColor(src, rgb, cv::COLOR_BGRA2RGB); break;
                    case 1:  cv::cvtColor(src, rgb, cv::COLOR_GRAY2RGB); break;
                    default: cv::cvtColor(src, rgb, cv::COLOR_BGR2RGB);  break;
                }
                const float scale = std::min(
                    static_cast<float>(target_w) / rgb.cols,
                    static_cast<float>(target_h) / rgb.rows);
                const int new_w = static_cast<int>(std::round(rgb.cols * scale));
                const int new_h = static_cast<int>(std::round(rgb.rows * scale));
                cv::Mat scaled;
                cv::resize(rgb, scaled, cv::Size(new_w, new_h));
                cv::Mat fitted;
                cv::copyMakeBorder(scaled, fitted,
                                   0, static_cast<int>(target_h) - new_h,
                                   0, static_cast<int>(target_w) - new_w,
                                   cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
                if (!fitted.isContinuous()) fitted = fitted.clone();
                out_frames.push_back(std::move(fitted));
            }
        };

        auto preprocess_thread = std::async(run_preprocess,
                                            std::ref(args.input),
                                            std::ref(args.net),
                                            std::ref(model),
                                            std::ref(input_type),
                                            std::ref(capture),
                                            std::ref(args.batch_size),
                                            std::ref(args.framerate),
                                            preprocessed_batch_queue,
                                            letterbox_preprocess);

        ModelInputQueuesMap input_queues = {
            { model.get_infer_model()->get_input_names().at(0), preprocessed_batch_queue }
        };

        auto inference_thread = std::async(run_inference_async,
                                        std::ref(model),
                                        std::ref(inference_time),
                                        std::ref(input_queues),
                                        results_queue);

        PostprocessCallback post_cb = [&](cv::Mat &frame_to_draw,
                                        const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &outputs)
        {
            auto decoded = decoder.run(outputs);

            const int channels_per_detection = (int)decoded.output_shape[1];
            const int num_detections         = (int)decoded.output_shape[2];
            const int num_classes            = 80;  // COCO-80
            const int mask_coeffs            = channels_per_detection - 4 - num_classes;

            auto dets_coeffs = parse_onnx_output(
                decoded.output,
                model_w, model_h,
                /*score_thr=*/0.6f,
                frame_to_draw.cols, frame_to_draw.rows,
                channels_per_detection, num_detections, num_classes, mask_coeffs);

            auto kept = nms_pairs(std::move(dets_coeffs), /*IoU*/0.7f, /*cross-class NMS*/true);

            const int P_C = (int)decoded.proto_shape[1];
            const int P_H = (int)decoded.proto_shape[2];
            const int P_W = (int)decoded.proto_shape[3];
            xt::xarray<float> proto = xt::zeros<float>({(size_t)P_H,(size_t)P_W,(size_t)P_C});
            {
                size_t idx = 0;
                for (int c=0;c<P_C;++c)
                for (int h=0;h<P_H;++h)
                for (int w=0;w<P_W;++w)
                    proto(h,w,c) = decoded.proto[idx++];
            }

            // Build masks in original frame space and draw
            auto dets_masks = decode_masks(kept, proto,
                                        /*org_h=*/frame_to_draw.rows,
                                        /*org_w=*/frame_to_draw.cols,
                                        /*model_h=*/model_h, /*model_w=*/model_w,
                                        /*proto_channels=*/P_C);

            std::vector<HailoDetection> dets;
            std::vector<cv::Mat> masks;
            dets.reserve(dets_masks.size());
            masks.reserve(dets_masks.size());
            for (auto &dm : dets_masks) {
                dets.push_back(dm.detection);
                masks.push_back(dm.mask);
            }

            draw_masks_and_boxes(frame_to_draw, dets, masks, 0.7f, 0.5f);
        };

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
