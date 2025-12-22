#ifndef _HAILO_COMMON_TOOLBOX_HPP_
#define _HAILO_COMMON_TOOLBOX_HPP_

#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <future>
#include <filesystem>
#include <opencv2/highgui.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp> 
#include "hailo/hailort.h"
#include "hailo/infer_model.hpp" 
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include "hailo_infer.hpp"



// -------------------------- COLOR MACROS --------------------------
#define RESET         "\033[0m"
#define MAGENTA       "\033[35m"
#define BOLDGREEN     "\033[1m\033[32m"
#define BOLDBLUE      "\033[1m\033[34m"
#define BOLDMAGENTA   "\033[1m\033[35m"

extern std::vector<cv::Scalar> COLORS;

namespace hailo_utils {

    namespace fs = std::filesystem;
    // Resolution table
    extern const std::unordered_map<std::string, std::pair<int,int>> RESOLUTION_MAP;
    // Resolved paths (toolbox resolves automatically at startup)
    extern const fs::path GET_HEF_BASH_SCRIPT_PATH;
    extern const fs::path GET_INPUT_BASH_SCRIPT_PATH;


    struct InferenceResult {
        cv::Mat org_frame;
        std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> output_data_and_infos;
        std::vector<std::shared_ptr<uint8_t>> output_guards;
    };

    struct InputType {
        bool is_image = false;
        bool is_video = false;
        bool is_directory = false;
        bool is_camera = false;
    };

    struct CommandLineArgs {
        std::string net;
        std::string input;
        std::string output_dir;
        std::string camera_resolution;
        std::string output_resolution;
        bool save_stream_output;
        size_t batch_size;
        double framerate;
    };
    
    // Callback types for task-specific processing
    using PreprocessCallback = std::function<void(const std::vector<cv::Mat>&, std::vector<cv::Mat>&, uint32_t, uint32_t)>;
    using PostprocessCallback = std::function<void(cv::Mat&, const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>>&)>;

    // Status/Error
    hailo_status check_status(const hailo_status &status, const std::string &message);
    hailo_status wait_and_check_threads(
        std::future<hailo_status> &f1, const std::string &name1,
        std::future<hailo_status> &f2, const std::string &name2,
        std::future<hailo_status> &f3, const std::string &name3,
        std::future<hailo_status> *f4 = nullptr, const std::string &name4 = std::string());

    // File / path
    bool is_image_file(const std::string &path);
    bool is_video_file(const std::string &path);
    bool is_directory_of_images(const std::string &path, size_t &entry_count, size_t batch_size);
    bool is_image(const std::string &path);
    bool is_video(const std::string &path);

    fs::path get_executable_dir();
    std::string get_hef_name(const std::string &path);
    InputType determine_input_type(const std::string& input_path,
                                cv::VideoCapture &capture,
                                double &org_height,
                                double &org_width,
                                size_t &frame_count,
                                size_t batch_size,
                                const std::string &camera_resolution);

    // CLI
    std::string getCmdOption(int argc, char *argv[], const std::string &option);
    bool has_flag(int argc, char *argv[], const std::string &flag);
    std::string getCmdOptionWithShortFlag(int argc, char *argv[],
                                        const std::string &longOption,
                                        const std::string &shortOption);
    CommandLineArgs parse_command_line_arguments(int argc, char **argv);
    void post_parse_args(const std::string &app, CommandLineArgs &args, int argc, char **argv);
    std::string parse_output_resolution_arg(int argc, char **argv);

    // Resolve -n/--net into a concrete .hef path.
    //
    // Accepted values:
    //   • Local HEF file:   path/to/model.hef
    //   • Model name:       logical name from networks.json (e.g., “yolov8n”)
    //                       → use --list-nets to see all available models.
    //
    // Behavior:
    //   - If a local .hef is provided: use it directly (after architecture check).
    //   - If a model name is provided:
    //        * Reuse existing downloaded HEF (if present), or
    //        * Download the correct HEF via get_hef.sh.
    //   - Returns the resolved absolute .hef path.
    std::string resolve_net_arg(const std::string &app,
                                const std::string &net_arg,
                                const std::string &dest_dir = ".");


    // Resolve -i/--input into a concrete input source.
    //
    // Accepted values:
    //   • Local file:      image.jpg, video.mp4
    //   • Directory:       folder/ (all images inside)
    //   • Camera:          "camera" or /dev/video*
    //   • Resource name:   logical ID from inputs.json (e.g., “bus”, “street”)
    //                      → use --list-inputs to see all available names.
    //
    // Behavior:
    //   - Uses local paths directly.
    //   - Downloads resource inputs via get_input.sh when needed.
    //   - Returns the resolved full path or "camera".
    std::string resolve_input_arg(const std::string &app,
                                const std::string &input_arg);  
                                
    // Print supported networks for this app (delegates to get_hef.sh list --app <app>)
    void list_networks(const std::string &app);

    // Print predefined inputs for this app (delegates to get_input.sh list --app <app>)
    void list_inputs(const std::string &app);
    std::string get_network_meta_value(const std::string &app,
                                    const std::string &model_name,
                                    const std::string &key,
                                    const std::string &sub_key = "");


    // ─────────────────────────────────────────────────────────────────────────────
    // DISPLAY & VIDEO
    // ─────────────────────────────────────────────────────────────────────────────

    void print_net_banner(const std::string &detection_model_name,
                        const std::vector<hailort::InferModel::InferStream> &inputs,
                        const std::vector<hailort::InferModel::InferStream> &outputs);
    void show_progress_helper(size_t current, size_t total);
    void show_progress(InputType &input_type, int progress, size_t frame_count);
    void print_inference_statistics(std::chrono::duration<double> inference_time,
                                    const std::string &hef_file,
                                    double frame_count,
                                    std::chrono::duration<double> total_time);
    void init_video_writer(const std::string &output_path, cv::VideoWriter &video,
        double framerate, int org_width, int org_height);

    cv::VideoCapture open_video_capture(const std::string &input_path,
        cv::VideoCapture &capture,
        double &org_height,
        double &org_width,
        size_t &frame_count,
        bool is_camera,
        const std::string &camera_resolution = "");

    template<typename T>
    class BoundedTSQueue {
    private:
        std::queue<T> m_queue;
        std::mutex m_mutex;
        std::condition_variable m_cond_not_empty;
        std::condition_variable m_cond_not_full;
        const size_t m_max_size;
        bool m_stopped;

    public:
        explicit BoundedTSQueue(size_t max_size) : m_max_size(max_size), m_stopped(false) {}
        ~BoundedTSQueue() { stop(); }

        BoundedTSQueue(const BoundedTSQueue&) = delete;
        BoundedTSQueue& operator=(const BoundedTSQueue&) = delete;

        void push(T item) {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cond_not_full.wait(lock, [this] { return m_queue.size() < m_max_size || m_stopped; });
            if (m_stopped) return;

            m_queue.push(std::move(item));
            m_cond_not_empty.notify_one();
        }

        bool pop(T &out_item) {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cond_not_empty.wait(lock, [this] { return !m_queue.empty() || m_stopped; });
            if (m_stopped && m_queue.empty()) {
                return false;
            }

            out_item = std::move(m_queue.front());
            m_queue.pop();
            m_cond_not_full.notify_one();
            return true;
        }

        void stop() {
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_stopped = true;
            }
            m_cond_not_empty.notify_all();
            m_cond_not_full.notify_all();
        }

        bool empty() const {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_queue.empty();
        }
    };


    using ModelInputQueuesMap = std::vector<
        std::pair<std::string,
            std::shared_ptr<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>>>>;


    // ─────────────────────────────────────────────────────────────────────────────
    // GENERIC PRE/POST PROCESSING FUNCTIONS
    // ─────────────────────────────────────────────────────────────────────────────

    // Generic preprocessing functions that accept callbacks
    void preprocess_video_frames(cv::VideoCapture &capture,
                            uint32_t &width, uint32_t &height, size_t &batch_size, double &framerate,
                            std::shared_ptr<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>> preprocessed_batch_queue,
                            PreprocessCallback preprocess_callback);

    void preprocess_image_frames(const std::string &input_path,
                            uint32_t &width, uint32_t &height, size_t &batch_size,
                            std::shared_ptr<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>> preprocessed_batch_queue,
                            PreprocessCallback preprocess_callback);

    void preprocess_directory_of_images(const std::string &input_path,
                                    uint32_t &width, uint32_t &height, size_t &batch_size,
                                    std::shared_ptr<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>> preprocessed_batch_queue,
                                    PreprocessCallback preprocess_callback);

    /**
     * @brief Preprocess frames for inference.
     *
     * Input can be PNG, JPEG, or video frames (OpenCV Mat).
     *
     * @param org_frames          Input frames (OpenCV Mats, BGR/BGRA/GRAY).
     * @param preprocessed_frames Output vector of processed RGB frames.
     * @param target_width        Desired frame width.
     * @param target_height       Desired frame height.
     */
    void preprocess_frames(const std::vector<cv::Mat> &org_frames,
                        std::vector<cv::Mat> &preprocessed_frames,
                        uint32_t target_width,
                        uint32_t target_height);

    // Generic postprocessing function that accepts callback
    hailo_status run_post_process(
        InputType &input_type,
        double &org_height,
        double &org_width,
        size_t &frame_count,
        cv::VideoCapture &capture,
        double &framerate,
        size_t &batch_size,
        const bool &save_stream_output,
        const std::string &output_dir,
        const std::string &output_resolution,
        std::shared_ptr<BoundedTSQueue<InferenceResult>> results_queue,
        PostprocessCallback postprocess_callback);

    hailo_status run_inference_async(
        HailoInfer &model,
        std::chrono::duration<double> &inference_time,
        ModelInputQueuesMap &model_input_queues,
        std::shared_ptr<BoundedTSQueue<InferenceResult>> results_queue);
    
    // Generic preprocessing runner that accepts callback
    hailo_status run_preprocess(const std::string& input_path,
                                const std::string& hef_path,
                                HailoInfer &model, 
                                InputType &input_type,
                                cv::VideoCapture &capture,
                                size_t &batch_size,
                                double &framerate,
                                std::shared_ptr<BoundedTSQueue<std::pair<std::vector<cv::Mat>,
                                std::vector<cv::Mat>>>> preprocessed_batch_queue,
                                PreprocessCallback preprocess_callback);

    // Resource management
    void release_resources(cv::VideoCapture &capture,
                            cv::VideoWriter &video,
                            InputType &input_type,
                            std::shared_ptr<BoundedTSQueue<std::pair<std::vector<cv::Mat>,
                            std::vector<cv::Mat>>>> preprocessed_batch_queue,
                            std::shared_ptr<BoundedTSQueue<InferenceResult>> results_queue);

} // namespace hailo_utils

#endif
