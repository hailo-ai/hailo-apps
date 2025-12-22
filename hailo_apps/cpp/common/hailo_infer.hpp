#ifndef _HAILO_ASYNC_INFERENCE_HPP_
#define _HAILO_ASYNC_INFERENCE_HPP_

#include "hailo/hailort.hpp"
#include <map>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace hailort;
using InputMap = std::map<std::string, std::vector<cv::Mat>>;
/**
 * @brief HailoInfer - Wrapper for HailoRT async inference
 * 
 * This class provides an interface for performing asynchronous inference
 * using HailoRT. It handles device management, model configuration,
 * input/output buffer management, and async job execution.
 * 
 */
class HailoInfer {
    private:
        std::unique_ptr<hailort::VDevice> vdevice;
        std::shared_ptr<hailort::InferModel> infer_model;
        hailort::ConfiguredInferModel configured_infer_model;
        std::vector<hailort::ConfiguredInferModel::Bindings> multiple_bindings;
        hailort::AsyncInferJob last_infer_job;
        std::map<std::string, hailo_vstream_info_t> output_vstream_info_by_name;
        size_t batch_size;

        // Private helper methods
        /**
         * @brief Sets input buffers for all model inputs
         * @param inputs  Map of input_name -> batch of inputs.
         * @param input_guards Vector to store input buffer guards
         */
        void set_input_buffers(
            const InputMap &inputs,
            std::vector<std::shared_ptr<cv::Mat>> &image_guards);

        /**
         * @brief Prepares output buffers for all model outputs
         * @param output_guards Vector to store output buffer guards
         * @return Vector of output data ptrs and their corresponding stream info
         */
        std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> prepare_output_buffers(
            std::vector<std::shared_ptr<uint8_t>> &output_guards);
        
        /**
         * @brief Executes async inference and manages the callback
         * @param output_data_and_infos Output data and stream info pairs
         * @param output_guards Output buffer guards
         * @param input_guards Input buffer guards
         * @param callback User-provided callback function for handling results
         */
        void run_async(
            const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &output_data_and_infos,
            const std::vector<std::shared_ptr<uint8_t>> &output_guards,
            const std::vector<std::shared_ptr<cv::Mat>> &input_image_guards,
            std::function<void(const hailort::AsyncInferCompletionInfo&,
                const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &,
                const std::vector<std::shared_ptr<uint8_t>> &)> callback);
    public:
        // Constructors and Destructor
        
        /**
         * @brief Constructor for single model on single device
         * @param hef_path Path to the HEF
         * 
         * Creates a VDevice and loads the specified model. This constructor
         * is used when you have only one model running on one device.
         */
        HailoInfer(const std::string &hef_path,
                size_t batch_size,
                hailo_format_type_t input_type = HAILO_FORMAT_TYPE_AUTO,
                hailo_format_type_t output_type = HAILO_FORMAT_TYPE_AUTO);
        
        /**
         * @brief Constructor for multiple models on the same device
         * @param hef_path Path to the HEF
         * @param group_id Unique identifier for device group management
         * 
         * Creates a VDevice with a specific group ID and loads the specified model.
         * This constructor is used when you have multiple models running on the same device
         * and need to manage them as a group.
         */
        HailoInfer(const std::string &hef_path,
                const std::string &group_id,
                size_t batch_size,
                hailo_format_type_t input_type = HAILO_FORMAT_TYPE_AUTO,
                hailo_format_type_t output_type = HAILO_FORMAT_TYPE_AUTO);

        // Getters
        
        /**
         * @brief Get input streams information
         * @return Reference to vector of input stream configurations
         */
        const std::vector<hailort::InferModel::InferStream>& get_inputs();
        
        /**
         * @brief Get output streams information
         * @return Reference to vector of output stream configurations
         */
        const std::vector<hailort::InferModel::InferStream>& get_outputs();
        
        /**
         * @brief Get ptr to the model
         * @return Ptr to the InferModel
         */
        const std::shared_ptr<hailort::InferModel> get_infer_model();
        
        /**
         * @brief Get the shape of the model's input layer. Assumes one input
         * @return Shape of the model's input layer
         */
        hailo_3d_image_shape_t get_model_shape();

        /**
         * @brief Get the size of the output vstream infos
         * @return Size of the output vstream infos
         */
        size_t get_output_vstream_infos_size();

        // Main inference method
        
        /**
         * @brief Perform async inference
         * @param inputs  Map of input_name -> batch of inputs.
         * @param callback User-defined callback function to handle inference results
         * 
         * This method performs asynchronous inference using the loaded model.
         * The inference runs in the background and results are delivered via the callback.
         * 
         * The callback receives:
         * - AsyncInferCompletionInfo: Status and timing information
         * - Vector of output data pointers and their stream info
         * - Vector of output buffer guards
         */
        void infer(
            const InputMap &inputs,
            std::function<void(const hailort::AsyncInferCompletionInfo&,
                            const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &,
                            const std::vector<std::shared_ptr<uint8_t>> &)> callback);
        /**
         * @brief Wait for the last inference job to complete
         */
        void wait_for_last_job();

};
#endif /* _HAILO_ASYNC_INFERENCE_HPP_ */