/**
 * Copyright (c) 2020-2023 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/

#include "hailo/hailort.hpp"
#include "common.h"
#include "tokenizer/nn_embeddings.hpp"
#include "../common/toolbox.hpp"

#include <iostream>
#include <future>
#include <mutex>
#include <fstream>
#include <functional>
#include <numeric>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp>

#if defined(__unix__)
#include <sys/mman.h>
#endif

std::mutex m;

using namespace hailort;

template <typename T> 
static std::shared_ptr<T> page_aligned_alloc(size_t size, void* buff = nullptr)
{
#if defined(__unix__)
    auto addr = mmap(buff, size, PROT_WRITE | PROT_READ, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (MAP_FAILED == addr) throw std::bad_alloc();
    return std::shared_ptr<T>(reinterpret_cast<T*>(addr), [size](void *addr) { munmap(addr, size); });
#elif defined(_MSC_VER)
    auto addr = VirtualAlloc(buff, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!addr) throw std::bad_alloc();
    return std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t*>(addr), [](void *addr){ VirtualFree(addr, 0, MEM_RELEASE); });
#else
#pragma error("Aligned alloc not supported")
#endif
}


std::vector<float> softmax(const std::vector<float>& probs) {
    std::vector<float> exp_probs(probs.size());
    float max_probs = *std::max_element(probs.begin(), probs.end());
    
    float sum_exp = 0.0;
    for (size_t i = 0; i < probs.size(); ++i) {
        exp_probs[i] = std::exp(probs[i] - max_probs);
        sum_exp += exp_probs[i];
    }
    
    for (size_t i = 0; i < exp_probs.size(); ++i) {
        exp_probs[i] /= sum_exp;
    }
    
    return exp_probs;
}


std::vector<float> dot(const std::vector<float>& image_embedding, 
                                        const std::vector<std::vector<float>>& text_embedding,
                                        float logit_scale) {
    std::vector<float> result(text_embedding.size());
 
    for (std::size_t i = 0; i < text_embedding.size(); ++i) {
        result[i] += std::inner_product(image_embedding.begin(), image_embedding.end(), text_embedding[i].begin(), 0.0f);
        result[i] = result[i] * logit_scale;
    }
 
    return result;
}


void normalize_vector(std::vector<float>& vec) {
    float norm = std::sqrt(std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0f));
 
    if (norm != 0.0f) {
        std::transform(vec.begin(), vec.end(), vec.begin(), [norm](float v) { return v / norm; });
    }
}

template <typename T>
void normalize(T& embeddings) {
    if constexpr (std::is_same<T, std::vector<float>>::value){
        normalize_vector(embeddings);
    }
    else if constexpr (std::is_same<T, std::vector<std::vector<float>>>::value){
        for (auto& embedding : embeddings) {
            normalize_vector(embedding);
        }
    }
    else {
        std::cerr << "Unsupported type for normalization" << std::endl;
    }
}

template <typename T>
void deqantize_output(std::vector<float>& output_data_buffer, T* output_data, std::shared_ptr<hailo_vstream_info_t> vstream_info) {
    for (size_t i = 0; i < output_data_buffer.size(); i++) {
        output_data_buffer[i] = static_cast<float>((static_cast<float>(output_data[i]) - vstream_info->quant_info.qp_zp) * vstream_info->quant_info.qp_scale);
    }
}

template <typename T>
hailo_status run_postprocess(const std::vector<std::string>& text_vec, TSQueue<std::vector<std::vector<float>>>& text_embeddings_queue,
                            TSQueue<std::vector<std::pair<T*, hailo_vstream_info_t>>>& inferred_data_queue, 
                            size_t frame_count){

    float logit_scale = 4.6051702f;
    logit_scale = std::exp(logit_scale);

    std::vector<std::vector<float>> dequantized_text_embeddings = text_embeddings_queue.pop();
    std::vector<float> dequantized_image_embedding;
    std::vector<float> raw_probs;
    std::vector<float> probs;

    for (size_t i = 0; i < frame_count; i++){
        auto output_data_and_infos = inferred_data_queue.pop();

        auto vstream_info_ptr = std::make_shared<hailo_vstream_info_t>(output_data_and_infos[0].second);
        dequantized_image_embedding.resize(vstream_info_ptr->shape.height * vstream_info_ptr->shape.width * vstream_info_ptr->shape.features);

        if constexpr (std::is_same<T, float32_t>::value){
            dequantized_image_embedding.assign(output_data_and_infos[0].first, output_data_and_infos[0].first + dequantized_image_embedding.size());
        }
        else {
            deqantize_output(std::ref(dequantized_image_embedding), output_data_and_infos[0].first, vstream_info_ptr);
        }

        normalize(dequantized_image_embedding);
        normalize(dequantized_text_embeddings);

        raw_probs = dot(dequantized_image_embedding, dequantized_text_embeddings, logit_scale);
        probs = softmax(std::ref(raw_probs));

        // Find the index of the maximum probability
        auto max_prob_iter = std::max_element(probs.begin(), probs.end());
        size_t max_prob_index = std::distance(probs.begin(), max_prob_iter);

        // Retrieve the corresponding prompt
        std::string best_prompt = text_vec[max_prob_index];

        // Print or store the best prompt
        std::cout << "Classification for frame " << i << ": " << best_prompt << std::endl;
    }

    return HAILO_SUCCESS;
}

template <typename T>
hailo_status run_inference(std::shared_ptr<hailort::InferModel> infer_model,
                            std::vector<std::promise<cv::Mat>>& frames_promises,
                            std::vector<std::future<cv::Mat>>& frames_futures,
                            size_t frame_count,
                            const std::vector<std::string>& output_names,
                            TSQueue<std::vector<std::pair<T*, hailo_vstream_info_t>>>& inferred_data_queue, 
                            std::chrono::duration<double>& inference_time) {

    if constexpr (std::is_same<T, float32_t>::value){
        infer_model->output()->set_format_type(HAILO_FORMAT_TYPE_FLOAT32);
    }
    else if constexpr (std::is_same<T, uint8_t>::value){
        infer_model->output()->set_format_type(HAILO_FORMAT_TYPE_UINT8);
    }
    else if constexpr (std::is_same<T, uint16_t>::value){
        infer_model->output()->set_format_type(HAILO_FORMAT_TYPE_UINT16);
    }
    else {
        std::cerr << "Unsupported type for output format" << std::endl;
    }

    size_t output_frame_size = infer_model->output()->get_frame_size();

    auto configured_infer_model = infer_model->configure();
    if (!configured_infer_model) {
        std::cerr << "Failed to create configured infer model, status = " << configured_infer_model.status() << std::endl;
        return configured_infer_model.status();
    }

    AsyncInferJob last_infer_job;

    std::shared_ptr<T> output_buffer;

    std::vector<std::shared_ptr<cv::Mat>> input_buffer_guards;
    std::vector<std::shared_ptr<T>> output_buffer_guards;

    auto bindings = configured_infer_model->create_bindings();
    if (!bindings) {
        std::cerr << "Failed to create infer bindings, status = " << bindings.status() << std::endl;
        return bindings.status();
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < frame_count; i++){

        auto preprocessed_image = std::make_shared<cv::Mat>(frames_futures[i].get());

        for (const auto &input_name : infer_model->get_input_names()) {
            size_t input_frame_size = infer_model->input(input_name)->get_frame_size();
            auto status = bindings->input(input_name)->set_buffer(MemoryView(preprocessed_image->data, input_frame_size));
            if (HAILO_SUCCESS != status) {
                std::cerr << "Failed to set infer input buffer, status = " << status << std::endl;
                return status;
            }

            input_buffer_guards.push_back(preprocessed_image);
        }

        std::vector<std::pair<T*, hailo_vstream_info_t>> output_data_and_infos;
        
        for (const auto &output_name : output_names) {
            output_buffer = page_aligned_alloc<T>(output_frame_size);
            auto status = bindings->output(output_name)->set_buffer(MemoryView(output_buffer.get(), output_frame_size));
            if (HAILO_SUCCESS != status) {
                std::cerr << "Failed to set infer output buffer, status = " << status << std::endl;
                return status;
            }

            output_data_and_infos.push_back(std::make_pair(
                                                            output_buffer.get(), 
                                                            infer_model->hef().get_output_vstream_infos().release()[0]
                                                        ));

            output_buffer_guards.push_back(output_buffer);
        }

        auto status = configured_infer_model->wait_for_async_ready(std::chrono::milliseconds(10000000));
        if (HAILO_SUCCESS != status) {
            std::cerr << "Failed to run wait_for_async_ready, status = " << status << std::endl;
            return status;
        }

        auto job = configured_infer_model->run_async(bindings.value(), 
                                                        [&inferred_data_queue, output_data_and_infos](const hailort::AsyncInferCompletionInfo& info){
            inferred_data_queue.push(output_data_and_infos);
        });

        if (!job) {
            std::cerr << "Failed to start async infer job, status = " << job.status() << std::endl;
            return job.status();
        }

        job->detach();
        if (i == frame_count - 1) {
            last_infer_job = job.release();
        }
    }

    auto status = last_infer_job.wait(std::chrono::milliseconds(10000000));
    if (HAILO_SUCCESS != status) {
        std::cerr << "Failed to wait for infer to finish, status = " << status << std::endl;
        return status;
    }

    auto end_time = std::chrono::high_resolution_clock::now();

    inference_time = end_time - start_time;

    return HAILO_SUCCESS;
}


void use_single_frame(std::vector<cv::Mat>& frames, 
                    std::vector<std::promise<cv::Mat>>& frames_promises,
                    cv::Mat& image, int frame_count, cv::Size shape){

    for(int i = 0; i < frame_count; i++) {
        std::lock_guard<std::mutex> lock(m);
        frames.push_back(image);

        cv::cvtColor(image, image, cv::ColorConversionCodes::COLOR_BGR2RGB);
        cv::resize(image, image, shape, 0, 0, cv::INTER_AREA);

        frames_promises[i].set_value(image);
    }
}

hailo_status run_preprocess(std::vector<cv::Mat>& frames, std::vector<std::promise<cv::Mat>>& frames_promises,
                            const std::string input_path, uint32_t height, uint32_t width, std::string& cmd_num_frames) {

    cv::Mat org_frame;

    if ((!cmd_num_frames.empty() && input_path.find(".avi") == std::string::npos && input_path.find(".mp4") == std::string::npos) || 
        (input_path.find(".jpg") != std::string::npos || input_path.find(".png") != std::string::npos || input_path.find(".bmp") != std::string::npos)){
        org_frame = cv::imread(input_path, cv::IMREAD_COLOR);
        int frame_count = !cmd_num_frames.empty() ? std::stoi(cmd_num_frames) : 1;
        use_single_frame(frames, frames_promises, std::ref(org_frame), frame_count, cv::Size(width, height));
        org_frame.release();
    }
    else { // Video or camera input
        cv::VideoCapture capture;
        if (input_path.empty()) {
            capture.open(0, cv::CAP_ANY);
            if (!capture.isOpened()) {
                throw "Unable to read camera input";
            }
        }
        else{
            capture.open(input_path, cv::CAP_ANY);
            if(!capture.isOpened())
                throw "Unable to read input file";
        }
        int i = 0;
        for(;;) {
            capture >> org_frame;
            if(org_frame.empty()) {
                break;
            }

            std::lock_guard<std::mutex> lock(m);
            frames.push_back(org_frame);

            
            cv::cvtColor(org_frame, org_frame, cv::ColorConversionCodes::COLOR_BGR2RGB); 
            cv::resize(org_frame, org_frame, cv::Size(width, height), 0, 0, cv::INTER_AREA);

            frames_promises[i].set_value(org_frame);

            i++;

        }
        capture.release();
    }

    return HAILO_SUCCESS;
}

template <typename T>
hailo_status configure_and_infer(const std::vector<std::string>& text_vec,
                                std::shared_ptr<hailort::InferModel> infer_model, 
                                TSQueue<std::vector<std::vector<float>>>& text_embeddings_queue,
                                const std::string input_path, 
                                std::chrono::time_point<std::chrono::system_clock>& start_time,
                                std::chrono::time_point<std::chrono::system_clock>& end_time, 
                                std::chrono::duration<double>& inference_time, size_t frame_count,
                                double org_height, double org_width, std::string cmd_img_num){

    std::vector<cv::Mat> frames;

    // The buffers are stored here as a guard for the memory. The buffer will be freed only after
    // configured_infer_model will be released.
    std::vector<std::shared_ptr<T>> buffer_guards;
    buffer_guards.reserve(infer_model->outputs().size());

    std::vector<std::string> output_names = infer_model->get_output_names();

    std::vector<std::promise<cv::Mat>> frames_promises(frame_count);
    std::vector<std::future<cv::Mat>> frames_futures(frame_count);

    for (size_t i = 0; i < frames_promises.size(); i++) {
        frames_futures[i] = frames_promises[i].get_future();
    }

    TSQueue<std::vector<std::pair<T*, hailo_vstream_info_t>>> inferred_data_queue;

    auto model_input_shape = infer_model->hef().get_input_vstream_infos().release()[0].shape;

    auto preprocess_thread(std::async(run_preprocess,
                                    std::ref(frames),
                                    std::ref(frames_promises),
                                    input_path, 
                                    model_input_shape.height, 
                                    model_input_shape.width,
                                    std::ref(cmd_img_num)));

    auto inference_thread(std::async(run_inference<T>,
                                    infer_model,
                                    std::ref(frames_promises),
                                    std::ref(frames_futures),
                                    frame_count,
                                    std::ref(output_names),
                                    std::ref(inferred_data_queue),
                                    std::ref(inference_time)));

    auto postprocess_thread = std::async(run_postprocess<T>,
                                        text_vec,
                                        std::ref(text_embeddings_queue),
                                        std::ref(inferred_data_queue),
                                        frame_count);

    auto preprocess_status = preprocess_thread.get();
    auto inference_status = inference_thread.get();
    auto postprocess_status = postprocess_thread.get();

    if (HAILO_SUCCESS != preprocess_status) {
        std::cerr << "Inference failed with status " << preprocess_status << std::endl;
        return preprocess_status; 
    }

    if (HAILO_SUCCESS != inference_status) {
        std::cerr << "Preprocess failed with status " << inference_status << std::endl;
        return inference_status; 
    }

    if (HAILO_SUCCESS != postprocess_status) {
        std::cerr << "Post-processing failed with status " << postprocess_status << std::endl;
        return postprocess_status;
    }

    return HAILO_SUCCESS;
}


void print_net_banner(const std::string model_name, 
                    const std::vector<hailort::InferModel::InferStream>& inputs, 
                    const std::vector<hailort::InferModel::InferStream>& outputs) {
    std::cout << BOLDMAGENTA << "-I-----------------------------------------------" << std::endl << RESET;
    std::cout << BOLDMAGENTA << "-I-  Network  Name                               " << std::endl << RESET;
    std::cout << BOLDMAGENTA << "-I-----------------------------------------------" << std::endl << RESET;
    std::cout << BOLDMAGENTA << "-I   " << model_name                               << std::endl << RESET;
    std::cout << BOLDMAGENTA << "-I-----------------------------------------------" << std::endl << RESET;
    for (auto& input: inputs) {
        auto shape = input.shape();
        std::cout << MAGENTA << "-I-  Input: " << input.name() 
        << ", Shape: (" << shape.height << ", " << shape.width << ", " << shape.features << ")" 
        << std::endl << RESET;
    }
    std::cout << BOLDMAGENTA << "-I-----------------------------------------------" << std::endl << RESET;
    for (auto& output: outputs) {
        auto shape = output.shape();
        std::cout << MAGENTA << "-I-  Output: " << output.name() 
        << ", Shape: (" << shape.height << ", " << shape.width << ", " << shape.features << ")" 
        << std::endl << RESET;
    }
    std::cout << BOLDMAGENTA << "-I-----------------------------------------------\n" << std::endl << RESET;
}


hailo_status run_image_encoder(std::vector<std::string> text_vec,
                                std::string image_encoder_hef, 
                                TSQueue<std::vector<std::vector<float>>>& text_embeddings_queue,
                                std::string input_image_path, std::string image_num, 
                                std::chrono::time_point<std::chrono::system_clock>& write_time_vec,
                                std::chrono::time_point<std::chrono::system_clock>& postprocess_end_time, 
                                std::chrono::duration<double>& inference_time) {

    auto vdevice_image = VDevice::create();
    if (!vdevice_image) {
        std::cerr << "Failed create vdevice, status = " << vdevice_image.status() << std::endl;
    }

    auto infer_model_image_encoder_exp = vdevice_image.value()->create_infer_model(image_encoder_hef);
    if (!infer_model_image_encoder_exp) {
        std::cerr << "Failed to create infer model, status = " << infer_model_image_encoder_exp.status() << std::endl;
    }

    std::shared_ptr<hailort::InferModel> infer_model_image_encoder = infer_model_image_encoder_exp.release();

    infer_model_image_encoder->set_batch_size(8);

    print_net_banner(image_encoder_hef, infer_model_image_encoder->inputs(), infer_model_image_encoder->outputs());

    cv::VideoCapture capture;
    double frame_count;
    if (input_image_path.empty()) {
        capture.open(0, cv::CAP_ANY);
        if (!capture.isOpened()) {
            throw "Error in camera input";
        }
        frame_count = -1;
    }
    else {
        capture.open(input_image_path, cv::CAP_ANY);
        if (!capture.isOpened()){
            throw "Error when reading video";
        }
        frame_count = capture.get(cv::CAP_PROP_FRAME_COUNT);
        if (!image_num.empty() && input_image_path.find(".avi") == std::string::npos && input_image_path.find(".mp4") == std::string::npos){
            frame_count = std::stoi(image_num);
        }
    }

    double org_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    double org_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);

    capture.release();


    auto status = configure_and_infer<float32_t>(std::ref(text_vec),
                                                    infer_model_image_encoder,
                                                    text_embeddings_queue, 
                                                    input_image_path, 
                                                    write_time_vec, 
                                                    postprocess_end_time, 
                                                    inference_time, 
                                                    (size_t)frame_count, 
                                                    org_height, 
                                                    org_width,
                                                    image_num);

    if (HAILO_SUCCESS != status) {
        std::cerr << "Failed to run configure_and_infer, status = " << status << std::endl;
        return status;
    }

    return HAILO_SUCCESS;
}

void get_postprocess_text_embeddings(std::vector<float>& text_embeddings, 
                                        std::vector<std::vector<float>>& text_projections, 
                                        std::vector<float>& postprocessed_text_embeddings) {
    
    for (size_t i = 0; i < text_projections.size(); i++) {
        for (size_t j = 0; j < text_embeddings.size(); j++) {
            postprocessed_text_embeddings[i] += text_embeddings[j] * text_projections[j][i];
        }
    }

    float norm = std::sqrt(std::inner_product(postprocessed_text_embeddings.begin(), postprocessed_text_embeddings.end(), postprocessed_text_embeddings.begin(), 0.0f));

    if (norm > 0) {
        for (size_t i = 0; i < postprocessed_text_embeddings.size(); i++) {
            postprocessed_text_embeddings[i] /= norm;
        }
    }
}


std::vector<float> readFloatsFromBinaryFile(const std::string& filename) {
    std::vector<float> numbers;
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return numbers;
    }

    float number;
    while (file.read(reinterpret_cast<char*>(&number), sizeof(float))) {
        numbers.push_back(number);
    }

    file.close();
    return numbers;
}


hailo_status run_text_encoder(std::string text_encoder_hef, std::vector<std::string>& input_text, TSQueue<std::vector<std::vector<float>>>& text_embeddings_queue) {

    auto vdevice_text = VDevice::create();
    if (!vdevice_text) {
        std::cerr << "Failed create vdevice, status = " << vdevice_text.status() << std::endl;
    }

    auto infer_model_text_encoder_exp = vdevice_text.value()->create_infer_model(text_encoder_hef);
    if (!infer_model_text_encoder_exp) {
        std::cerr << "Failed to create infer model, status = " << infer_model_text_encoder_exp.status() << std::endl;
    }

    std::shared_ptr<hailort::InferModel> infer_model_text_encoder = infer_model_text_encoder_exp.release();

    print_net_banner(text_encoder_hef, infer_model_text_encoder->inputs(), infer_model_text_encoder->outputs());

    auto [tokenized_text, last_tokens] = tokenizer::get_hailo_input(input_text);

    int num_of_tokens = 77;
    int token_length = 768;

    infer_model_text_encoder->input()->set_format_type(HAILO_FORMAT_TYPE_FLOAT32);
    size_t input_frame_size = infer_model_text_encoder->input()->get_frame_size();

    infer_model_text_encoder->output()->set_format_type(HAILO_FORMAT_TYPE_FLOAT32);
    size_t output_frame_size = infer_model_text_encoder->output()->get_frame_size();

    auto configured_infer_model = infer_model_text_encoder->configure();
    if (!configured_infer_model) {
        std::cerr << "Failed to create configured infer model, status = " << configured_infer_model.status() << std::endl;
        return configured_infer_model.status();
    }

    AsyncInferJob last_infer_job;

    std::shared_ptr<float32_t> output_buffer;
    std::shared_ptr<hailo_vstream_info_t> text_encoder_vstream_infos;

    std::vector<std::shared_ptr<float32_t>> output_buffer_guards;
    std::vector<std::shared_ptr<hailo_vstream_info_t>> vstream_infos_guards;

    auto bindings = configured_infer_model->create_bindings();
    if (!bindings) {
        std::cerr << "Failed to create infer bindings, status = " << bindings.status() << std::endl;
        return bindings.status();
    }

    std::vector<std::vector<float>> text_embeddings;

    for (size_t i = 0; i < tokenized_text.size(); i++) {

        auto status = bindings->input()->set_buffer(MemoryView(tokenized_text[i].data(), input_frame_size));
        if (HAILO_SUCCESS != status) {
            std::cerr << "Failed to set infer input buffer, status = " << status << std::endl;
            return status;
        }
        
        for (const auto &output_name : infer_model_text_encoder->get_output_names()) {
            output_buffer = page_aligned_alloc<float32_t>(output_frame_size);
            auto status = bindings->output(output_name)->set_buffer(MemoryView(output_buffer.get(), output_frame_size));
            if (HAILO_SUCCESS != status) {
                std::cerr << "Failed to set infer output buffer, status = " << status << std::endl;
                return status;
            }

            text_encoder_vstream_infos = std::make_shared<hailo_vstream_info_t>(infer_model_text_encoder->hef().get_output_vstream_infos().release()[0]);

            output_buffer_guards.push_back(output_buffer);
            vstream_infos_guards.push_back(text_encoder_vstream_infos);
        }

        status = configured_infer_model->wait_for_async_ready(std::chrono::milliseconds(1000));
        if (HAILO_SUCCESS != status) {
            std::cerr << "Failed to run wait_for_async_ready, status = " << status << std::endl;
            return status;
        }

        auto job = configured_infer_model->run_async(bindings.value());

        if (!job) {
            std::cerr << "Failed to start async infer job, status = " << job.status() << std::endl;
            return job.status();
        }

        job->detach();
        last_infer_job = job.release();

        status = last_infer_job.wait(std::chrono::milliseconds(1000));
        if (HAILO_SUCCESS != status) {
            std::cerr << "Failed to wait for infer to finish, status = " << status << std::endl;
            return status;
        }

        std::vector<float> dequantized_output_buffer(text_encoder_vstream_infos->shape.height * text_encoder_vstream_infos->shape.width * text_encoder_vstream_infos->shape.features);

        std::vector<std::vector<float>> dequantized_text_embeddings(num_of_tokens);
        for (size_t i = 0; i < dequantized_text_embeddings.size(); i++) {
            dequantized_text_embeddings[i] = std::vector<float>(token_length);
            std::copy(output_buffer.get() + i * token_length, output_buffer.get() + (i + 1) * token_length, dequantized_text_embeddings[i].begin());
        }

        auto raw_text_projections = readFloatsFromBinaryFile("text_projection.bin");
        std::vector<std::vector<float>> text_projections(token_length);
        for (size_t i = 0; i < text_projections.size(); i++) {
            text_projections[i] = std::vector<float>(token_length);
            std::copy(raw_text_projections.begin() + i * token_length, raw_text_projections.begin() + (i + 1) * token_length, text_projections[i].begin());
        }

        std::vector<float> postprocessed_text_embeddings(text_projections.size(), 0.0);

        get_postprocess_text_embeddings(dequantized_text_embeddings[last_tokens[i] - 1], text_projections, std::ref(postprocessed_text_embeddings));

        text_embeddings.push_back(postprocessed_text_embeddings);
    }

    text_embeddings_queue.push(text_embeddings);

    return HAILO_SUCCESS;
}


std::string get_hef_name(std::string path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return path;
    }
    return path.substr(pos + 1);
}


struct ZeroShotArgs {
    std::string text_encoder;   // -te=
    std::string image_encoder;  // -ie=
    std::string prompt;         // -p=
    std::string input_path;     // -i=
    std::string image_num;      // -n=
    bool        list_nets;      // --list-nets
    bool        list_inputs;    // --list-inputs
};

static inline ZeroShotArgs parse_zero_shot_args(int argc, char **argv)
{
    ZeroShotArgs args{
        hailo_utils::getCmdOption(argc, argv, "-te="),
        hailo_utils::getCmdOption(argc, argv, "-ie="),
        hailo_utils::getCmdOption(argc, argv, "-p="),
        hailo_utils::getCmdOption(argc, argv, "-i="),
        hailo_utils::getCmdOption(argc, argv, "-n="),
        hailo_utils::has_flag(argc, argv, "--list-nets"),
        hailo_utils::has_flag(argc, argv, "--list-inputs")
    };
    return args;
}

static inline std::pair<std::string, std::string>
resolve_zero_shot_nets(const std::string &app,
                       const ZeroShotArgs &args,
                       const std::string &dest_dir = "hefs")
{
    if (args.text_encoder.empty() && args.image_encoder.empty()) {
        std::cerr
            << "Error: Missing -te= and -ie=.\n"
            << app << " requires 2 networks: text encoder + image encoder.\n";
        hailo_utils::list_networks(app);
        std::exit(1);
    }
    if (args.text_encoder.empty() || args.image_encoder.empty()) {
        std::cerr
            << "Error: Both -te= and -ie= must be provided.\n";
        hailo_utils::list_networks(app);
        std::exit(1);
    }

    std::string text_hef  = hailo_utils::resolve_net_arg(app, args.text_encoder,  dest_dir);
    std::string image_hef = hailo_utils::resolve_net_arg(app, args.image_encoder, dest_dir);
    return {text_hef, image_hef};
}


int main(int argc, char** argv)
{
    using clock = std::chrono::high_resolution_clock;

    std::chrono::duration<double> total_time;
    const std::string APP_NAME = "zero_shot_classification";

    // Parse all CLI args once
    ZeroShotArgs args = parse_zero_shot_args(argc, argv);

    if (hailo_utils::has_flag(argc, argv, "--list-nets")) {
        hailo_utils::list_networks(APP_NAME);
        return 0;
    }
    if (hailo_utils::has_flag(argc, argv, "--list-inputs")) {
        hailo_utils::list_inputs(APP_NAME);
        return 0;
    }

    // Resolve to actual HEF paths (model name or .hef → real path + arch check)
    auto [text_encoder_hef, image_encoder_hef] =
        resolve_zero_shot_nets(APP_NAME, args, "hefs");

    // Build text_vec from prompt (possibly comma-separated)
    std::vector<std::string> text_vec;
    std::string prompt = args.prompt;  // local copy, we modify it

    if (prompt.find(',') == std::string::npos) {
        if (!prompt.empty()) {
            text_vec.push_back(prompt);
        }
    } else {
        const std::string delimiter = ",";
        size_t pos = 0;
        std::string token;
        while ((pos = prompt.find(delimiter)) != std::string::npos) {
            token = prompt.substr(0, pos);
            text_vec.push_back(token);
            prompt.erase(0, pos + delimiter.length());
        }
        text_vec.push_back(prompt);
    }

    std::chrono::time_point<std::chrono::system_clock> write_time_vec;
    std::chrono::time_point<std::chrono::system_clock> postprocess_end_time;
    std::chrono::duration<double> inference_time;

    TSQueue<std::vector<std::vector<float>>> text_embeddings_queue;

    auto t_start = clock::now();

    auto status = run_text_encoder(text_encoder_hef,
                                   std::ref(text_vec),
                                   std::ref(text_embeddings_queue));
    if (HAILO_SUCCESS != status) {
        std::cerr << "Failed to run text encoder, status = " << status << std::endl;
        return status;
    }

    status = run_image_encoder(text_vec,
                               image_encoder_hef,
                               std::ref(text_embeddings_queue),
                               std::ref(args.input_path),
                               args.image_num,
                               std::ref(write_time_vec),
                               std::ref(postprocess_end_time),
                               std::ref(inference_time));
    if (HAILO_SUCCESS != status) {
        std::cerr << "Failed to run image encoder, status = " << status << std::endl;
        return status;
    }

    auto t_end = clock::now();
    total_time = t_end - t_start;

    std::cout << BOLDBLUE << "\n-I- Application run finished successfully" << RESET << std::endl;
    std::cout << BOLDBLUE << "-I- Total application run time: "
              << total_time.count() << " sec" << RESET << std::endl;

    return HAILO_SUCCESS;
}
