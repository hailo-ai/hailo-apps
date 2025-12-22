#include "hailo_infer.hpp"
#if defined(__unix__)
#include <sys/mman.h>
#endif
#include <cstring>

static std::shared_ptr<uint8_t> page_aligned_alloc(size_t size, void* buff = nullptr) {
    #if defined(__unix__)
        auto addr = mmap(buff, size, PROT_WRITE | PROT_READ, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
        if (MAP_FAILED == addr) throw std::bad_alloc();
        return std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t*>(addr), [size](void *addr) { munmap(addr, size); });
    #elif defined(_MSC_VER)
        auto addr = VirtualAlloc(buff, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (!addr) throw std::bad_alloc();
        return std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t*>(addr), [](void *addr){ VirtualFree(addr, 0, MEM_RELEASE); });
    #else
    #pragma error("Aligned alloc not supported")
    #endif
}


HailoInfer::HailoInfer(const std::string &hef_path,
                       size_t batch_size,
                       hailo_format_type_t input_type,
                       hailo_format_type_t output_type)
{
    this->vdevice = hailort::VDevice::create().expect("Failed to create VDevice");
    this->batch_size = batch_size;
    this->infer_model = vdevice->create_infer_model(hef_path).expect("Failed to create infer model");
    this->infer_model->set_batch_size(batch_size);

    if (output_type != HAILO_FORMAT_TYPE_AUTO)
        for (auto &&name : infer_model->get_output_names())
            infer_model->output(name)->set_format_type(output_type);

    if (input_type != HAILO_FORMAT_TYPE_AUTO)
        for (auto &&name : infer_model->get_input_names())
            infer_model->input(name)->set_format_type(input_type);

    for (auto& output_vstream_info : this->infer_model->hef().get_output_vstream_infos().release()) {
        std::string name(output_vstream_info.name);
        this->output_vstream_info_by_name[name] = output_vstream_info;
    }
    this->configured_infer_model = this->infer_model->configure().expect("Failed to create configured infer model");
    this->multiple_bindings = std::vector<ConfiguredInferModel::Bindings>();
}

HailoInfer::HailoInfer(const std::string &hef_path,
                       const std::string &group_id,
                       size_t batch_size,
                       hailo_format_type_t input_type,
                       hailo_format_type_t output_type)
{
    hailo_vdevice_params_t vdevice_params = {0};
    hailo_init_vdevice_params(&vdevice_params);
    vdevice_params.group_id = group_id.c_str();
    vdevice_params.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN;
    
    this->vdevice = hailort::VDevice::create(vdevice_params).expect("Failed to create VDevice");
    this->batch_size = batch_size;
    this->infer_model = vdevice->create_infer_model(hef_path).expect("Failed to create infer model");
    this->infer_model->set_batch_size(batch_size);

    if (output_type != HAILO_FORMAT_TYPE_AUTO)
        for (auto &&name : infer_model->get_output_names())
            infer_model->output(name)->set_format_type(output_type);

    if (input_type != HAILO_FORMAT_TYPE_AUTO)
        for (auto &&name : infer_model->get_input_names())
            infer_model->input(name)->set_format_type(input_type);

    for (auto& output_vstream_info : this->infer_model->hef().get_output_vstream_infos().release()) {
        std::string name(output_vstream_info.name);
        this->output_vstream_info_by_name[name] = output_vstream_info;
    }
    this->configured_infer_model = this->infer_model->configure().expect("Failed to create configured infer model");
    this->multiple_bindings = std::vector<ConfiguredInferModel::Bindings>();
}

const std::vector<hailort::InferModel::InferStream>& HailoInfer::get_inputs(){
    return std::move(this->infer_model->inputs());
}

const std::vector<hailort::InferModel::InferStream>& HailoInfer::get_outputs(){
    return std::move(this->infer_model->outputs());
}

const std::shared_ptr<hailort::InferModel> HailoInfer::get_infer_model(){
    return this->infer_model;
}

hailo_3d_image_shape_t HailoInfer::get_model_shape(){
    auto input_infos = this->infer_model->hef().get_input_vstream_infos().expect("Failed to get input vstream infos");
    return input_infos[0].shape;
} 

size_t HailoInfer::get_output_vstream_infos_size(){
    return this->infer_model->hef().get_output_vstream_infos().release().size();
}

void HailoInfer::infer(
    const InputMap &inputs,
    std::function<void(const hailort::AsyncInferCompletionInfo&,
                       const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &,
                       const std::vector<std::shared_ptr<uint8_t>> &)> callback) {

    std::vector<std::shared_ptr<cv::Mat>> input_image_guards;
    set_input_buffers(inputs, input_image_guards);
    std::vector<std::shared_ptr<uint8_t>> output_guards;
    auto output_data_and_infos = prepare_output_buffers(output_guards);
    run_async(output_data_and_infos, output_guards, input_image_guards, callback);
}

void HailoInfer::set_input_buffers(
    const InputMap &inputs,
    std::vector<std::shared_ptr<cv::Mat>> &image_guards)
{
    this->multiple_bindings.clear();
    const auto &model_inputs = infer_model->get_input_names();

    for (size_t i = 0; i < this->batch_size; ++i) {
        auto bindings = this->configured_infer_model.create_bindings().expect("Failed");
        for (const auto &input_name : model_inputs) {
            const cv::Mat &input = inputs.at(input_name)[i];
            size_t frame_size = infer_model->input(input_name)->get_frame_size();
            auto status = bindings.input(input_name)->set_buffer(MemoryView(input.data, frame_size));
            if (HAILO_SUCCESS != status) {
                std::cerr << "Failed to set input buffer for '" << input_name
                          << "', status = " << status << std::endl;
            }
            // keep input data alive until the async job completes
            image_guards.push_back(std::make_shared<cv::Mat>(input));

        }
        this->multiple_bindings.push_back(std::move(bindings));
    }
}

std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> HailoInfer::prepare_output_buffers(
    std::vector<std::shared_ptr<uint8_t>> &output_guards) {

    std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> result;
    for (auto& binding : this->multiple_bindings) {
        for (const auto &output_name : this->infer_model->get_output_names()) {
            size_t frame_size = this->infer_model->output(output_name)->get_frame_size();
            auto buffer = page_aligned_alloc(frame_size);
            output_guards.push_back(buffer);
            auto status = binding.output(output_name)->set_buffer(MemoryView(buffer.get(), frame_size));
            if (HAILO_SUCCESS != status) {
                std::cerr << "Failed to set output buffer, status = " << status << std::endl;
            }
            result.emplace_back(buffer.get(), output_vstream_info_by_name[output_name]);
        }
    }
    return result;
}

void HailoInfer::run_async(
    const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &output_data_and_infos,
    const std::vector<std::shared_ptr<uint8_t>> &output_guards,
    const std::vector<std::shared_ptr<cv::Mat>> &input_image_guards,
    std::function<void(const hailort::AsyncInferCompletionInfo&,
                       const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &,
                       const std::vector<std::shared_ptr<uint8_t>> &)> callback)
{
    auto status = configured_infer_model.wait_for_async_ready(std::chrono::milliseconds(50000), this->batch_size);
    if (HAILO_SUCCESS != status) {
        std::cerr << "Failed wait_for_async_ready, status = " << status << std::endl;
    }
    auto job = configured_infer_model.run_async(
        this->multiple_bindings,
        [callback, output_data_and_infos, input_image_guards, output_guards](const hailort::AsyncInferCompletionInfo& info)
        {
            // callback sent by the applicative side
            callback(info, output_data_and_infos, output_guards);
        }
    );
    if (!job) {
        std::cerr << "Failed to start async infer job, status = " << job.status() << std::endl;
    }
    job->detach();
    last_infer_job = std::move(job.release());
}

void HailoInfer::wait_for_last_job()
{
    auto st = last_infer_job.wait(std::chrono::milliseconds(50000));
    if (HAILO_SUCCESS != st && HAILO_TIMEOUT != st) {
        std::cerr << "Failed waiting for last infer job, status = " << st << std::endl;
    }
}
