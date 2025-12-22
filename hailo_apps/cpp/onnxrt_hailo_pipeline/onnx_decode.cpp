#include "onnx_decode.hpp"
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <cmath>
#include "../common/labels/coco_eighty.hpp"

namespace {
struct LetterboxParams { int model_w, model_h; int resized_w, resized_h; int pad_right, pad_bottom; float scale; };

static inline LetterboxParams compute_letterbox(int org_w, int org_h, int model_w, int model_h)
{
    const float fh = static_cast<float>(org_h) / static_cast<float>(model_h);
    const float fw = static_cast<float>(org_w) / static_cast<float>(model_w);
    const float factor = std::max(fh, fw);
    const int resized_w = static_cast<int>(std::round(static_cast<float>(org_w) / factor));
    const int resized_h = static_cast<int>(std::round(static_cast<float>(org_h) / factor));
    LetterboxParams p; p.model_w=model_w; p.model_h=model_h; p.resized_w=std::min(model_w, resized_w); p.resized_h=std::min(model_h, resized_h); p.pad_right=model_w - p.resized_w; p.pad_bottom=model_h - p.resized_h; p.scale=factor; return p;
}

static inline HailoBBox unletterbox_xywh_to_orig_norm(float cx_m, float cy_m, float w_m, float h_m,
                                                      const LetterboxParams &lb, int org_w, int org_h)
{
    float x1_m = cx_m - 0.5f * w_m;
    float y1_m = cy_m - 0.5f * h_m;
    float x2_m = cx_m + 0.5f * w_m;
    float y2_m = cy_m + 0.5f * h_m;
    x1_m = std::max(0.0f, std::min<float>(x1_m, static_cast<float>(lb.resized_w)));
    y1_m = std::max(0.0f, std::min<float>(y1_m, static_cast<float>(lb.resized_h)));
    x2_m = std::max(0.0f, std::min<float>(x2_m, static_cast<float>(lb.resized_w)));
    y2_m = std::max(0.0f, std::min<float>(y2_m, static_cast<float>(lb.resized_h)));
    float x1_o = x1_m * lb.scale; float y1_o = y1_m * lb.scale; float x2_o = x2_m * lb.scale; float y2_o = y2_m * lb.scale;
    float nx1 = std::clamp(x1_o / static_cast<float>(org_w), 0.f, 1.f);
    float ny1 = std::clamp(y1_o / static_cast<float>(org_h), 0.f, 1.f);
    float nx2 = std::clamp(x2_o / static_cast<float>(org_w), 0.f, 1.f);
    float ny2 = std::clamp(y2_o / static_cast<float>(org_h), 0.f, 1.f);
    return HailoBBox(nx1, ny1, nx2 - nx1, ny2 - ny1);
}
} // namespace


InstanceSegDecodeONNX::InstanceSegDecodeONNX(const std::string &model_path)
: _env(ORT_LOGGING_LEVEL_WARNING, "inst_seg"),
  _sess(nullptr),
  _meminfo(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)) {

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    _sess = Ort::Session(_env, model_path.c_str(), session_options);

    // cache I/O names
    Ort::AllocatorWithDefaultOptions alloc;
    size_t n_in  = _sess.GetInputCount();
    size_t n_out = _sess.GetOutputCount();
    _in_names.reserve(n_in);
    _out_names.reserve(n_out);
    for (size_t i=0;i<n_in;i++)  { auto input_name = _sess.GetInputNameAllocated(i, alloc);  _in_names.emplace_back(input_name.get()); }
    for (size_t i=0;i<n_out;i++) { auto output_name = _sess.GetOutputNameAllocated(i, alloc); _out_names.emplace_back(output_name.get()); }

    if (n_in != 10 || n_out != 2) {
        throw std::runtime_error("Cropped ONNX must have 10 inputs (9 heads + proto_in) and 2 outputs (output0, output1).");
    }
}

// --------------------------- HWC (uint8) -> NCHW (float) ----------------------
static inline float deq(uint8_t quantized_value, float scale, float zero_point) {
    return (static_cast<float>(quantized_value) - zero_point) * scale;
}

std::vector<float>
InstanceSegDecodeONNX::to_nchw_float(const std::pair<uint8_t*, hailo_vstream_info_t> &src,
                                      std::array<int64_t,4> &nchw) const
{
    const auto &inf = src.second;
    const int H = (int)inf.shape.height;
    const int W = (int)inf.shape.width;
    const int C = (int)inf.shape.features;

    nchw = {1, C, H, W};

    const auto *qptr = src.first; // HWC uint8
    const float scale = inf.quant_info.qp_scale;
    const float zero_point = inf.quant_info.qp_zp;

    std::vector<float> out(static_cast<size_t>(H)*W*C);
    size_t idx = 0;

    // Reorder HWC -> NCHW while dequantizing
    for (int channel=0;channel<C;++channel) {
        for (int height=0;height<H;++height) {
            for (int width=0;width<W;++width) {
                const size_t hwc_index = (size_t)height*W*C + (size_t)width*C + (size_t)channel;
                out[idx++] = deq(qptr[hwc_index], scale, zero_point);
            }
        }
    }
    return out;
}

HEFOutputs InstanceSegDecodeONNX::run(const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &hef_outputs) {

    std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> pool = hef_outputs;
    // Build ORT inputs
    const size_t n_in = _sess.GetInputCount();
    std::vector<std::array<int64_t,4>> input_shapes(n_in);
    std::vector<std::vector<float>>     buffers(n_in);
    std::vector<Ort::Value>             ort_inputs; ort_inputs.reserve(n_in);
    std::vector<const char*>            in_names;   in_names.reserve(n_in);

    for (size_t i = 0; i < n_in; ++i) {
        in_names.push_back(_in_names[i].c_str());

        // --- get ONNX input's expected NCHW shape ---
        std::vector<int64_t> shape;
        bool got_shape = false;
        try {
            auto type_info = _sess.GetInputTypeInfo(i); 
            auto tensor_shape = type_info.GetTensorTypeAndShapeInfo();
            shape = tensor_shape.GetShape();
            got_shape = (shape.size() == 4);
        } catch (...) {
            got_shape = false;
        }

        int64_t N = 1, C = -1, H = -1, W = -1;
        if (got_shape) {
            N = (shape[0] < 0 ? 1 : shape[0]);
            C = shape[1];
            H = shape[2];
            W = shape[3];
        } else {
        }

        // --- Find a matching HEF vstream by HWC = (H, W, C) ---
        int pick = -1;
        for (int pool_index = 0; pool_index < (int)pool.size(); ++pool_index) {
            const auto &inf = pool[pool_index].second;
            const int64_t height = (int64_t)inf.shape.height;
            const int64_t width = (int64_t)inf.shape.width;
            const int64_t channels = (int64_t)inf.shape.features;

            if (got_shape) {
                if (height == H && width == W && channels == C) { pick = pool_index; break; }
            } else {
                // No ONNX shape -> match any available HEF tensor
                // This is a fallback when ONNX doesn't provide shape info
                pick = pool_index; break;
            }
        }
        if (pick < 0) {
            throw std::runtime_error("Could not match ONNX input #" + std::to_string(i) +
                                    " (" + _in_names[i] + ") to any HEF vstream by HWC.");
        }

        const auto &inf = pool[pick].second;

        // If ONNX shape was not available, derive from HEF HWC
        if (!got_shape) {
            H = (int64_t)inf.shape.height;
            W = (int64_t)inf.shape.width;
            C = (int64_t)inf.shape.features;
            N = 1;
        }

        input_shapes[i] = {N, C, H, W};

        // Convert that HEF HWC uint8 tensor to NCHW float
        buffers[i] = to_nchw_float(pool[pick], input_shapes[i]);
        pool.erase(pool.begin() + pick);

        ort_inputs.emplace_back(
            Ort::Value::CreateTensor<float>(_meminfo,
                    buffers[i].data(), buffers[i].size(),
                    input_shapes[i].data(), input_shapes[i].size()));
    }

    // prepare output names in session order
    std::vector<const char*> out_names;
    out_names.reserve(_out_names.size());
    for (auto &output_name : _out_names) out_names.push_back(output_name.c_str());

    auto outs = _sess.Run(Ort::RunOptions{nullptr},
                        in_names.data(), ort_inputs.data(), ort_inputs.size(),
                        out_names.data(), out_names.size());

    HEFOutputs ret;

    // ---- output0: detection outputs ----
    {
        auto &output0 = outs[0];
        auto info = output0.GetTensorTypeAndShapeInfo();
        auto shapes = info.GetShape();
        // compute element count, treat negatives as 1
        size_t size = 1;
        for (auto dimension : shapes) { size *= (size_t)((dimension < 0) ? 1 : dimension); }
        if (size > 20'000'000) {
            throw std::runtime_error("output0 element count suspiciously large: " + std::to_string(size));
        }

        ret.output.resize(size);
        const float *src = output0.GetTensorData<float>();
        std::memcpy(ret.output.data(), src, size*sizeof(float));
        
        ret.output_shape.resize(shapes.size());
        for (size_t i = 0; i < shapes.size(); ++i) {
            ret.output_shape[i] = (shapes[i] < 0 ? 1 : shapes[i]);
        }
    }

    // ---- output1: proto outputs ----
    {
        auto &output1 = outs[1];
        auto info = output1.GetTensorTypeAndShapeInfo();
        auto shapes = info.GetShape();

        size_t size = 1;
        for (auto dimension : shapes) { size *= (size_t)((dimension < 0) ? 1 : dimension); }
        if (size > 50'000'000) {
            throw std::runtime_error("output1 element count suspiciously large: " + std::to_string(size));
        }

        ret.proto.resize(size);
        const float *src = output1.GetTensorData<float>();
        std::memcpy(ret.proto.data(), src, size*sizeof(float));
        
        ret.proto_shape.resize(shapes.size());
        for (size_t i = 0; i < shapes.size(); ++i) {
            ret.proto_shape[i] = (shapes[i] < 0 ? 1 : shapes[i]);
        }
    }

    return ret;
}

// ---------------- Parsing ONNX decoded detection output -----------------
std::vector<std::pair<HailoDetection, xt::xarray<float>>>
parse_onnx_output(const std::vector<float> &output,
                  int model_w, int model_h,
                  float score_thr,
                  int org_w, int org_h,
                  int channels_per_detection,
                  int num_detections,
                  int num_classes,
                  int mask_coeffs)
{
    const int expected_size = channels_per_detection * num_detections;
    if (static_cast<int>(output.size()) != expected_size) {
        throw std::runtime_error("Output size mismatch: expected " + std::to_string(expected_size) + 
                                ", got " + std::to_string(output.size()));
    }

    auto at = [&](int channel, int detection_idx)->float { return output[channel * num_detections + detection_idx]; };

    std::vector<std::pair<HailoDetection, xt::xarray<float>>> res;
    res.reserve(256);

    const LetterboxParams lb = compute_letterbox(org_w, org_h, model_w, model_h);

    const int xywh_start = 0;
    const int classes_start = 4;
    const int mask_start = 4 + num_classes;

    for (int detection_idx = 0; detection_idx < num_detections; ++detection_idx) {
        int best_class = -1; float best_score = 0.f;
        for (int class_idx = 0; class_idx < num_classes; ++class_idx) {
            float score = at(classes_start + class_idx, detection_idx);
            if (score > best_score) { best_score = score; best_class = class_idx; }
        }
        if (best_score < score_thr) continue;

        const float cx_m = at(xywh_start + 0, detection_idx);
        const float cy_m = at(xywh_start + 1, detection_idx);
        const float  w_m = at(xywh_start + 2, detection_idx);
        const float  h_m = at(xywh_start + 3, detection_idx);
        const HailoBBox bb = unletterbox_xywh_to_orig_norm(cx_m, cy_m, w_m, h_m, lb, org_w, org_h);

        std::string label = common::coco_eighty[best_class + 1];
        HailoDetection det(bb, best_class, label, best_score);

        xt::xarray<float> coeff = xt::zeros<float>({mask_coeffs});
        for (int i = 0; i < mask_coeffs; ++i) { coeff(i) = at(mask_start + i, detection_idx); }
        res.emplace_back(det, std::move(coeff));
    }
    return res;
}
