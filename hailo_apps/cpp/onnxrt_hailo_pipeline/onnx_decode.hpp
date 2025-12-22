#pragma once
/**
 * Wrapper for instance segmentation decode-only ONNX using raw Hailo vstream outputs.
 * Supports any number of inputs and outputs with flexible dimensions.
 */

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <array>
#include <cstdint>
#include <hailo/hailort.h>
#include "../common/general/hailo_objects.hpp"
#include <xtensor/xarray.hpp>

struct HEFOutputs {
    std::vector<float> output;               // Detection outputs
    std::vector<int64_t> output_shape;
    std::vector<float> proto;
    std::vector<int64_t> proto_shape;     
};

class InstanceSegDecodeONNX {
public:
    explicit InstanceSegDecodeONNX(const std::string &model_path);

    // Directly consume the raw HEF outputs for a single frame:
    // vector of (buffer pointer, vstream info)
    HEFOutputs run(const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &hef_outputs);

private:
    Ort::Env _env;
    Ort::Session _sess;
    Ort::MemoryInfo _meminfo;

    std::vector<std::string> _in_names;
    std::vector<std::string> _out_names;

    // Convert HWC uint8 vstream to NCHW float.
    std::vector<float> to_nchw_float(const std::pair<uint8_t*, hailo_vstream_info_t> &src,
                                     std::array<int64_t,4> &nchw) const;
};

// ------------ Parsing ONNX decoded detection output ------------
/**
 * Parse ONNX decode-only detections into (HailoDetection, mask_coeffs) pairs.
 */
std::vector<std::pair<HailoDetection, xt::xarray<float>>>
parse_onnx_output(const std::vector<float> &output,
                  int model_w, int model_h,
                  float score_thr,
                  int org_w, int org_h,
                  int channels_per_detection,
                  int num_detections,
                  int num_classes,
                  int mask_coeffs);
