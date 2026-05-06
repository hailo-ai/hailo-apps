#include "yolov8seg.hpp"

#include "common/labels/coco_eighty.hpp"
#include "common/math.hpp"
#include "common/nms.hpp"
#include "common/tensors.hpp"
#include "hailo_common.hpp"
#include "json_config.hpp"
#include "mask_decoding.hpp"

#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/schema.h"

#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xmanipulation.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xview.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <map>
#include <cstdio>
#include <stdexcept>
#include <tuple>
#include <vector>

#if __GNUC__ > 8
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

using namespace xt::placeholders;

namespace
{
constexpr int REGRESSION_LENGTH = 15;
constexpr int BOX_CHANNELS = 4 * (REGRESSION_LENGTH + 1);
constexpr int MASK_COEFFICIENTS = 32;

struct ScaleTensors
{
    HailoTensorPtr boxes;
    HailoTensorPtr scores;
    HailoTensorPtr masks;
    int stride;
};

float dequantize_value(uint8_t value, float qp_scale, float qp_zp)
{
    return (float(value) - qp_zp) * qp_scale;
}

std::vector<xt::xarray<double>> get_centers(const std::vector<int> &strides, const std::vector<int> &network_dims)
{
    std::vector<xt::xarray<double>> centers(strides.size());

    for (uint i = 0; i < strides.size(); i++)
    {
        int strided_width = network_dims[0] / strides[i];
        int strided_height = network_dims[1] / strides[i];

        xt::xarray<int> grid_x = xt::arange(0, strided_width);
        xt::xarray<int> grid_y = xt::arange(0, strided_height);
        auto mesh = xt::meshgrid(grid_x, grid_y);
        grid_x = std::get<1>(mesh);
        grid_y = std::get<0>(mesh);

        // Use standard meshgrid indexing to build centers
        auto ct_row = (xt::flatten(grid_y) + 0.5) * strides[i];
        auto ct_col = (xt::flatten(grid_x) + 0.5) * strides[i];
        centers[i] = xt::stack(xt::xtuple(ct_col, ct_row, ct_col, ct_row), 1);
    }

    return centers;
}

int stride_from_tensor(const HailoTensorPtr &tensor, int input_width)
{
    return input_width / static_cast<int>(tensor->height());
}

/**
 * @brief Collect scale tensors using the channel-count heuristic.
 *
 * Each output tensor is classified by its channel count:
 *   - BOX_CHANNELS (64)        → box regression tensor
 *   - num_classes               → class-score tensor  (was hardcoded to 80)
 *   - MASK_COEFFICIENTS (32)   → mask-coefficient tensor (proto excluded by stride==4)
 *
 * @param tensors     All tensors from the ROI
 * @param input_width Network input width (used to compute stride)
 * @param num_classes Number of detection classes in the model
 */
std::vector<ScaleTensors> collect_scale_tensors(
    const std::vector<HailoTensorPtr> &tensors,
    int input_width,
    int num_classes)
{
    std::map<int, ScaleTensors> grouped;

    for (const auto &tensor : tensors)
    {
        if (tensor->shape().size() != 3)
        {
            continue;
        }

        int channels = static_cast<int>(tensor->shape()[2]);
        int stride = stride_from_tensor(tensor, input_width);
        auto &group = grouped[stride];
        group.stride = stride;

        if (channels == BOX_CHANNELS)
        {
            group.boxes = tensor;
        }
        else if (channels == num_classes)
        {
            group.scores = tensor;
        }
        else if (channels == MASK_COEFFICIENTS && stride != 4)
        {
            group.masks = tensor;
        }
    }

    std::vector<ScaleTensors> scales;
    for (auto &[_, group] : grouped)
    {
        if (group.boxes && group.scores && group.masks)
        {
            scales.push_back(group);
        }
    }

    std::sort(scales.begin(), scales.end(), [](const ScaleTensors &a, const ScaleTensors &b) {
        return a.stride < b.stride;
    });
    return scales;
}

/**
 * @brief Collect scale tensors using explicit tensor names from the JSON config.
 *
 * This is fully robust: it does not rely on channel counts at all.
 * The three name lists (boxes, scores, masks) must be the same length.
 * Strides are derived at runtime from the boxes tensor shape.
 *
 * @param tensors_by_name  Map of tensor name → tensor pointer (from roi->get_tensors_by_name())
 * @param names            Explicit output name struct from params
 * @param input_width      Network input width (used to compute stride from tensor shape)
 */
std::vector<ScaleTensors> collect_scale_tensors_by_name(
    const std::map<std::string, HailoTensorPtr> &tensors_by_name,
    const Yolov8segOutputsName &names,
    int input_width,
    int num_classes)
{
    std::vector<ScaleTensors> scales;

    for (size_t i = 0; i < names.boxes.size(); i++)
    {
        auto box_it   = tensors_by_name.find(names.boxes[i]);
        auto score_it = tensors_by_name.find(names.scores[i]);
        auto mask_it  = tensors_by_name.find(names.masks[i]);

        if (box_it == tensors_by_name.end() ||
            score_it == tensors_by_name.end() ||
            mask_it == tensors_by_name.end())
        {
            std::cerr << "Warning: one or more tensors not found for scale " << i
                      << " (boxes='" << names.boxes[i]
                      << "', scores='" << names.scores[i]
                      << "', masks='" << names.masks[i] << "')" << std::endl;
            continue;
        }

        ScaleTensors s;
        s.boxes  = box_it->second;
        s.scores = score_it->second;
        s.masks  = mask_it->second;

        if (s.boxes->shape().size() != 3 ||
            s.scores->shape().size() != 3 ||
            s.masks->shape().size() != 3)
        {
            std::cerr << "Warning: invalid tensor rank for scale " << i
                      << " (expected rank-3 for boxes/scores/masks)" << std::endl;
            continue;
        }

        const int box_channels = static_cast<int>(s.boxes->shape()[2]);
        const int score_channels = static_cast<int>(s.scores->shape()[2]);
        const int mask_channels = static_cast<int>(s.masks->shape()[2]);
        if (box_channels != BOX_CHANNELS || mask_channels != MASK_COEFFICIENTS || score_channels != num_classes)
        {
            std::cerr << "Warning: unexpected channels for scale " << i
                      << " (boxes=" << box_channels
                      << ", scores=" << score_channels
                      << ", masks=" << mask_channels
                      << ", expected boxes=" << BOX_CHANNELS
                      << ", scores=" << num_classes
                      << ", masks=" << MASK_COEFFICIENTS << ")" << std::endl;
            continue;
        }

        s.stride = stride_from_tensor(s.boxes, input_width);
        scales.push_back(s);
    }

    std::sort(scales.begin(), scales.end(), [](const ScaleTensors &a, const ScaleTensors &b) {
        return a.stride < b.stride;
    });
    return scales;
}

HailoTensorPtr find_proto_tensor(const std::vector<HailoTensorPtr> &tensors, int input_width)
{
    for (const auto &tensor : tensors)
    {
        if (tensor->shape().size() != 3)
        {
            continue;
        }

        int channels = static_cast<int>(tensor->shape()[2]);
        int stride = stride_from_tensor(tensor, input_width);
        if (channels == MASK_COEFFICIENTS && stride == 4)
        {
            return tensor;
        }
    }

    return nullptr;
}

/**
 * @brief Find the prototype tensor by its explicit name from the JSON config.
 */
HailoTensorPtr find_proto_tensor_by_name(
    const std::map<std::string, HailoTensorPtr> &tensors_by_name,
    const std::string &proto_name)
{
    auto it = tensors_by_name.find(proto_name);
    if (it == tensors_by_name.end())
    {
        std::cerr << "Warning: proto tensor '" << proto_name << "' not found" << std::endl;
        return nullptr;
    }
    return it->second;
}

void fill_dequantized_box(
    std::array<float, BOX_CHANNELS> &decoded_box,
    const xt::xarray<uint8_t> &quantized_boxes,
    int proposal_index,
    float qp_scale,
    float qp_zp)
{
    for (size_t i = 0; i < 4; i++)
    {
        for (size_t j = 0; j < REGRESSION_LENGTH + 1; j++)
        {
            decoded_box[i * (REGRESSION_LENGTH + 1) + j] =
                dequantize_value(quantized_boxes(proposal_index, i, j), qp_scale, qp_zp);
        }
    }
}

std::vector<HailoDetection> decode_scale(
    const ScaleTensors &scale,
    const xt::xarray<double> &centers,
    const std::vector<int> &network_dims,
    float score_threshold)
{
    auto boxes_ptr = scale.boxes;
    auto scores_ptr = scale.scores;
    auto masks_ptr = scale.masks;

    auto box_tensor = common::get_xtensor(boxes_ptr);
    auto score_tensor = common::dequantize(
        common::get_xtensor(scores_ptr),
        scores_ptr->quant_info().qp_scale,
        scores_ptr->quant_info().qp_zp);
    auto mask_tensor = common::dequantize(
        common::get_xtensor(masks_ptr),
        masks_ptr->quant_info().qp_scale,
        masks_ptr->quant_info().qp_zp);

    if (box_tensor.shape().size() != 3 || score_tensor.shape().size() != 3 || mask_tensor.shape().size() != 3)
    {
        std::cerr << "Warning: invalid tensor rank in decode_scale (expected rank-3)" << std::endl;
        return {};
    }

    int num_proposals_boxes = static_cast<int>(box_tensor.shape(0) * box_tensor.shape(1));
    int num_proposals_scores = static_cast<int>(score_tensor.shape(0) * score_tensor.shape(1));
    int num_proposals_masks = static_cast<int>(mask_tensor.shape(0) * mask_tensor.shape(1));
    if (num_proposals_boxes <= 0 || num_proposals_scores <= 0 || num_proposals_masks <= 0 ||
        num_proposals_boxes != num_proposals_scores ||
        num_proposals_boxes != num_proposals_masks)
    {
        std::cerr << "Warning: proposal-count mismatch in decode_scale "
                  << "(boxes=" << num_proposals_boxes
                  << ", scores=" << num_proposals_scores
                  << ", masks=" << num_proposals_masks << ")" << std::endl;
        return {};
    }

    if (score_tensor.shape(2) == 0)
    {
        std::cerr << "Warning: score tensor has zero classes in decode_scale" << std::endl;
        return {};
    }

    int num_proposals = box_tensor.shape(0) * box_tensor.shape(1);
    int num_classes = static_cast<int>(scores_ptr->shape()[2]);
    auto quantized_boxes = xt::reshape_view(box_tensor, {num_proposals, 4, REGRESSION_LENGTH + 1});
    auto masks = xt::reshape_view(mask_tensor, {num_proposals, MASK_COEFFICIENTS});
    auto regression_distance = xt::reshape_view(xt::arange(0, REGRESSION_LENGTH + 1), {1, 1, REGRESSION_LENGTH + 1});

    std::vector<HailoDetection> detections;
    detections.reserve(num_proposals / 4);

    for (int proposal_index = 0; proposal_index < num_proposals; proposal_index++)
    {
        auto score_offset = proposal_index * num_classes;
        int class_index = 0;
        float confidence = score_tensor.data()[score_offset];
        for (int c = 1; c < num_classes; c++)
        {
            float score = score_tensor.data()[score_offset + c];
            if (score > confidence)
            {
                confidence = score;
                class_index = c;
            }
        }
        if (confidence < score_threshold)
        {
            continue;
        }

        std::array<float, BOX_CHANNELS> box{};
        fill_dequantized_box(
            box,
            quantized_boxes,
            proposal_index,
            boxes_ptr->quant_info().qp_scale,
            boxes_ptr->quant_info().qp_zp);
        common::softmax_2D(box.data(), 4, REGRESSION_LENGTH + 1);

        auto box_view = xt::adapt(box, {4UL, static_cast<size_t>(REGRESSION_LENGTH + 1)});
        auto box_distance = box_view * regression_distance;
        xt::xarray<float> reduced_distances = xt::sum(box_distance, {2});
        auto strided_distances = reduced_distances * scale.stride;

        auto distance_view1 = xt::view(strided_distances, xt::all(), xt::range(_, 2)) * -1;
        auto distance_view2 = xt::view(strided_distances, xt::all(), xt::range(2, _));
        auto distance_view = xt::concatenate(xt::xtuple(distance_view1, distance_view2), 1);
        auto decoded_box = centers + distance_view;

        float xmin = decoded_box(proposal_index, 0) / network_dims[0];
        float ymin = decoded_box(proposal_index, 1) / network_dims[1];
        float xmax = decoded_box(proposal_index, 2) / network_dims[0];
        float ymax = decoded_box(proposal_index, 3) / network_dims[1];

        if (!std::isfinite(xmin) || !std::isfinite(ymin) || !std::isfinite(xmax) || !std::isfinite(ymax))
        {
            continue;
        }

        xmin = std::clamp(xmin, 0.0f, 1.0f);
        ymin = std::clamp(ymin, 0.0f, 1.0f);
        xmax = std::clamp(xmax, 0.0f, 1.0f);
        ymax = std::clamp(ymax, 0.0f, 1.0f);

        float width = xmax - xmin;
        float height = ymax - ymin;
        const float min_norm_w = 2.0f / static_cast<float>(network_dims[0]);
        const float min_norm_h = 2.0f / static_cast<float>(network_dims[1]);
        if (width <= min_norm_w || height <= min_norm_h)
        {
            continue;
        }

        // Reject boxes that become zero-sized after integer quantization in pixel space.
        int net_xmin = static_cast<int>(std::floor(xmin * network_dims[0]));
        int net_xmax = static_cast<int>(std::ceil(xmax * network_dims[0]));
        int net_ymin = static_cast<int>(std::floor(ymin * network_dims[1]));
        int net_ymax = static_cast<int>(std::ceil(ymax * network_dims[1]));
        if ((net_xmax - net_xmin) <= 0 || (net_ymax - net_ymin) <= 0)
        {
            continue;
        }

        // Also validate proto-space crop dimensions (proto is stride 4 in this model).
        int proto_w = network_dims[0] / 4;
        int proto_h = network_dims[1] / 4;
        int proto_xmin = static_cast<int>(std::floor(xmin * proto_w));
        int proto_xmax = static_cast<int>(std::ceil(xmax * proto_w));
        int proto_ymin = static_cast<int>(std::floor(ymin * proto_h));
        int proto_ymax = static_cast<int>(std::ceil(ymax * proto_h));
        if ((proto_xmax - proto_xmin) <= 0 || (proto_ymax - proto_ymin) <= 0)
        {
            continue;
        }

        HailoBBox bbox(xmin, ymin, width, height);

        int hailo_class_index = class_index + 1;
        std::string label;
        if (hailo_class_index >= 0 && hailo_class_index < static_cast<int>(common::coco_eighty.size()))
        {
            label = common::coco_eighty[hailo_class_index];
        }
        else
        {
            label = "class_" + std::to_string(class_index);
        }
        HailoDetection detection(bbox, hailo_class_index, label, confidence);

        auto mask_coefficients = xt::eval(xt::view(masks, proposal_index, xt::all()));
        std::vector<float> mask_data(mask_coefficients.size());
        memcpy(mask_data.data(), mask_coefficients.data(), sizeof(float) * mask_coefficients.size());
        detection.add_object(std::make_shared<HailoMatrix>(mask_data, mask_coefficients.size(), 1));

        detections.push_back(detection);
    }
    return detections;
}

std::vector<HailoDetection> yolov8seg_post(
    const std::vector<ScaleTensors> &scales,
    const std::vector<int> &network_dims,
    float iou_threshold,
    float score_threshold)
{
    if (scales.empty())
    {
        return {};
    }

    std::vector<int> strides;
    strides.reserve(scales.size());
    for (const auto &scale : scales)
    {
        strides.push_back(scale.stride);
    }
    auto centers = get_centers(strides, network_dims);

    std::vector<HailoDetection> all_detections;
    for (size_t i = 0; i < scales.size(); i++)
    {
        auto detections = decode_scale(scales[i], centers[i], network_dims, score_threshold);
        all_detections.insert(all_detections.end(), detections.begin(), detections.end());
    }

    common::nms(all_detections, iou_threshold);
    return all_detections;
}
} // namespace

Yolov8segParams *init(const std::string config_path, const std::string function_name)
{
    auto *params = new Yolov8segParams();
    (void)function_name;
    if (!fs::exists(config_path))
    {
        std::cerr << "Config file '" << config_path << "' not found, using default parameters" << std::endl;
        return params;
    }

    char config_buffer[4096];
    // Schema covers all supported fields; outputs_name sub-object is optional.
    const char *json_schema = R""""({
      "$schema": "http://json-schema.org/draft-07/schema#",
      "type": "object",
        "properties": {
        "iou_threshold":   {"type": "number", "minimum": 0, "maximum": 1},
        "score_threshold": {"type": "number", "minimum": 0, "maximum": 1},
        "num_classes":     {"type": "integer", "minimum": 1},
        "input_shape":     {
          "type": "array",
          "items": {"type": "integer", "minimum": 1},
          "minItems": 2,
          "maxItems": 2
        },
        "outputs_name": {
          "type": "object",
          "required": ["proto", "boxes", "scores", "masks"],
          "properties": {
            "proto":  {"type": "string"},
            "boxes":  {
              "type": "array",
              "items": {"type": "string"},
              "minItems": 3,
              "maxItems": 3
            },
            "scores": {
              "type": "array",
              "items": {"type": "string"},
              "minItems": 3,
              "maxItems": 3
            },
            "masks":  {
              "type": "array",
              "items": {"type": "string"},
              "minItems": 3,
              "maxItems": 3
            }
          }
        }
      }
    })"""";

    std::FILE *fp = fopen(config_path.c_str(), "r");
    if (fp == nullptr)
    {
        throw std::runtime_error("JSON config file is not valid");
    }
    rapidjson::FileReadStream stream(fp, config_buffer, sizeof(config_buffer));
    if (common::validate_json_with_schema(stream, json_schema))
    {
        std::rewind(fp);
        rapidjson::FileReadStream parse_stream(fp, config_buffer, sizeof(config_buffer));
        rapidjson::Document doc;
        doc.ParseStream(parse_stream);
        if (doc.HasParseError() || !doc.IsObject())
        {
            fclose(fp);
            throw std::runtime_error("JSON config file parse failed");
        }

        if (doc.HasMember("iou_threshold"))
        {
            params->iou_threshold = doc["iou_threshold"].GetFloat();
        }
        if (doc.HasMember("score_threshold"))
        {
            params->score_threshold = doc["score_threshold"].GetFloat();
        }
        if (doc.HasMember("num_classes"))
        {
            params->num_classes = doc["num_classes"].GetInt();
        }
        if (doc.HasMember("input_shape"))
        {
            auto config_input_shape = doc["input_shape"].GetArray();
            params->input_shape.clear();
            for (uint j = 0; j < config_input_shape.Size(); j++)
            {
                params->input_shape.push_back(config_input_shape[j].GetInt());
            }
        }
        if (doc.HasMember("outputs_name"))
        {
            const auto &on = doc["outputs_name"];
            if (on.HasMember("proto"))
            {
                params->outputs_name.proto = on["proto"].GetString();
            }

            auto parse_string_array = [](const rapidjson::Value &arr) {
                std::vector<std::string> result;
                for (uint i = 0; i < arr.Size(); i++)
                {
                    result.push_back(arr[i].GetString());
                }
                return result;
            };

            if (on.HasMember("boxes"))
                params->outputs_name.boxes  = parse_string_array(on["boxes"]);
            if (on.HasMember("scores"))
                params->outputs_name.scores = parse_string_array(on["scores"]);
            if (on.HasMember("masks"))
                params->outputs_name.masks  = parse_string_array(on["masks"]);

            if (!params->outputs_name.is_set())
            {
                std::cerr << "Warning: 'outputs_name' in config is incomplete; "
                             "falling back to channel-count heuristic" << std::endl;
                params->outputs_name = Yolov8segOutputsName{};
            }
        }
    }
    fclose(fp);
    return params;
}

void free_resources(void *params_void_ptr)
{
    delete reinterpret_cast<Yolov8segParams *>(params_void_ptr);
}

void filter(HailoROIPtr roi, void *params_void_ptr)
{
    auto *params = reinterpret_cast<Yolov8segParams *>(params_void_ptr);
    if (!roi->has_tensors())
    {
        return;
    }
    if (params->input_shape.size() < 2 || params->input_shape[0] <= 0 || params->input_shape[1] <= 0)
    {
        std::cerr << "Warning: invalid input_shape in params; skipping frame" << std::endl;
        return;
    }

    std::vector<int> network_dims = {params->input_shape[0], params->input_shape[1]};
    auto tensors = roi->get_tensors();

    std::vector<ScaleTensors> scales;
    HailoTensorPtr proto_tensor_ptr;

    if (params->outputs_name.is_set())
    {
        // ── Name-based lookup ─────────────────────────────────────────────────
        // Fully robust: ignores channel counts, works for any num_classes and
        // any model naming convention.
        auto tensors_by_name = roi->get_tensors_by_name();
        scales = collect_scale_tensors_by_name(tensors_by_name, params->outputs_name, network_dims[0], params->num_classes);
        proto_tensor_ptr = find_proto_tensor_by_name(tensors_by_name, params->outputs_name.proto);
    }
    else
    {
        // ── Channel-count heuristic ───────────────────────────────────────────
        // Uses params->num_classes (default 80, JSON-overridable) so that
        // retrained models with a different class count still work correctly
        // without requiring explicit tensor names.
        scales = collect_scale_tensors(tensors, network_dims[0], params->num_classes);
        proto_tensor_ptr = find_proto_tensor(tensors, network_dims[0]);
    }

    if (scales.empty())
    {
        std::cerr << "Warning: no valid scale tensors found; check 'num_classes' or 'outputs_name' in config" << std::endl;
        return;
    }
    if (proto_tensor_ptr == nullptr)
    {
        std::cerr << "Warning: prototype tensor not found" << std::endl;
        return;
    }

    auto detections = yolov8seg_post(scales, network_dims, params->iou_threshold, params->score_threshold);
    auto proto_tensor = common::dequantize(
        common::get_xtensor(proto_tensor_ptr),
        proto_tensor_ptr->quant_info().qp_scale,
        proto_tensor_ptr->quant_info().qp_zp);

    decode_masks(detections, proto_tensor);
    hailo_common::add_detections(roi, detections);
}

void yolov8seg(HailoROIPtr roi, void *params_void_ptr)
{
    filter(roi, params_void_ptr);
}

void filter_letterbox(HailoROIPtr roi, void *params_void_ptr)
{
    auto *params = reinterpret_cast<Yolov8segParams *>(params_void_ptr);
    filter(roi, params_void_ptr);
    HailoBBox roi_bbox = hailo_common::create_flattened_bbox(roi->get_bbox(), roi->get_scaling_bbox());
    auto detections = hailo_common::get_hailo_detections(roi);
    const int input_w = (params->input_shape.size() >= 1) ? params->input_shape[0] : 640;
    const int input_h = (params->input_shape.size() >= 2) ? params->input_shape[1] : 640;
    const float kMinBoxSize = 1.0f / static_cast<float>(std::max(input_w, input_h));
    for (auto &detection : detections)
    {
        auto detection_bbox = detection->get_bbox();
        float xmin = (detection_bbox.xmin() * roi_bbox.width()) + roi_bbox.xmin();
        float ymin = (detection_bbox.ymin() * roi_bbox.height()) + roi_bbox.ymin();
        float xmax = (detection_bbox.xmax() * roi_bbox.width()) + roi_bbox.xmin();
        float ymax = (detection_bbox.ymax() * roi_bbox.height()) + roi_bbox.ymin();

        if (!std::isfinite(xmin) || !std::isfinite(ymin) || !std::isfinite(xmax) || !std::isfinite(ymax))
        {
            continue;
        }

        xmin = std::clamp(xmin, 0.0f, 1.0f);
        ymin = std::clamp(ymin, 0.0f, 1.0f);
        xmax = std::clamp(xmax, 0.0f, 1.0f);
        ymax = std::clamp(ymax, 0.0f, 1.0f);

        if (xmax <= xmin)
        {
            xmax = std::min(1.0f, xmin + kMinBoxSize);
        }
        if (ymax <= ymin)
        {
            ymax = std::min(1.0f, ymin + kMinBoxSize);
        }

        float width = xmax - xmin;
        float height = ymax - ymin;
        if (width < kMinBoxSize)
        {
            width = kMinBoxSize;
            if (xmin + width > 1.0f)
            {
                xmin = std::max(0.0f, 1.0f - width);
            }
        }
        if (height < kMinBoxSize)
        {
            height = kMinBoxSize;
            if (ymin + height > 1.0f)
            {
                ymin = std::max(0.0f, 1.0f - height);
            }
        }

        detection->set_bbox(HailoBBox(xmin, ymin, width, height));
    }
    roi->clear_scaling_bbox();
}
