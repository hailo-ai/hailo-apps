#include "instance_seg_postprocess.hpp"
#include "toolbox.hpp"
#include "general/math.hpp"
#include "general/nms.hpp"
#include "general/tensors.hpp"
#include "labels/coco_eighty.hpp"

using namespace hailo_utils;
using namespace xt::placeholders;

#define SCORE_THRESHOLD 0.6f
#define IOU_THRESHOLD 0.7f
#define NUM_CLASSES 80

std::vector<cv::Scalar> COLORS = {
    cv::Scalar(244, 67, 54),
    cv::Scalar(233, 30, 99),
    cv::Scalar(156, 39, 176),
    cv::Scalar(103, 58, 183),
    cv::Scalar(63, 81, 181),
    cv::Scalar(33, 150, 243),
    cv::Scalar(3, 169, 244),
    cv::Scalar(0, 188, 212),
    cv::Scalar(0, 150, 136),
    cv::Scalar(76, 175, 80),
    cv::Scalar(139, 195, 74),
    cv::Scalar(205, 220, 57),
    cv::Scalar(255, 235, 59),
    cv::Scalar(255, 193, 7),
    cv::Scalar(255, 152, 0),
    cv::Scalar(255, 87, 34),
    cv::Scalar(121, 85, 72),
    cv::Scalar(158, 158, 158),
    cv::Scalar(96, 125, 139),
    cv::Scalar(0, 0, 0)
};

cv::Vec3b indexToColor(size_t index)
{
    return color_table[index % color_table.size()];
}

std::vector<const hailo_detection_with_byte_mask_t*> get_detections(const uint8_t *src_ptr)
{
    std::vector<const hailo_detection_with_byte_mask_t*> detections;
    const uint16_t detections_count = *reinterpret_cast<const uint16_t*>(src_ptr);
    detections.reserve(detections_count);

    size_t offset = sizeof(uint16_t);
    for (size_t i = 0; i < detections_count; ++i) {
        const auto *det_ptr =
            reinterpret_cast<const hailo_detection_with_byte_mask_t*>(src_ptr + offset);
        offset += sizeof(hailo_detection_with_byte_mask_t) + det_ptr->mask_size;
        detections.emplace_back(det_ptr);
    }
    return detections;
}

cv::Mat draw_detections_and_mask(const uint8_t *src_ptr,
                                          int width, int height,
                                          cv::Mat &frame)
{
    auto detections = get_detections(src_ptr);
    cv::Mat overlay = cv::Mat::zeros(height, width, CV_8UC3);
    for (const auto *detection : detections) {
        const int box_w = static_cast<int>(std::ceil((detection->box.x_max - detection->box.x_min) * width));
        const int box_h = static_cast<int>(std::ceil((detection->box.y_max - detection->box.y_min) * height));
        const cv::Vec3b color = indexToColor(detection->class_id);

        const uint8_t *mask_ptr = reinterpret_cast<const uint8_t*>(detection) + sizeof(hailo_detection_with_byte_mask_t);
        const size_t expected = static_cast<size_t>(box_w) * static_cast<size_t>(box_h);
        (void)expected;

        for (int i = 0; i < box_h; ++i) {
            for (int j = 0; j < box_w; ++j) {
                const size_t idx = static_cast<size_t>(i) * static_cast<size_t>(box_w) + static_cast<size_t>(j);
                if (idx < detection->mask_size && mask_ptr[idx]) {
                    const int ox = j + static_cast<int>(detection->box.x_min * width);
                    const int oy = i + static_cast<int>(detection->box.y_min * height);
                    if (0 <= ox && ox < width && 0 <= oy && oy < height) {
                        overlay.at<cv::Vec3b>(oy, ox) = color;
                    }
                }
            }
        }
        cv::rectangle(
            frame,
            cv::Rect(static_cast<int>(detection->box.x_min * width),
                     static_cast<int>(detection->box.y_min * height),
                     box_w, box_h),
            color, 1);
    }

    cv::addWeighted(frame, 1.0, overlay, 0.7, 0.0, frame);
    return frame;
}

cv::Mat pad_frame_letterbox(const cv::Mat &frame, int model_h, int model_w)
{
    // scale to fit (preserve aspect), pad bottom & right
    const float fh = static_cast<float>(frame.rows) / static_cast<float>(model_h);
    const float fw = static_cast<float>(frame.cols) / static_cast<float>(model_w);
    const float factor = std::max(fh, fw); // exactly like your old code

    cv::Mat resized;
    cv::resize(frame, resized,
               cv::Size(static_cast<int>(std::round(frame.cols / factor)),
                        static_cast<int>(std::round(frame.rows / factor))),
               0, 0, cv::INTER_AREA);

    const int pad_h = std::max(0, model_h - resized.rows);
    const int pad_w = std::max(0, model_w - resized.cols);

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, 0, pad_h, 0, pad_w,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return padded;
}

cv::Mat make_model_space_canvas(const cv::Mat &src,
    int model_w, int model_h,
    LetterboxMap &map)
{
const float fh = static_cast<float>(src.rows) / static_cast<float>(model_h);
const float fw = static_cast<float>(src.cols) / static_cast<float>(model_w);
map.factor = std::max(fh, fw);

cv::Mat resized;
cv::resize(src, resized,
cv::Size(static_cast<int>(std::round(src.cols / map.factor)),
static_cast<int>(std::round(src.rows / map.factor))),
0, 0, cv::INTER_AREA);

map.pad_h = std::max(0, model_h - resized.rows);
map.pad_w = std::max(0, model_w - resized.cols);

cv::Mat model_space;
cv::copyMakeBorder(resized, model_space, 0, map.pad_h, 0, map.pad_w,
cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

map.crop_h = static_cast<int>(std::round(src.rows / map.factor));
map.crop_w = static_cast<int>(std::round(src.cols / map.factor));
return model_space;
}

void map_model_to_frame(const cv::Mat &model_space,
    const LetterboxMap &map,
    cv::Mat &dst_frame)
{
const int cw = std::min(map.crop_w, model_space.cols);
const int ch = std::min(map.crop_h, model_space.rows);
const cv::Rect roi(0, 0, cw, ch);

cv::Mat cropped = model_space(roi).clone();
cv::resize(cropped, dst_frame, dst_frame.size(), 0, 0, cv::INTER_LINEAR);
} 

// -------------------- helpers - HEF without HailoRT-Postprocess  --------------------

HailoROIPtr build_roi_from_outputs(
    const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &outputs)
{
    // Full-frame ROI in normalized coordinates
    auto roi = std::make_shared<HailoROI>(HailoBBox(0.f, 0.f, 1.f, 1.f));

    for (const auto &o : outputs) {
        // HailoTensor: thin wrapper over (ptr + vstream_info).
        // If your HailoTensor ctor signature differs, adjust accordingly.
        auto tensor = std::make_shared<HailoTensor>(o.first, o.second);
        roi->add_tensor(tensor);
    }
    return roi;
}


std::vector<HailoDetectionPtr> get_detections_from_roi(const HailoROIPtr &roi)
{
    std::vector<HailoDetectionPtr> dets;
    for (auto &obj : roi->get_objects_typed(HAILO_DETECTION)) {
        if (auto det = std::dynamic_pointer_cast<HailoDetection>(obj)) {
            dets.emplace_back(det);
        }
    }
    return dets;
}

void draw_masks_and_boxes(
    cv::Mat &frame,
    const std::vector<HailoDetectionPtr> &dets,
    const std::vector<cv::Mat> &masks,
    float alpha,
    float thresh)
{
    (void)alpha; 
    (void)thresh; 
    CV_Assert(static_cast<int>(dets.size()) == static_cast<int>(masks.size()));
    cv::Mat overlay(frame.size(), CV_8UC3, cv::Scalar(0,0,0));

    for (size_t i = 0; i < dets.size(); ++i) {
        const auto &det = dets[i];
        cv::Mat mask = masks[i];
        if (mask.size() != frame.size()) {
            cv::resize(mask, mask, frame.size(), 0, 0, cv::INTER_LINEAR);
        }

        const int cls_id = std::max(0, det->get_class_id());
        const cv::Scalar color = COLORS[static_cast<size_t>(cls_id) % COLORS.size()];

        // Fast threshold blend
        for (int y = 0; y < mask.rows; ++y) {
            const float *mp = mask.ptr<float>(y);
            cv::Vec3b *op = overlay.ptr<cv::Vec3b>(y);
            for (int x = 0; x < mask.cols; ++x) {
                if (mp[x] > thresh) {
                    op[x][0] = static_cast<uchar>(color[0]);
                    op[x][1] = static_cast<uchar>(color[1]);
                    op[x][2] = static_cast<uchar>(color[2]);
                }
            }
        }

        // Draw bbox (HailoBBox is normalized)
        const auto &bb = det->get_bbox();
        const int x = static_cast<int>(bb.xmin() * frame.cols);
        const int y = static_cast<int>(bb.ymin() * frame.rows);
        const int w = static_cast<int>((bb.xmax() - bb.xmin()) * frame.cols);
        const int h = static_cast<int>((bb.ymax() - bb.ymin()) * frame.rows);
        cv::rectangle(frame, cv::Rect(x, y, w, h), color, 1);
    }

    cv::addWeighted(frame, 1.0, overlay, alpha, 0.0, frame);
}


// ====================  FUNCTIONS ====================

cv::Mat xarray_to_mat(xt::xarray<float> xarr) {
    cv::Mat mat (xarr.shape()[0], xarr.shape()[1], CV_32FC1, xarr.data(), 0);
    return mat;
}

void sigmoid(float *data, const int size) {
    for (int i = 0; i < size; i++)
        data[i] = 1.0f / (1.0f + std::exp(-1.0 * data[i]));
}

cv::Mat crop_mask(cv::Mat mask, HailoBBox box) {
    auto x_min = box.xmin();
    auto y_min = box.ymin(); 
    auto x_max = box.xmax(); 
    auto y_max = box.ymax();

    int rows = mask.rows;
    int cols = mask.cols;

    // Ensure ROI coordinates are within the valid range
    int top_start = std::max(0, static_cast<int>(std::ceil(y_min * rows)));
    int bottom_end = std::min(rows, static_cast<int>(std::ceil(y_max * rows)));
    int left_start = std::max(0, static_cast<int>(std::ceil(x_min * cols)));
    int right_end = std::min(cols, static_cast<int>(std::ceil(x_max * cols)));

    // Create ROI rectangles
    cv::Rect top_roi(0, 0, cols, top_start);
    cv::Rect bottom_roi(0, bottom_end, cols, rows - bottom_end);
    cv::Rect left_roi(0, 0, left_start, rows);
    cv::Rect right_roi(right_end, 0, cols - right_end, rows);

    // Set values to zero in the specified ROIs
    mask(top_roi) = 0;
    mask(bottom_roi) = 0;
    mask(left_roi) = 0;
    mask(right_roi) = 0;
    
    return mask;
}

xt::xarray<float> dot(xt::xarray<float> mask, xt::xarray<float> reshaped_proto, 
                    size_t proto_height, size_t proto_width, size_t mask_num = 32){
    
    auto shape = {proto_height, proto_width};
    xt::xarray<float> mask_product(shape);

    for (size_t i = 0; i < mask_product.shape(0); i++) {
        for (size_t j = 0; j < mask_product.shape(1); j++) {
            for (size_t k = 0; k < mask_num; k++) {
                mask_product(i,j) += mask(k) * reshaped_proto(k, i, j);
            }
        }
    }
    return mask_product;
}

std::vector<DetectionAndMask> decode_masks(std::vector<std::pair<HailoDetection, xt::xarray<float>>> detections_and_masks_after_nms, 
                                                                        xt::xarray<float> proto, int org_image_height, int org_image_width){
    
    std::vector<DetectionAndMask> detections_and_cropped_masks(detections_and_masks_after_nms.size(), 
                                                                DetectionAndMask({
                                                                    HailoDetection(HailoBBox(0.0,0.0,0.0,0.0), "", 0.0), 
                                                                    cv::Mat(org_image_height, org_image_width, CV_32FC1)}
                                                                    ));

    int mask_height = static_cast<int>(proto.shape(0));
    int mask_width = static_cast<int>(proto.shape(1));
    int mask_features = static_cast<int>(proto.shape(2));

    auto reshaped_proto = xt::reshape_view(xt::transpose(xt::reshape_view(proto, {-1, mask_features}), {1,0}), {-1, mask_height, mask_width});
    
    for (int i = 0; i < detections_and_masks_after_nms.size(); i++) {

        auto curr_detection = detections_and_masks_after_nms[i].first;
        auto curr_mask = detections_and_masks_after_nms[i].second;

        auto mask_product = dot(curr_mask, reshaped_proto, reshaped_proto.shape(1), reshaped_proto.shape(2), curr_mask.shape(0));

        sigmoid(mask_product.data(), mask_product.size());

        cv::Mat mask = xarray_to_mat(mask_product).clone();
        cv::resize(mask, mask, cv::Size(org_image_width, org_image_height), 0, 0, cv::INTER_LINEAR);

        mask = crop_mask(mask, curr_detection.get_bbox());

        detections_and_cropped_masks[i] = DetectionAndMask({curr_detection, mask});
    }

    return detections_and_cropped_masks;
}

float iou_calc(const HailoBBox &box_1, const HailoBBox &box_2)
{
    // Calculate IOU between two detection boxes
    const float width_of_overlap_area = std::min(box_1.xmax(), box_2.xmax()) - std::max(box_1.xmin(), box_2.xmin());
    const float height_of_overlap_area = std::min(box_1.ymax(), box_2.ymax()) - std::max(box_1.ymin(), box_2.ymin());
    const float positive_width_of_overlap_area = std::max(width_of_overlap_area, 0.0f);
    const float positive_height_of_overlap_area = std::max(height_of_overlap_area, 0.0f);
    const float area_of_overlap = positive_width_of_overlap_area * positive_height_of_overlap_area;
    const float box_1_area = (box_1.ymax() - box_1.ymin()) * (box_1.xmax() - box_1.xmin());
    const float box_2_area = (box_2.ymax() - box_2.ymin()) * (box_2.xmax() - box_2.xmin());
    // The IOU is a ratio of how much the boxes overlap vs their size outside the overlap.
    // Boxes that are similar will have a higher overlap threshold.
    return area_of_overlap / (box_1_area + box_2_area - area_of_overlap);
}
std::vector<std::pair<HailoDetection, xt::xarray<float>>> nms(std::vector<std::pair<HailoDetection, xt::xarray<float>>> &detections_and_masks, 
                                                            const float iou_thr, bool should_nms_cross_classes = false) {

    std::vector<std::pair<HailoDetection, xt::xarray<float>>> detections_and_masks_after_nms;

    for (uint index = 0; index < detections_and_masks.size(); index++)
    {
        if (detections_and_masks[index].first.get_confidence() != 0.0f)
        {
            for (uint jindex = index + 1; jindex < detections_and_masks.size(); jindex++)
            {
                if ((should_nms_cross_classes || (detections_and_masks[index].first.get_class_id() == detections_and_masks[jindex].first.get_class_id())) &&
                    detections_and_masks[jindex].first.get_confidence() != 0.0f)
                {
                    // For each detection, calculate the IOU against each following detection.
                    float iou = iou_calc(detections_and_masks[index].first.get_bbox(), detections_and_masks[jindex].first.get_bbox());
                    // If the IOU is above threshold, then we have two similar detections,
                    // and want to delete the one.
                    if (iou >= iou_thr)
                    {
                        // The detections are arranged in highest score order,
                        // so we want to erase the latter detection.
                        detections_and_masks[jindex].first.set_confidence(0.0f);
                    }
                }
            }
        }
    }
    for (uint index = 0; index < detections_and_masks.size(); index++)
    {
        if (detections_and_masks[index].first.get_confidence() != 0.0f)
        {
            detections_and_masks_after_nms.push_back(std::make_pair(detections_and_masks[index].first, detections_and_masks[index].second));
        }
    }
    return detections_and_masks_after_nms;
}

float dequantize_value(uint8_t val, float32_t qp_scale, float32_t qp_zp){
    return (float(val) - qp_zp) * qp_scale;
}

void dequantize_mask_values(xt::xarray<float>& dequantized_outputs, int index, 
                        xt::xarray<uint8_t>& quantized_outputs,
                        size_t dim1, float32_t qp_scale, float32_t qp_zp){
    for (size_t i = 0; i < dim1; i++){
        dequantized_outputs(i) = dequantize_value(quantized_outputs(index, i), qp_scale, qp_zp);
    }
}
                   
void dequantize_box_values(xt::xarray<float>& dequantized_outputs, int index, 
                        xt::xarray<uint8_t>& quantized_outputs,
                        size_t dim1, size_t dim2, float32_t qp_scale, float32_t qp_zp){
    for (size_t i = 0; i < dim1; i++){
        for (size_t j = 0; j < dim2; j++){
            dequantized_outputs(i, j) = dequantize_value(quantized_outputs(index, i, j), qp_scale, qp_zp);
        }
    }
}

std::vector<xt::xarray<double>> get_centers(std::vector<int>& strides, std::vector<int>& network_dims,
                                        std::size_t boxes_num, int strided_width, int strided_height){

        std::vector<xt::xarray<double>> centers(boxes_num);

        for (uint i=0; i < boxes_num; i++) {
            strided_width = network_dims[0] / strides[i];
            strided_height = network_dims[1] / strides[i];

            // Create a meshgrid of the proper strides
            xt::xarray<int> grid_x = xt::arange(0, strided_width);
            xt::xarray<int> grid_y = xt::arange(0, strided_height);

            auto mesh = xt::meshgrid(grid_x, grid_y);
            grid_x = std::get<1>(mesh);
            grid_y = std::get<0>(mesh);

            // Use the meshgrid to build up box center prototypes
            auto ct_row = (xt::flatten(grid_y) + 0.5) * strides[i];
            auto ct_col = (xt::flatten(grid_x) + 0.5) * strides[i];

            centers[i] = xt::stack(xt::xtuple(ct_col, ct_row, ct_col, ct_row), 1);
        }

        return centers;
}
std::vector<std::pair<HailoDetection, xt::xarray<float>>> decode_boxes_and_extract_masks(std::vector<HailoTensorPtr> raw_boxes_outputs,
                                                                                std::vector<HailoTensorPtr> raw_masks_outputs,
                                                                                xt::xarray<float> scores,
                                                                                std::vector<int> network_dims,
                                                                                std::vector<int> strides,
                                                                                int regression_length) {
    int strided_width, strided_height, class_index;
    std::vector<std::pair<HailoDetection, xt::xarray<float>>> detections_and_masks;
    int instance_index = 0;
    float confidence = 0.0;
    std::string label;

    auto centers = get_centers(std::ref(strides), std::ref(network_dims), raw_boxes_outputs.size(), strided_width, strided_height);

    // Box distribution to distance
    auto regression_distance =  xt::reshape_view(xt::arange(0, regression_length + 1), {1, 1, regression_length + 1});

    for (uint i = 0; i < raw_boxes_outputs.size(); i++)
    {
        // Boxes setup
        float32_t qp_scale = raw_boxes_outputs[i]->vstream_info().quant_info.qp_scale;
        float32_t qp_zp = raw_boxes_outputs[i]->vstream_info().quant_info.qp_zp;

        auto output_b = common::get_xtensor(raw_boxes_outputs[i]);
        int num_proposals = output_b.shape(0) * output_b.shape(1);
        auto output_boxes = xt::view(output_b, xt::all(), xt::all(), xt::all());
        xt::xarray<uint8_t> quantized_boxes = xt::reshape_view(output_boxes, {num_proposals, 4, regression_length + 1});

        auto shape = {quantized_boxes.shape(1), quantized_boxes.shape(2)};

        // Masks setup
        float32_t qp_scale_mask = raw_masks_outputs[i]->vstream_info().quant_info.qp_scale;
        float32_t qp_zp_mask = raw_masks_outputs[i]->vstream_info().quant_info.qp_zp;

        auto output_m = common::get_xtensor(raw_masks_outputs[i]);
        int num_proposals_masks = output_m.shape(0) * output_m.shape(1);
        auto output_masks = xt::view(output_m, xt::all(), xt::all(), xt::all());
        xt::xarray<uint8_t> quantized_masks = xt::reshape_view(output_masks, {num_proposals_masks, 32});

        auto mask_shape = {quantized_masks.shape(1)};

        // Bbox decoding
        for (uint j = 0; j < num_proposals; j++) {
            class_index = xt::argmax(xt::row(scores, instance_index))(0);
            confidence = scores(instance_index, class_index);
            instance_index++;
            if (confidence < SCORE_THRESHOLD)
                continue;

            xt::xarray<float> box(shape);
    
            dequantize_box_values(box, j, quantized_boxes, 
                                    box.shape(0), box.shape(1), 
                                    qp_scale, qp_zp);
            common::softmax_2D(box.data(), box.shape(0), box.shape(1));

            xt::xarray<float> mask(mask_shape);

            dequantize_mask_values(mask, j, quantized_masks, 
                                    mask.shape(0), qp_scale_mask, 
                                    qp_zp_mask);

            auto box_distance = box * regression_distance;
            xt::xarray<float> reduced_distances = xt::sum(box_distance, {2});
            auto strided_distances = reduced_distances * strides[i];

            // Decode box
            auto distance_view1 = xt::view(strided_distances, xt::all(), xt::range(_, 2)) * -1;
            auto distance_view2 = xt::view(strided_distances, xt::all(), xt::range(2, _));
            auto distance_view = xt::concatenate(xt::xtuple(distance_view1, distance_view2), 1);
            auto decoded_box = centers[i] + distance_view;

            HailoBBox bbox(decoded_box(j, 0) / network_dims[0],
                           decoded_box(j, 1) / network_dims[1],
                           (decoded_box(j, 2) - decoded_box(j, 0)) / network_dims[0],
                           (decoded_box(j, 3) - decoded_box(j, 1)) / network_dims[1]);

            label = common::coco_eighty[class_index + 1];
            HailoDetection detected_instance(bbox, class_index, label, confidence);

            detections_and_masks.push_back(std::make_pair(detected_instance, mask));

        }
    }

    return detections_and_masks;
}

HailoTensorPtr pop_proto(std::vector<HailoTensorPtr> &tensors){
    auto it = tensors.begin();
    while (it != tensors.end()) {
        auto tensor = *it;
        if (tensor->features() == 32 && tensor->height() == 160 && tensor->width() == 160){
            auto proto = tensor;
            tensors.erase(it);
            return proto;
        }
        else{
            ++it;
        }
    }
    return nullptr;
}

Quadruple get_boxes_scores_masks(std::vector<HailoTensorPtr> &tensors, int num_classes, int regression_length){

    auto raw_proto = pop_proto(tensors);

    std::vector<HailoTensorPtr> outputs_boxes(tensors.size() / 3);
    std::vector<HailoTensorPtr> outputs_masks(tensors.size() / 3);
    
    // Prepare the scores xarray at the size we will fill in in-place
    int total_scores = 0;
    for (int i = 0; i < tensors.size(); i = i + 3) {
        total_scores += tensors[i+1]->width() * tensors[i+1]->height();
    }

    std::vector<size_t> scores_shape = { (long unsigned int)total_scores, (long unsigned int)num_classes };
    
    xt::xarray<float> scores(scores_shape);

    std::vector<size_t> proto_shape = { {(long unsigned int)raw_proto->height(), 
                                                (long unsigned int)raw_proto->width(), 
                                                (long unsigned int)raw_proto->features()} };
    xt::xarray<float> proto(proto_shape);

    int view_index_scores = 0;

    for (uint i = 0; i < tensors.size(); i = i + 3)
    {
        // Bounding boxes extraction will be done later on only on the boxes that surpass the score threshold
        outputs_boxes[i / 3] = tensors[i];

        // Extract and dequantize the scores outputs
        auto dequantized_output_s = common::dequantize(common::get_xtensor(tensors[i+1]), tensors[i+1]->vstream_info().quant_info.qp_scale, tensors[i+1]->vstream_info().quant_info.qp_zp);
        int num_proposals_scores = dequantized_output_s.shape(0)*dequantized_output_s.shape(1);

        // From the layer extract the scores
        auto output_scores = xt::view(dequantized_output_s, xt::all(), xt::all(), xt::all());
        xt::view(scores, xt::range(view_index_scores, view_index_scores + num_proposals_scores), xt::all()) = xt::reshape_view(output_scores, {num_proposals_scores, num_classes});
        view_index_scores += num_proposals_scores;

        // Keypoints extraction will be done later according to the boxes that surpass the threshold
        outputs_masks[i / 3] = tensors[i+2];
    }
    
    proto = common::dequantize(common::get_xtensor(raw_proto), raw_proto->vstream_info().quant_info.qp_scale, raw_proto->vstream_info().quant_info.qp_zp);
    
    return Quadruple{outputs_boxes, scores, outputs_masks, proto};
}

std::vector<DetectionAndMask> segmentation_postprocess(std::vector<HailoTensorPtr> &tensors,
                                                                                std::vector<int> network_dims,
                                                                                std::vector<int> strides,
                                                                                int regression_length,
                                                                                int num_classes,
                                                                                int org_image_height, 
                                                                                int org_image_width) {
    std::vector<DetectionAndMask> detections_and_cropped_masks;
    if (tensors.size() == 0)
    {
        return detections_and_cropped_masks;
    }

    Quadruple boxes_scores_masks_mask_matrix = get_boxes_scores_masks(tensors, num_classes, regression_length);

    std::vector<HailoTensorPtr> raw_boxes = boxes_scores_masks_mask_matrix.boxes;
    xt::xarray<float> scores = boxes_scores_masks_mask_matrix.scores;
    std::vector<HailoTensorPtr> raw_masks = boxes_scores_masks_mask_matrix.masks;
    xt::xarray<float> proto = boxes_scores_masks_mask_matrix.proto_data;

    // Decode the boxes and get masks
    auto detections_and_masks = decode_boxes_and_extract_masks(raw_boxes, raw_masks, scores, network_dims, strides, regression_length);

    // Filter with NMS
    auto detections_and_masks_after_nms = nms(detections_and_masks, IOU_THRESHOLD, true);

    // Decode the masking
    auto detections_and_decoded_masks = decode_masks(detections_and_masks_after_nms, proto, org_image_height, org_image_width);

    return detections_and_decoded_masks;
}

std::vector<cv::Mat> filter(HailoROIPtr roi, int org_image_height, int org_image_width)
{
    // anchor params
    int regression_length = 15;
    std::vector<int> strides = {8, 16, 32};
    std::vector<int> network_dims = {640, 640};

    std::vector<HailoTensorPtr> tensors = roi->get_tensors();
    auto filtered_detections_and_masks = segmentation_postprocess(tensors, 
                                                            network_dims, 
                                                            strides, 
                                                            regression_length, 
                                                            NUM_CLASSES, 
                                                            org_image_height, 
                                                            org_image_width);

    std::vector<HailoDetection> detections;
    std::vector<cv::Mat> masks;

    for (auto& det_and_msk : filtered_detections_and_masks){
        detections.push_back(det_and_msk.detection);
        masks.push_back(det_and_msk.mask);
    }

    hailo_common::add_detections(roi, detections);

    return masks;
}
