#include "instance_seg_postprocess.hpp"
#include <algorithm>
#include <cmath>
#include "toolbox.hpp"

using namespace xt::placeholders;

static const std::vector<cv::Scalar> PALETTE = {
    {244,  67,  54}, {233,  30,  99}, {156,  39, 176}, {103,  58, 183},
    { 63,  81, 181}, { 33, 150, 243}, {  3, 169, 244}, {  0, 188, 212},
    {  0, 150, 136}, { 76, 175,  80}, {139, 195,  74}, {205, 220,  57},
    {255, 235,  59}, {255, 193,   7}, {255, 152,   0}, {255,  87,  34},
    {121,  85,  72}, {158, 158, 158}, { 96, 125, 139}, {  0,   0,   0}
};


cv::Mat pad_frame_letterbox(const cv::Mat &frame, int model_h, int model_w) {
     const float fh = static_cast<float>(frame.rows) / static_cast<float>(model_h);
     const float fw = static_cast<float>(frame.cols) / static_cast<float>(model_w);
     const float factor = std::max(fh, fw); // fit inside model canvas

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

HailoROIPtr build_roi_from_outputs(const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &outputs) {
     auto roi = std::make_shared<HailoROI>(HailoBBox(0.f,0.f,1.f,1.f));
     for (const auto &o : outputs) {
         auto tensor = std::make_shared<HailoTensor>(o.first, o.second);
         roi->add_tensor(tensor);
     }
     return roi;
 }


static inline float iou_calc(const HailoBBox &a, const HailoBBox &b)
{
    const float x1 = std::max(a.xmin(), b.xmin());
    const float y1 = std::max(a.ymin(), b.ymin());
    const float x2 = std::min(a.xmax(), b.xmax());
    const float y2 = std::min(a.ymax(), b.ymax());
    const float w = std::max(0.0f, x2 - x1);
    const float h = std::max(0.0f, y2 - y1);
    const float inter = w * h;
    const float areaA = (a.xmax() - a.xmin()) * (a.ymax() - a.ymin());
    const float areaB = (b.xmax() - b.xmin()) * (b.ymax() - b.ymin());
    const float denom = areaA + areaB - inter;
    return (denom > 0.f) ? (inter / denom) : 0.f;
}

struct LetterboxParams {
    int model_w, model_h;
    int resized_w, resized_h;
    int pad_right, pad_bottom;
    float scale;
};

static inline LetterboxParams compute_letterbox(int org_w, int org_h, int model_w, int model_h)
{
    // factor = max(fh, fw), fh = org_h/model_h, fw = org_w/model_w
    // resized = round(org / factor), pad only right/bottom
    const float fh = static_cast<float>(org_h) / static_cast<float>(model_h);
    const float fw = static_cast<float>(org_w) / static_cast<float>(model_w);
    const float factor = std::max(fh, fw); // >= 1 for downscale; could be <1 for up-scale

    const int resized_w = static_cast<int>(std::round(static_cast<float>(org_w) / factor));
    const int resized_h = static_cast<int>(std::round(static_cast<float>(org_h) / factor));

    LetterboxParams p;
    p.model_w = model_w;
    p.model_h = model_h;
    p.resized_w = std::min(model_w, resized_w);
    p.resized_h = std::min(model_h, resized_h);
    p.pad_right  = model_w - p.resized_w;
    p.pad_bottom = model_h - p.resized_h;
    p.scale = factor;
    return p;
}

// Map XYWH (in model space) -> original image - normalized coords
static inline HailoBBox unletterbox_xywh_to_orig_norm(float cx_m, float cy_m, float w_m, float h_m,
                                                      const LetterboxParams &lb, int org_w, int org_h)
{
    // Convert to corner form in model space
    float x1_m = cx_m - 0.5f * w_m;
    float y1_m = cy_m - 0.5f * h_m;
    float x2_m = cx_m + 0.5f * w_m;
    float y2_m = cy_m + 0.5f * h_m;

    // Remove padding
    x1_m = std::max(0.0f, std::min<float>(x1_m, static_cast<float>(lb.resized_w)));
    y1_m = std::max(0.0f, std::min<float>(y1_m, static_cast<float>(lb.resized_h)));
    x2_m = std::max(0.0f, std::min<float>(x2_m, static_cast<float>(lb.resized_w)));
    y2_m = std::max(0.0f, std::min<float>(y2_m, static_cast<float>(lb.resized_h)));

    
    float x1_o = x1_m * lb.scale;
    float y1_o = y1_m * lb.scale;
    float x2_o = x2_m * lb.scale;
    float y2_o = y2_m * lb.scale;

    // Normalize to original
    float nx1 = std::clamp(x1_o / static_cast<float>(org_w), 0.f, 1.f);
    float ny1 = std::clamp(y1_o / static_cast<float>(org_h), 0.f, 1.f);
    float nx2 = std::clamp(x2_o / static_cast<float>(org_w), 0.f, 1.f);
    float ny2 = std::clamp(y2_o / static_cast<float>(org_h), 0.f, 1.f);

    return HailoBBox(nx1, ny1, nx2 - nx1, ny2 - ny1);
}



std::vector<std::pair<HailoDetection, xt::xarray<float>>>
nms_pairs(std::vector<std::pair<HailoDetection, xt::xarray<float>>> dets,
          float iou_thr, bool cross_class)
{
    std::sort(dets.begin(), dets.end(),
              [](auto &a, auto &b){ return a.first.get_confidence() > b.first.get_confidence(); });

    std::vector<std::pair<HailoDetection, xt::xarray<float>>> kept;
    std::vector<bool> removed(dets.size(), false);

    for (size_t i = 0; i < dets.size(); ++i) {
        if (removed[i]) continue;
        kept.push_back(dets[i]);
        HailoDetection di = dets[i].first;
        const int cls_i = di.get_class_id();
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (removed[j]) continue;
            if (!cross_class) {
                HailoDetection dj = dets[j].first;
                if (cls_i != dj.get_class_id()) continue;
            }
            if (iou_calc(dets[i].first.get_bbox(), dets[j].first.get_bbox()) >= iou_thr) {
                removed[j] = true;
            }
        }
    }
    return kept;
}

// ---------- mask utilities ----------
static inline void sigmoid_inplace(cv::Mat &m) {
    const int rows = m.rows, cols = m.cols;
    for (int y=0;y<rows;++y) {
        float *p = m.ptr<float>(y);
        for (int x=0;x<cols;++x) p[x] = 1.f / (1.f + std::exp(-p[x]));
    }
}

// crop mask outside bbox (bbox is normalized in ORIGINAL frame space)
static cv::Mat crop_mask_norm(cv::Mat m, const HailoBBox &bb) {
    const int H = m.rows, W = m.cols;
    const int x1 = std::max(0, std::min(W, int(std::floor(bb.xmin() * W))));
    const int y1 = std::max(0, std::min(H, int(std::floor(bb.ymin() * H))));
    const int x2 = std::max(0, std::min(W, int(std::ceil ((bb.xmax()) * W))));
    const int y2 = std::max(0, std::min(H, int(std::ceil ((bb.ymax()) * H))));

    if (y1 > 0) m(cv::Rect(0, 0, W, y1)) = 0.0f;
    if (y2 < H) m(cv::Rect(0, y2, W, H - y2)) = 0.0f;
    if (x1 > 0) m(cv::Rect(0, 0, x1, H)) = 0.0f;
    if (x2 < W) m(cv::Rect(x2, 0, W - x2, H)) = 0.0f;
    return m;
}

// --- crop to the unpadded (resized) region ---
static inline cv::Mat crop_to_unpadded(const cv::Mat &m_model, const LetterboxParams &lb)
{
    const int W = m_model.cols;
    const int H = m_model.rows;
    const int cw = std::min(W, lb.resized_w);
    const int ch = std::min(H, lb.resized_h);
    return m_model(cv::Rect(0, 0, cw, ch)).clone();
}

std::vector<DetectionAndMask>
decode_masks(const std::vector<std::pair<HailoDetection, xt::xarray<float>>> &kept,
             const xt::xarray<float> &proto,
             int org_h, int org_w,
             int model_h, int model_w,
             int proto_channels)
{
    // reshape proto HWC -> (H*W, proto_channels) matrix for fast coeff * proto
    const int H = (int)proto.shape(0);
    const int W = (int)proto.shape(1);
    const int C = (int)proto.shape(2);
    CV_Assert(H > 0 && W > 0 && C == proto_channels);

    cv::Mat proto_hw_c(H*W, proto_channels, CV_32F);
    for (int c=0;c<proto_channels;++c) {
        int idx = 0;
        for (int y=0;y<H;++y)
            for (int x=0;x<W;++x)
                proto_hw_c.at<float>(idx++, c) = proto(y,x,c);
    }

    // Compute letterbox params (must match preprocess)
    const LetterboxParams lb = compute_letterbox(org_w, org_h, model_w, model_h);

    std::vector<DetectionAndMask> out;
    out.reserve(kept.size());

    for (auto pm : kept) { // copy pair so HailoDetection is non-const
        HailoDetection det = pm.first;

        // coeffs (1,proto_channels)
        cv::Mat coeff(1, proto_channels, CV_32F);
        for (int i=0;i<proto_channels;++i) coeff.at<float>(0,i) = pm.second(i);

        // (1,proto_channels) * (proto_channels, H*W) -> (1, H*W)
        cv::Mat mask_lin = coeff * proto_hw_c.t();
        cv::Mat mask_proto(H, W, CV_32F);
        std::memcpy(mask_proto.data, mask_lin.ptr<float>(0), size_t(H)*W*sizeof(float));

        sigmoid_inplace(mask_proto);

        cv::Mat mask_model;
        cv::resize(mask_proto, mask_model, cv::Size(model_w, model_h), 0, 0, cv::INTER_LINEAR);
        cv::Mat mask_unpadded = crop_to_unpadded(mask_model, lb);

        // scale unpadded region back to original size
        cv::Mat mask_big;
        cv::resize(mask_unpadded, mask_big, cv::Size(org_w, org_h), 0, 0, cv::INTER_LINEAR);

        // crop by the detection bbox
        HailoBBox bb = det.get_bbox();
        mask_big = crop_mask_norm(mask_big, bb);

        out.push_back(DetectionAndMask{ det, mask_big });
    }
    return out;
}

void draw_masks_and_boxes(cv::Mat &frame,
                          const std::vector<HailoDetection> &dets,
                          const std::vector<cv::Mat> &masks,
                          float alpha, float thresh)
{
    CV_Assert((int)dets.size() == (int)masks.size());
    cv::Mat overlay(frame.size(), CV_8UC3, cv::Scalar(0,0,0));

    const size_t palette = PALETTE.size();

    for (size_t i=0;i<dets.size();++i) {
        HailoDetection d = dets[i];
        cv::Mat mask = masks[i];
        if (mask.size() != frame.size())
            cv::resize(mask, mask, frame.size(), 0, 0, cv::INTER_LINEAR);

        int cid = std::max(0, d.get_class_id());
        cv::Scalar color = (palette > 0) ? PALETTE[cid % palette] : cv::Scalar(0,255,0);

        for (int y=0;y<mask.rows;++y) {
            const float *mp = mask.ptr<float>(y);
            cv::Vec3b *op = overlay.ptr<cv::Vec3b>(y);
            for (int x=0;x<mask.cols;++x) {
                if (mp[x] > thresh) op[x] = cv::Vec3b((uchar)color[0], (uchar)color[1], (uchar)color[2]);
            }
        }

        HailoBBox bb = d.get_bbox();
        int x = int(bb.xmin() * frame.cols);
        int y = int(bb.ymin() * frame.rows);
        int w = int((bb.xmax() - bb.xmin()) * frame.cols);
        int h = int((bb.ymax() - bb.ymin()) * frame.rows);
        cv::rectangle(frame, cv::Rect(x,y,w,h), color, 1);
    }

    cv::addWeighted(frame, 1.0, overlay, alpha, 0.0, frame);
}
