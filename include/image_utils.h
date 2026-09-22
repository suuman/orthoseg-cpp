#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <string>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>

struct Box {
    float x0 = 0.0f;
    float y0 = 0.0f;
    float x1 = 0.0f;
    float y1 = 0.0f;
};

// Generic geometry transformation information between original image and network canvas
struct PadInfo {
    int target_size = 1024; // Network input canvas size (default 1024)
    int orig_w = 0;         // Original image width (any size)
    int orig_h = 0;         // Original image height (any size)
    int scaled_w = 0;       // Scaled width on canvas
    int scaled_h = 0;       // Scaled height on canvas
    int x0 = 0;             // Canvas offset X
    int y0 = 0;             // Canvas offset Y
    float scale_x = 1.0f;   // scaled_w / orig_w
    float scale_y = 1.0f;   // scaled_h / orig_h
    bool pad_to_square = true;
};

inline PadInfo computePadInfo(int orig_w, int orig_h, int target_size = 1024, bool pad_to_square = true) {
    PadInfo pad;
    pad.target_size = target_size;
    pad.orig_w = orig_w;
    pad.orig_h = orig_h;
    pad.pad_to_square = pad_to_square;

    if (pad_to_square) {
        float scale = static_cast<float>(target_size) / static_cast<float>(std::max(orig_w, orig_h));
        pad.scaled_w = std::clamp(static_cast<int>(std::round(orig_w * scale)), 1, target_size);
        pad.scaled_h = std::clamp(static_cast<int>(std::round(orig_h * scale)), 1, target_size);
        pad.x0 = (target_size - pad.scaled_w) / 2;
        pad.y0 = (target_size - pad.scaled_h) / 2;
    } else {
        pad.scaled_w = target_size;
        pad.scaled_h = target_size;
        pad.x0 = 0;
        pad.y0 = 0;
    }
    pad.scale_x = static_cast<float>(pad.scaled_w) / static_cast<float>(orig_w);
    pad.scale_y = static_cast<float>(pad.scaled_h) / static_cast<float>(orig_h);
    return pad;
}

// Load image of any size, bit-depth (8/16-bit), and channels, returning 3-channel RGB CV_8UC3
inline cv::Mat loadImageRGB(const std::string& path) {
    cv::Mat raw = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (raw.empty()) {
        std::cerr << "[OpenCV5] Error: Failed to load image from " << path << std::endl;
        return cv::Mat();
    }

    cv::Mat raw8;
    if (raw.depth() == CV_16U) {
        double min_v = 0.0, max_v = 65535.0;
        cv::minMaxLoc(raw, &min_v, &max_v);
        double range = (max_v > min_v) ? (max_v - min_v) : 1.0;
        raw.convertTo(raw8, CV_8U, 255.0 / range, -min_v * 255.0 / range);
    } else if (raw.depth() != CV_8U) {
        raw.convertTo(raw8, CV_8U);
    } else {
        raw8 = raw;
    }

    cv::Mat rgb;
    if (raw8.channels() == 1) {
        cv::cvtColor(raw8, rgb, cv::COLOR_GRAY2RGB);
    } else if (raw8.channels() == 3) {
        cv::cvtColor(raw8, rgb, cv::COLOR_BGR2RGB);
    } else if (raw8.channels() == 4) {
        cv::cvtColor(raw8, rgb, cv::COLOR_BGRA2RGB);
    } else {
        std::cerr << "[OpenCV5] Unsupported channel count: " << raw8.channels() << std::endl;
        return cv::Mat();
    }

    return rgb;
}

// Load 1-channel label PNG of any size
inline cv::Mat loadLabelPNG(const std::string& path) {
    cv::Mat mask = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (mask.empty()) {
        std::cerr << "[OpenCV5] Error loading label mask from: " << path << std::endl;
    }
    return mask;
}

// Map any image size onto target square canvas (target_size x target_size)
inline cv::Mat padImageToSquare(const cv::Mat& img, const PadInfo& pad) {
    cv::Mat canvas = cv::Mat::zeros(pad.target_size, pad.target_size, img.type());
    cv::Mat resized;
    if (img.cols == pad.scaled_w && img.rows == pad.scaled_h) {
        resized = img;
    } else {
        cv::resize(img, resized, cv::Size(pad.scaled_w, pad.scaled_h), 0, 0, cv::INTER_LINEAR);
    }
    resized.copyTo(canvas(cv::Rect(pad.x0, pad.y0, pad.scaled_w, pad.scaled_h)));
    return canvas;
}

// Unpad canvas mask back to original image dimensions (orig_w x orig_h)
inline cv::Mat unpadMask(const cv::Mat& padded_mask, const PadInfo& pad) {
    cv::Mat roi = padded_mask(cv::Rect(pad.x0, pad.y0, pad.scaled_w, pad.scaled_h));
    if (roi.cols == pad.orig_w && roi.rows == pad.orig_h) {
        return roi.clone();
    }
    cv::Mat unpadded;
    cv::resize(roi, unpadded, cv::Size(pad.orig_w, pad.orig_h), 0, 0, cv::INTER_NEAREST);
    return unpadded;
}

// Extract tight bounding box around a specific class_id in mask of any size
inline bool boxFromMask(
    const cv::Mat& mask,
    uint8_t class_id,
    Box& out_box,
    int pad_px = 0
) {
    if (mask.empty() || mask.channels() != 1) return false;

    cv::Mat pts;
    cv::findNonZero(mask == class_id, pts);
    if (pts.empty()) return false;

    cv::Rect r = cv::boundingRect(pts);
    out_box.x0 = static_cast<float>(std::max(0, r.x - pad_px));
    out_box.y0 = static_cast<float>(std::max(0, r.y - pad_px));
    out_box.x1 = static_cast<float>(std::min(mask.cols - 1, r.x + r.width - 1 + pad_px));
    out_box.y1 = static_cast<float>(std::min(mask.rows - 1, r.y + r.height - 1 + pad_px));
    return true;
}

// Convert binary mask of any size to 256x256 logits tensor [1, 1, 256, 256] mapped to canvas geometry
inline cv::Mat makeMaskLogits(
    const cv::Mat& mask_img,
    uint8_t class_id,
    const PadInfo& pad
) {
    int S = pad.target_size;
    cv::Mat padded = cv::Mat::zeros(S, S, CV_32FC1);

    cv::Mat mask_c = (mask_img == class_id);
    cv::Mat mask_scaled;
    if (mask_c.cols == pad.scaled_w && mask_c.rows == pad.scaled_h) {
        mask_c.convertTo(mask_scaled, CV_32FC1, 1.0 / 255.0);
    } else {
        cv::Mat mask_c_f;
        mask_c.convertTo(mask_c_f, CV_32FC1, 1.0 / 255.0);
        cv::resize(mask_c_f, mask_scaled, cv::Size(pad.scaled_w, pad.scaled_h), 0, 0, cv::INTER_NEAREST);
    }
    mask_scaled.copyTo(padded(cv::Rect(pad.x0, pad.y0, pad.scaled_w, pad.scaled_h)));

    cv::Mat low_res;
    cv::resize(padded, low_res, cv::Size(256, 256), 0, 0, cv::INTER_LINEAR);

    int sz[] = {1, 1, 256, 256};
    cv::Mat logits(4, sz, CV_32F);
    float* out_ptr = logits.ptr<float>();
    const float* in_ptr = low_res.ptr<float>();
    for (int i = 0; i < 256 * 256; ++i) {
        out_ptr[i] = (in_ptr[i] > 0.5f) ? 10.0f : -10.0f;
    }

    return logits;
}

// Save 1-channel mask PNG
inline bool saveMaskPNG(const std::string& path, const cv::Mat& mask) {
    return cv::imwrite(path, mask);
}

// Generate color overlay visualization PNG for any image size
inline bool saveOverlayPNG(
    const std::string& path,
    const cv::Mat& image_rgb,
    const cv::Mat& mask,
    float alpha = 0.45f
) {
    if (image_rgb.empty() || mask.empty() || image_rgb.size() != mask.size()) {
        std::cerr << "[saveOverlayPNG] Dimension mismatch: image " << image_rgb.cols << "x" << image_rgb.rows
                  << " vs mask " << mask.cols << "x" << mask.rows << std::endl;
        return false;
    }

    struct Color { uint8_t r, g, b; };
    const std::map<uint8_t, Color> default_colors = {
        {1, {255, 0, 0}},    // Femur = Red
        {2, {0, 255, 0}},    // Tibia = Green
        {3, {0, 0, 255}},    // Blue
        {4, {255, 255, 0}},  // Yellow
        {5, {255, 0, 255}},  // Magenta
        {6, {0, 255, 255}},  // Cyan
    };

    auto getColor = [&](uint8_t lbl) -> Color {
        auto it = default_colors.find(lbl);
        if (it != default_colors.end()) return it->second;
        // Deterministic distinct colors for arbitrary class IDs
        uint8_t r = static_cast<uint8_t>((lbl * 67 + 100) % 256);
        uint8_t g = static_cast<uint8_t>((lbl * 131 + 50) % 256);
        uint8_t b = static_cast<uint8_t>((lbl * 197 + 150) % 256);
        return {r, g, b};
    };

    cv::Mat overlay_bgr(image_rgb.rows, image_rgb.cols, CV_8UC3);
    for (int r = 0; r < image_rgb.rows; ++r) {
        const cv::Vec3b* img_row = image_rgb.ptr<cv::Vec3b>(r);
        const uint8_t* mask_row = mask.ptr<uint8_t>(r);
        cv::Vec3b* out_row = overlay_bgr.ptr<cv::Vec3b>(r);
        for (int c = 0; c < image_rgb.cols; ++c) {
            uint8_t lbl = mask_row[c];
            float red = img_row[c][0];
            float grn = img_row[c][1];
            float blu = img_row[c][2];

            if (lbl > 0) {
                Color col = getColor(lbl);
                red = (1.0f - alpha) * red + alpha * col.r;
                grn = (1.0f - alpha) * grn + alpha * col.g;
                blu = (1.0f - alpha) * blu + alpha * col.b;
            }

            out_row[c][0] = static_cast<uint8_t>(std::clamp(blu, 0.0f, 255.0f)); // OpenCV BGR order
            out_row[c][1] = static_cast<uint8_t>(std::clamp(grn, 0.0f, 255.0f));
            out_row[c][2] = static_cast<uint8_t>(std::clamp(red, 0.0f, 255.0f));
        }
    }

    return cv::imwrite(path, overlay_bgr);
}

// Compute Dice metric for any image size
inline float computeDice(const cv::Mat& pred, const cv::Mat& gt, uint8_t class_id) {
    if (pred.empty() || gt.empty() || pred.size() != gt.size()) return 0.0f;

    int inter = 0, pred_cnt = 0, gt_cnt = 0;
    for (int r = 0; r < pred.rows; ++r) {
        const uint8_t* p = pred.ptr<uint8_t>(r);
        const uint8_t* g = gt.ptr<uint8_t>(r);
        for (int c = 0; c < pred.cols; ++c) {
            bool pb = (p[c] == class_id);
            bool gb = (g[c] == class_id);
            if (pb && gb) inter++;
            if (pb) pred_cnt++;
            if (gb) gt_cnt++;
        }
    }
    int denom = pred_cnt + gt_cnt;
    if (denom == 0) return 1.0f;
    return 2.0f * static_cast<float>(inter) / static_cast<float>(denom);
}

// Compute IoU metric for any image size
inline float computeIoU(const cv::Mat& pred, const cv::Mat& gt, uint8_t class_id) {
    if (pred.empty() || gt.empty() || pred.size() != gt.size()) return 0.0f;

    int inter = 0, union_cnt = 0;
    for (int r = 0; r < pred.rows; ++r) {
        const uint8_t* p = pred.ptr<uint8_t>(r);
        const uint8_t* g = gt.ptr<uint8_t>(r);
        for (int c = 0; c < pred.cols; ++c) {
            bool pb = (p[c] == class_id);
            bool gb = (g[c] == class_id);
            if (pb && gb) inter++;
            if (pb || gb) union_cnt++;
        }
    }
    if (union_cnt == 0) return 1.0f;
    return static_cast<float>(inter) / static_cast<float>(union_cnt);
}
