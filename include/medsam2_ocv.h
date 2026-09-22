#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/cuda.hpp>

#include <string>
#include <vector>
#include <map>
#include <memory>

#include "image_utils.h"

struct Prompt {
    int class_id = 1;
    Box box;                  // Unpadded bounding box in original image coordinates
    bool has_mask = false;
    cv::Mat mask_logits;      // Optional (1, 1, 256, 256) CV_32F prompt logit map
};

struct SegmentationResult {
    int width = 0;
    int height = 0;
    cv::Mat label_map;               // Final 1-channel mask with class IDs (CV_8UC1)
    std::map<int, float> iou_scores; // Predicted IoU per class_id
};

class MedSAM2Engine {
public:
    MedSAM2Engine(
        const std::string& encoder_path,
        const std::string& decoder_path,
        const std::string& device_str = "cuda"
    );

    // Segment image with provided prompts (matches medsam2_infer_custom.py)
    SegmentationResult segment(
        const cv::Mat& image_rgb,
        const std::vector<Prompt>& prompts,
        bool pad_to_square = true,
        int pad_size = 1024,
        float score_thresh = 0.0f
    );

    std::string getDevice() const { return device_str_; }

private:
    cv::dnn::Net encoder_;
    cv::dnn::Net decoder_;
    std::string device_str_;
    int backend_id_;
    int target_id_;
};
