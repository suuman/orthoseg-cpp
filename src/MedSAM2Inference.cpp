#include "MedSAM2Inference.h"
#include <filesystem>
#include <stdexcept>

namespace orthoseg {

std::vector<Prompt> MedSAM2Inference::preparePrompts(const AIFillRequest& r) {
    if (r.imageBGR.empty()) throw std::runtime_error("Load an image first.");
    if (r.imageBGR.dims != 2 || r.imageBGR.type() != CV_8UC3)
        throw std::runtime_error("AI Fill requires the displayed 8-bit BGR source image.");

    std::vector<Prompt> prompts;

    if (r.type == AIFillPromptType::BoundingBox) {
        if (r.femurBox) {
            Prompt p;
            p.class_id = static_cast<int>(Label::Femur);
            p.box = validatedBox(*r.femurBox, r.imageBGR.size());
            prompts.push_back(p);
        }
        if (r.tibiaBox) {
            Prompt p;
            p.class_id = static_cast<int>(Label::Tibia);
            p.box = validatedBox(*r.tibiaBox, r.imageBGR.size());
            prompts.push_back(p);
        }
        if (prompts.empty()) {
            if (!r.box) throw std::runtime_error("Draw a bounding box first.");
            if (r.target != Label::Femur && r.target != Label::Tibia)
                throw std::runtime_error("Select Femur or Tibia under Select Anatomy for AI Fill.");
            Prompt p;
            p.class_id = static_cast<int>(r.target);
            p.box = validatedBox(*r.box, r.imageBGR.size());
            prompts.push_back(p);
        }
    } else if (r.type == AIFillPromptType::NormalFillMask) {
        if (r.promptMask.empty() || r.promptMask.size() != r.imageBGR.size())
            throw std::runtime_error("Normal fill mask is empty or does not match the image.");
        const auto pad = computePadInfo(r.imageBGR.cols, r.imageBGR.rows, 1024, true);

        Box fBox;
        if (boxFromMask(r.promptMask, static_cast<uint8_t>(Label::Femur), fBox)) {
            Prompt p;
            p.class_id = static_cast<int>(Label::Femur);
            p.box = fBox;
            p.has_mask = true;
            p.mask_logits = makeMaskLogits(r.promptMask, static_cast<uint8_t>(Label::Femur), pad);
            prompts.push_back(p);
        }

        Box tBox;
        if (boxFromMask(r.promptMask, static_cast<uint8_t>(Label::Tibia), tBox)) {
            Prompt p;
            p.class_id = static_cast<int>(Label::Tibia);
            p.box = tBox;
            p.has_mask = true;
            p.mask_logits = makeMaskLogits(r.promptMask, static_cast<uint8_t>(Label::Tibia), pad);
            prompts.push_back(p);
        }

        if (prompts.empty()) {
            Box b;
            if (boxFromMask(r.promptMask, 255, b)) {
                if (r.target != Label::Femur && r.target != Label::Tibia)
                    throw std::runtime_error("Select Femur or Tibia under Select Anatomy for AI Fill.");
                Prompt p;
                p.class_id = static_cast<int>(r.target);
                p.box = b;
                p.has_mask = true;
                p.mask_logits = makeMaskLogits(r.promptMask, 255, pad);
                prompts.push_back(p);
            }
        }

        if (prompts.empty()) {
            throw std::runtime_error("Normal fill mask contains no Femur or Tibia annotations.");
        }
    } else {
        const auto pad = computePadInfo(r.imageBGR.cols, r.imageBGR.rows, 1024, true);

        // Check if promptMask has multi-class labels (1 for Femur, 2 for Tibia) - e.g. from previous AI result
        Box fBox, tBox;
        bool hasFemur = boxFromMask(r.promptMask, static_cast<uint8_t>(Label::Femur), fBox);
        bool hasTibia = boxFromMask(r.promptMask, static_cast<uint8_t>(Label::Tibia), tBox);

        if (hasFemur || hasTibia) {
            if (hasFemur) {
                Prompt p;
                p.class_id = static_cast<int>(Label::Femur);
                p.box = fBox;
                p.has_mask = true;
                p.mask_logits = makeMaskLogits(r.promptMask, static_cast<uint8_t>(Label::Femur), pad);
                prompts.push_back(p);
            }
            if (hasTibia) {
                Prompt p;
                p.class_id = static_cast<int>(Label::Tibia);
                p.box = tBox;
                p.has_mask = true;
                p.mask_logits = makeMaskLogits(r.promptMask, static_cast<uint8_t>(Label::Tibia), pad);
                prompts.push_back(p);
            }
        } else {
            if (r.target != Label::Femur && r.target != Label::Tibia)
                throw std::runtime_error("Select Femur or Tibia under Select Anatomy for AI Fill.");
            Prompt p;
            p.class_id = static_cast<int>(r.target);
            cv::Mat binary = binaryPromptMask(r.promptMask, r.imageBGR.size());
            boxFromMask(binary, 255, p.box);
            p.has_mask = true;
            p.mask_logits = makeMaskLogits(binary, 255, pad);
            prompts.push_back(p);
        }
    }
    return prompts;
}

Prompt MedSAM2Inference::preparePrompt(const AIFillRequest& r) {
    auto prompts = preparePrompts(r);
    if (prompts.empty()) throw std::runtime_error("No prompt available.");
    return prompts.front();
}

cv::Mat MedSAM2Inference::run(const AIFillRequest& r) {
    const auto prompts = preparePrompts(r);
    if (!engine_ || modelDirectory_ != r.modelDirectory) {
        const auto dir = std::filesystem::path(r.modelDirectory);
        const auto encoder = dir / "medsam2_image_encoder.onnx";
        const auto decoder = dir / "medsam2_mask_decoder.onnx";
        if (!std::filesystem::is_regular_file(encoder) || !std::filesystem::is_regular_file(decoder))
            throw std::runtime_error("Model directory must contain medsam2_image_encoder.onnx and medsam2_mask_decoder.onnx.");
        const auto targets = cv::dnn::getAvailableTargets(cv::dnn::DNN_BACKEND_CUDA);
        if (cv::cuda::getCudaEnabledDeviceCount() <= 0 ||
            std::find(targets.begin(), targets.end(), cv::dnn::DNN_TARGET_CUDA) == targets.end())
            throw std::runtime_error("CUDA is unavailable. AI Fill requires an NVIDIA driver/device and OpenCV DNN built with CUDA and cuDNN.");
        cv::cuda::setDevice(0);
        // Release the previous models before changing directories to bound GPU use.
        engine_.reset();
        engine_ = std::make_unique<MedSAM2Engine>(encoder.string(), decoder.string(), "cuda");
        modelDirectory_ = r.modelDirectory;
    }
    cv::Mat rgb;
    cv::cvtColor(r.imageBGR, rgb, cv::COLOR_BGR2RGB);
    try {
        auto result = engine_->segment(rgb, prompts, true, 1024, 0.0f);
        std::vector<int> allowedClasses;
        for (const auto& p : prompts) allowedClasses.push_back(p.class_id);
        validateAIResult(result.label_map, r.imageBGR.size(), allowedClasses);
        return result.label_map;
    } catch (...) {
        // A failed forward (including GPU OOM) may leave DNN buffers unusable.
        engine_.reset();
        throw;
    }
}

} // namespace orthoseg
