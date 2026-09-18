#include "MedSAM2Inference.h"
#include <filesystem>
#include <stdexcept>

namespace orthoseg {

Prompt MedSAM2Inference::preparePrompt(const AIFillRequest& r) {
    if (r.imageBGR.empty()) throw std::runtime_error("Load an image first.");
    if (r.imageBGR.dims != 2 || r.imageBGR.type() != CV_8UC3)
        throw std::runtime_error("AI Fill requires the displayed 8-bit BGR source image.");
    if (r.target != Label::Femur && r.target != Label::Tibia)
        throw std::runtime_error("Select Femur or Tibia under Select Anatomy for AI Fill.");
    Prompt p;
    p.class_id = static_cast<int>(r.target);
    if (r.type == AIFillPromptType::BoundingBox) {
        if (!r.box) throw std::runtime_error("Draw a bounding box first.");
        p.box = validatedBox(*r.box, r.imageBGR.size());
    } else {
        cv::Mat binary = binaryPromptMask(r.promptMask, r.imageBGR.size());
        boxFromMask(binary, 255, p.box);
        p.has_mask = true;
        p.mask_logits = makeMaskLogits(binary, 255,
            computePadInfo(r.imageBGR.cols, r.imageBGR.rows, 1024, true));
    }
    return p;
}

cv::Mat MedSAM2Inference::run(const AIFillRequest& r) {
    const Prompt prompt = preparePrompt(r);
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
        auto result = engine_->segment(rgb, {prompt}, true, 1024, 0.0f);
        validateAIResult(result.label_map, r.imageBGR.size(), r.target);
        return result.label_map;
    } catch (...) {
        // A failed forward (including GPU OOM) may leave DNN buffers unusable.
        engine_.reset();
        throw;
    }
}

} // namespace orthoseg
