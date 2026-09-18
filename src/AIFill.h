#pragma once
#include "Labels.h"
#include "image_utils.h"
#include <optional>

namespace orthoseg {

enum class AIFillPromptType { BoundingBox, PaintedMask, LoadedMask };

struct AIFillState {
    AIFillPromptType promptType = AIFillPromptType::BoundingBox;
    std::optional<Box> box;
    cv::Mat promptMask; // Source-resolution binary 0/255, separate from annotations.
    cv::Mat resultMask; // Source-resolution indexed labels.
    bool showResult = true;
};

// Uses ocv's inclusive pixel-center convention, bounded by width/height - 1.
Box validatedBox(Box box, cv::Size size);
cv::Mat binaryPromptMask(const cv::Mat& mask, cv::Size size);
void validateAIResult(const cv::Mat& mask, cv::Size size, Label target);

} // namespace orthoseg
