#pragma once
#include "Labels.h"
#include "image_utils.h"
#include <optional>

#include <vector>

namespace orthoseg {

enum class AIFillPromptType { BoundingBox, PaintedMask, LoadedMask, NormalFillMask };

struct AIFillState {
    AIFillPromptType promptType = AIFillPromptType::BoundingBox;
    std::optional<Box> box;       // Active/most-recent box (maintained for backward compatibility & tests)
    std::optional<Box> femurBox;  // Bounding box for Femur (class 1, red)
    std::optional<Box> tibiaBox;  // Bounding box for Tibia (class 2, green)
    std::optional<Box> femurBox2; // Second Femur box for bilateral MONAI SAM2
    std::optional<Box> tibiaBox2; // Second Tibia box for bilateral MONAI SAM2
    int activeBoxNumber = 1;     // Which box receives the next canvas drag
    bool showSecondaryBoxes = false;
    cv::Mat promptMask; // Source-resolution binary 0/255, separate from annotations.
    cv::Mat resultMask; // Source-resolution indexed labels.
    bool resultReplacesAnatomy = false; // Automatic models also clear old Femur/Tibia pixels.
    bool showResult = true;
    bool showPrompt = true;
};

// Uses ocv's inclusive pixel-center convention, bounded by width/height - 1.
Box validatedBox(Box box, cv::Size size);
cv::Mat binaryPromptMask(const cv::Mat& mask, cv::Size size);
void validateAIResult(const cv::Mat& mask, cv::Size size, Label target);
void validateAIResult(const cv::Mat& mask, cv::Size size, const std::vector<int>& allowedClasses);

} // namespace orthoseg
