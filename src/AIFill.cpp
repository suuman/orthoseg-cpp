#include "AIFill.h"
#include <stdexcept>

namespace orthoseg {

Box validatedBox(Box b, cv::Size size) {
    if (size.width <= 0 || size.height <= 0)
        throw std::runtime_error("Load an image first.");
    if (!std::isfinite(b.x0) || !std::isfinite(b.y0) ||
        !std::isfinite(b.x1) || !std::isfinite(b.y1))
        throw std::runtime_error("Invalid bounding-box coordinates.");
    if (b.x0 > b.x1) std::swap(b.x0, b.x1);
    if (b.y0 > b.y1) std::swap(b.y0, b.y1);
    b.x0 = std::clamp(b.x0, 0.f, float(size.width - 1));
    b.x1 = std::clamp(b.x1, 0.f, float(size.width - 1));
    b.y0 = std::clamp(b.y0, 0.f, float(size.height - 1));
    b.y1 = std::clamp(b.y1, 0.f, float(size.height - 1));
    if (b.x0 >= b.x1 || b.y0 >= b.y1)
        throw std::runtime_error("Draw a non-empty bounding box inside the image.");
    return b;
}

cv::Mat binaryPromptMask(const cv::Mat& mask, cv::Size size) {
    if (mask.empty()) throw std::runtime_error("The prompt mask is empty or could not be decoded.");
    if (mask.dims != 2 || mask.size() != size)
        throw std::runtime_error("Prompt mask dimensions must match the displayed image; no resizing is performed.");
    if ((mask.depth() != CV_8U && mask.depth() != CV_16U) ||
        (mask.channels() != 1 && mask.channels() != 3 && mask.channels() != 4))
        throw std::runtime_error("Use an 8-bit or 16-bit grayscale, RGB, or RGBA prompt mask.");
    // Any nonzero color channel is foreground; alpha is not a label channel.
    std::vector<cv::Mat> channels;
    cv::split(mask, channels);
    cv::Mat binary = channels[0] != 0;
    for (int c = 1; c < std::min(3, mask.channels()); ++c)
        binary |= channels[c] != 0;
    if (cv::countNonZero(binary) == 0)
        throw std::runtime_error("The prompt mask has no foreground pixels.");
    return binary;
}

void validateAIResult(const cv::Mat& mask, cv::Size size, const std::vector<int>& allowedClasses) {
    if (mask.empty() || mask.dims != 2 || mask.type() != CV_8UC1 || mask.size() != size)
        throw std::runtime_error("MedSAM2 returned an incompatible result mask.");
    cv::Mat unexpected = (mask != 0);
    for (int c : allowedClasses) {
        unexpected &= (mask != c);
    }
    if (cv::countNonZero(unexpected) != 0)
        throw std::runtime_error("MedSAM2 returned unexpected anatomy labels.");
}

void validateAIResult(const cv::Mat& mask, cv::Size size, Label target) {
    validateAIResult(mask, size, std::vector<int>{static_cast<int>(target)});
}

} // namespace orthoseg
