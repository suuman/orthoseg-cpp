#pragma once
#include "AIFill.h"
#include "medsam2_ocv.h"
#include <memory>

namespace orthoseg {

struct AIFillRequest {
    cv::Mat imageBGR; // Immutable Document source, retained by reference counting.
    Label target = Label::Femur;
    AIFillPromptType type = AIFillPromptType::BoundingBox;
    std::optional<Box> box;
    cv::Mat promptMask; // Owned snapshot: UI painting must not mutate this data.
    std::string modelDirectory;
};

// Owned and used exclusively by the persistent inference worker thread.
class MedSAM2Inference {
public:
    static Prompt preparePrompt(const AIFillRequest& request);
    cv::Mat run(const AIFillRequest& request);
private:
    std::unique_ptr<MedSAM2Engine> engine_;
    std::string modelDirectory_;
};

} // namespace orthoseg
