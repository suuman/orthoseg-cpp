#pragma once
#include "AIFill.h"
#include "medsam2_ocv.h"
#include <memory>
#include <vector>

namespace orthoseg {

struct AIFillRequest {
    cv::Mat imageBGR; // Immutable Document source, retained by reference counting.
    Label target = Label::Femur;
    AIFillPromptType type = AIFillPromptType::BoundingBox;
    std::optional<Box> box;       // Legacy / single box fallback
    std::optional<Box> femurBox;  // Bounding box for Femur (class 1, red)
    std::optional<Box> tibiaBox;  // Bounding box for Tibia (class 2, green)
    cv::Mat promptMask; // Owned snapshot: UI painting must not mutate this data.
    std::string modelDirectory;
};

// Owned and used exclusively by the persistent inference worker thread.
class MedSAM2Inference {
public:
    static Prompt preparePrompt(const AIFillRequest& request);
    static std::vector<Prompt> preparePrompts(const AIFillRequest& request);
    cv::Mat run(const AIFillRequest& request);
private:
    std::unique_ptr<MedSAM2Engine> engine_;
    std::string modelDirectory_;
};

} // namespace orthoseg
