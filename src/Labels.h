#pragma once
#include <opencv2/core.hpp>
#include <array>
#include <string>

namespace orthoseg {

// Label IDs match the design spec. Background acts as an eraser.
enum class Label : int {
    Background = 0,
    Femur     = 1,
    Tibia     = 2,
    Fibula    = 3,
};

enum class Tool { Brush, Fill, Eraser };

// The first three grow a single region from one seed click. The last three are
// multi-seed competition methods: the user scribbles seeds for several labels
// (including background) and the labels compete for the ambiguous pixels.
enum class FillAlgorithm {
    Standard,       // single-click region grow
    EdgeEmbedded,   // single-click region grow, halts at edges
    SplitMerge,     // single-click block union-find
    GrowCut,        // scribble: cellular-automata label competition
    RandomWalker,   // scribble: harmonic probability field, argmax
    GraphCut,       // scribble: grabCut (GMM + min-cut/max-flow), fg vs bg
};

// True for the scribble/seed-competition algorithms (GrowCut, RandomWalker,
// GraphCut), which need a seed layer and an explicit Run step rather than a
// single seed click.
inline bool isScribbleAlgorithm(FillAlgorithm a) {
    return a == FillAlgorithm::GrowCut ||
           a == FillAlgorithm::RandomWalker ||
           a == FillAlgorithm::GraphCut;
}

struct LabelInfo {
    Label       id;
    const char* name;
    // BGR because OpenCV stores color in BGR order.
    cv::Vec3b   colorBGR;
};

// Femur #ef4444, Tibia #22c55e, Fibula #3b82f6 (stored BGR).
inline const std::array<LabelInfo, 4>& labels() {
    static const std::array<LabelInfo, 4> kLabels = {{
        {Label::Background, "Background", cv::Vec3b(0, 0, 0)},
        {Label::Femur,      "Femur",      cv::Vec3b(0x44, 0x44, 0xef)},
        {Label::Tibia,      "Tibia",      cv::Vec3b(0x5e, 0xc5, 0x22)},
        {Label::Fibula,     "Fibula",     cv::Vec3b(0xf6, 0x82, 0x3b)},
    }};
    return kLabels;
}

inline const LabelInfo& labelInfo(Label l) {
    return labels()[static_cast<int>(l)];
}

} // namespace orthoseg
