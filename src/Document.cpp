#include "Document.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <cmath>

namespace orthoseg {

bool Document::loadImage(const std::string& path) {
    cv::Mat color = cv::imread(path, cv::IMREAD_COLOR);
    if (color.empty()) return false;

    sourceColor_ = color;
    // Plain channel average (R+G+B)/3, matching the reference implementation
    // (canvasUtils.ts) rather than luma-weighted cvtColor grayscale.
    cv::transform(color, sourceGray_, cv::Matx13f(1.f / 3, 1.f / 3, 1.f / 3));
    edgeMap_ = computeEdgeMap(sourceGray_);
    mask_ = cv::Mat::zeros(sourceGray_.size(), CV_8UC1);
    seeds_ = cv::Mat(sourceGray_.size(), CV_8UC1, cv::Scalar(kNoSeed));

    // Empty history: callers snapshot the current state before each mutation,
    // so there is nothing to undo back to until the first edit.
    history_.clear();
    return true;
}

bool Document::exportMask(const std::string& path) const {
    if (mask_.empty()) return false;
    // Render the indexed mask to a BGR image using label colors.
    cv::Mat out(mask_.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    for (int y = 0; y < mask_.rows; ++y) {
        const uchar* mrow = mask_.ptr<uchar>(y);
        cv::Vec3b* orow = out.ptr<cv::Vec3b>(y);
        for (int x = 0; x < mask_.cols; ++x) {
            if (mrow[x] != 0)
                orow[x] = labelInfo(static_cast<Label>(mrow[x])).colorBGR;
        }
    }
    return cv::imwrite(path, out);
}

void Document::paintLine(cv::Point a, cv::Point b, Label label, int brushSize) {
    if (mask_.empty()) return;
    const uchar id = static_cast<uchar>(label);
    cv::line(mask_, a, b, cv::Scalar(id), brushSize, cv::LINE_8);
    // Round caps so consecutive segments join smoothly.
    int r = std::max(1, brushSize / 2);
    cv::circle(mask_, a, r, cv::Scalar(id), cv::FILLED);
    cv::circle(mask_, b, r, cv::Scalar(id), cv::FILLED);
}

void Document::fill(cv::Point seed, Label label, FillAlgorithm algo,
                    int intensityThreshold, int edgePenaltyThreshold) {
    if (mask_.empty()) return;
    switch (algo) {
        case FillAlgorithm::Standard:
            regionGrowStandard(sourceGray_, mask_, seed, label,
                               intensityThreshold);
            break;
        case FillAlgorithm::EdgeEmbedded:
            regionGrowEdgeEmbedded(sourceGray_, mask_, seed, label,
                                   intensityThreshold, edgePenaltyThreshold,
                                   edgeMap_);
            break;
        case FillAlgorithm::SplitMerge:
            regionGrowSplitMerge(sourceGray_, mask_, seed, label,
                                 intensityThreshold, edgePenaltyThreshold,
                                 4, edgeMap_);
            break;
    }
}

void Document::clearMask() {
    if (mask_.empty()) return;
    mask_.setTo(cv::Scalar(0));
}

void Document::paintSeedLine(cv::Point a, cv::Point b, Label label, int brushSize) {
    if (seeds_.empty()) return;
    const uchar id = static_cast<uchar>(label); // Background (0) is a real seed
    cv::line(seeds_, a, b, cv::Scalar(id), brushSize, cv::LINE_8);
    int r = std::max(1, brushSize / 2);
    cv::circle(seeds_, a, r, cv::Scalar(id), cv::FILLED);
    cv::circle(seeds_, b, r, cv::Scalar(id), cv::FILLED);
}

void Document::clearSeeds() {
    if (seeds_.empty()) return;
    seeds_.setTo(cv::Scalar(kNoSeed));
}

bool Document::hasSeeds() const {
    if (seeds_.empty()) return false;
    return cv::countNonZero(seeds_ != kNoSeed) > 0;
}

bool Document::hasSeedForLabel(Label label) const {
    if (seeds_.empty()) return false;
    return cv::countNonZero(seeds_ == static_cast<uchar>(label)) > 0;
}

int Document::seedLabelCount() const {
    if (seeds_.empty()) return 0;
    int count = 0;
    for (int l = 0; l < 4; ++l)
        if (cv::countNonZero(seeds_ == static_cast<uchar>(l)) > 0) ++count;
    return count;
}

// Downscale a seed layer while preserving every scribble: stamp each seeded
// pixel into its target cell (a plain resize would drop thin strokes).
static cv::Mat downscaleSeeds(const cv::Mat& seeds, cv::Size dst) {
    cv::Mat out(dst, CV_8UC1, cv::Scalar(kNoSeed));
    double sx = static_cast<double>(dst.width)  / seeds.cols;
    double sy = static_cast<double>(dst.height) / seeds.rows;
    for (int y = 0; y < seeds.rows; ++y) {
        const uchar* s = seeds.ptr<uchar>(y);
        for (int x = 0; x < seeds.cols; ++x) {
            if (s[x] == kNoSeed) continue;
            int ox = std::min(dst.width - 1,  static_cast<int>(x * sx));
            int oy = std::min(dst.height - 1, static_cast<int>(y * sy));
            out.at<uchar>(oy, ox) = s[x];
        }
    }
    return out;
}

bool Document::runSeedSegmentation(FillAlgorithm algo, Label foreground,
                                   double beta) {
    if (sourceGray_.empty() || !hasSeeds()) return false;
    if (!isScribbleAlgorithm(algo)) return false;

    // Cap the working resolution so the iterative solvers stay interactive.
    constexpr int kMaxWorkDim = 512;
    const int w = width(), h = height();
    const double scale = std::min(1.0, static_cast<double>(kMaxWorkDim) /
                                       std::max(w, h));
    const bool down = scale < 1.0;
    cv::Size ws(std::max(1, static_cast<int>(std::lround(w * scale))),
                std::max(1, static_cast<int>(std::lround(h * scale))));

    cv::Mat wgray, wcolor, wseeds, wout;
    if (down) {
        cv::resize(sourceGray_,  wgray,  ws, 0, 0, cv::INTER_AREA);
        cv::resize(sourceColor_, wcolor, ws, 0, 0, cv::INTER_AREA);
        wseeds = downscaleSeeds(seeds_, ws);
        cv::resize(mask_, wout, ws, 0, 0, cv::INTER_NEAREST); // baseline for graphcut
    } else {
        wgray = sourceGray_;
        wcolor = sourceColor_;
        wseeds = seeds_;
        wout = mask_.clone();
    }

    bool ok = true;
    switch (algo) {
        case FillAlgorithm::GrowCut:
            growCutFromSeeds(wgray, wseeds, wout, beta);
            break;
        case FillAlgorithm::RandomWalker:
            randomWalkerFromSeeds(wgray, wseeds, wout, beta);
            break;
        case FillAlgorithm::GraphCut:
            ok = graphCutFromSeeds(wcolor, wseeds, wout, foreground);
            break;
        default:
            return false;
    }
    if (!ok) return false;

    if (down)
        cv::resize(wout, mask_, sourceGray_.size(), 0, 0, cv::INTER_NEAREST);
    else
        wout.copyTo(mask_);
    return true;
}

void Document::pushHistory() {
    if (mask_.empty()) return;
    history_.push_back({mask_.clone(), seeds_.clone()});
    while (history_.size() > kMaxHistory) history_.pop_front();
}

void Document::undo() {
    if (history_.empty()) return;
    // Each snapshot is the state captured just before a mutation; restoring the
    // most recent one returns mask + seeds to before the last edit.
    Snapshot s = history_.back();
    history_.pop_back();
    s.mask.copyTo(mask_);
    s.seeds.copyTo(seeds_);
}

} // namespace orthoseg
