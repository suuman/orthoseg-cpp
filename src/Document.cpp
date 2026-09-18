#include "Document.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <cmath>

namespace orthoseg {

bool Document::loadImage(const std::string& path) {
    try {
        cv::Mat color = cv::imread(path, cv::IMREAD_COLOR);
        if (color.empty()) return false;

        // Prepare all layers before replacing the current document, so a
        // decoder or processing failure leaves the previous annotations intact.
        cv::Mat gray;
        // Use the channel mean, as in the web reference, rather than luma weights.
        cv::transform(color, gray, cv::Matx13f(1.f / 3, 1.f / 3, 1.f / 3));
        cv::Mat edges = computeEdgeMap(gray);
        cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8UC1);
        cv::Mat seeds(gray.size(), CV_8UC1, cv::Scalar(kNoSeed));

        sourceColor_ = color;
        sourceGray_ = gray;
        edgeMap_ = edges;
        mask_ = mask;
        seeds_ = seeds;
        history_.clear();
        aiFill_ = AIFillState{};
        return true;
    } catch (const cv::Exception&) {
        return false;
    }
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
    try {
        return cv::imwrite(path, out);
    } catch (const cv::Exception&) {
        // Missing/unsupported extensions and encoder failures can throw rather
        // than return false. Let the UI show its export error in either case.
        return false;
    }
}

void Document::applyAIResult() {
    if (aiFill_.resultMask.empty()) return;
    const cv::Mat& result = aiFill_.resultMask;
    if (result.size() != mask_.size() || result.type() != CV_8UC1) return;
    pushHistory();
    result.copyTo(mask_, result != 0);
    aiFill_.resultMask.release();
}

void Document::paintAIPrompt(cv::Point a, cv::Point b, bool erase, int brushSize) {
    if (!hasImage()) return;
    if (aiFill_.promptMask.empty())
        aiFill_.promptMask = cv::Mat::zeros(sourceGray_.size(), CV_8UC1);
    const cv::Scalar value(erase ? 0 : 255);
    cv::line(aiFill_.promptMask, a, b, value, brushSize, cv::LINE_8);
    const int radius = std::max(1, brushSize / 2);
    cv::circle(aiFill_.promptMask, a, radius, value, cv::FILLED);
    cv::circle(aiFill_.promptMask, b, radius, value, cv::FILLED);
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
        default:
            break; // Scribble algorithms require runSeedSegmentation().
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

// Stamp seeds into the working grid so thin strokes are not skipped. Labels
// can collide; the caller checks for lost classes and restores the original
// hard constraints after upsampling the result.
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
    if (sourceGray_.empty() || seedLabelCount() < 2) return false;
    if (!isScribbleAlgorithm(algo)) return false;
    if (algo == FillAlgorithm::GraphCut &&
        (foreground == Label::Background || !hasSeedForLabel(foreground)))
        return false;

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
        // Never silently run a different competition after an entire seed
        // class disappears into another label's working pixel.
        for (const auto& info : labels()) {
            if (hasSeedForLabel(info.id) &&
                cv::countNonZero(wseeds == static_cast<uchar>(info.id)) == 0)
                return false;
        }
    } else {
        wgray = sourceGray_;
        wcolor = sourceColor_;
        wseeds = seeds_;
    }
    // Graph Cut produces a foreground selection here; merge it into the
    // original mask later so unrelated labels never take a resizing round trip.
    wout = cv::Mat::zeros(ws, CV_8UC1);

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

    cv::Mat result;
    if (down)
        cv::resize(wout, result, sourceGray_.size(), 0, 0, cv::INTER_NEAREST);
    else
        result = wout;

    if (algo == FillAlgorithm::GraphCut) {
        const uchar fg = static_cast<uchar>(foreground);
        cv::Mat selected = result == fg;
        selected.setTo(0, (seeds_ != kNoSeed) & (seeds_ != fg));
        selected.setTo(255, seeds_ == fg);
        result = mask_.clone();
        result.setTo(0, (mask_ == fg) & (selected == 0));
        result.setTo(fg, selected);
    } else {
        // Working-grid collisions must not override explicit full-size seeds.
        seeds_.copyTo(result, seeds_ != kNoSeed);
    }

    // Snapshot only a successful run, immediately before committing its result.
    pushHistory();
    mask_ = result;
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
