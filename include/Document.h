#pragma once
#include "Labels.h"
#include "AIFill.h"
#include "SegmentationEngine.h"
#include <opencv2/core.hpp>
#include <deque>
#include <string>
#include <vector>

namespace orthoseg {

// Holds the source X-ray, independent bone channels, an indexed compatibility
// view for exclusive-label algorithms, and a capped undo history.
// Knows nothing about Qt so it stays unit-testable.
class Document {
public:
    static constexpr size_t kMaxHistory = 20;

    bool loadImage(const std::string& path);
    bool importMask(const std::string& path); // Exact 8-bit RGB label channels or grayscale 0..3.
    bool exportMask(const std::string& path) const;
    const std::vector<unsigned char>& originalPng() const { return originalPng_; }
    const std::string& sourcePath() const { return sourcePath_; }
    bool replaceAnatomyMask(const cv::Mat& mapped); // One undo step; preserve unrelated labels.

    bool hasImage() const { return !sourceGray_.empty(); }
    int  width()  const { return sourceGray_.cols; }
    int  height() const { return sourceGray_.rows; }

    const cv::Mat& sourceGray() const { return sourceGray_; }
    const cv::Mat& sourceColor() const { return sourceColor_; }
    const cv::Mat& mask() const { return mask_; }
    const cv::Mat& maskChannels() const { return channels_; } // BGR: 3 Fibula, 2 Tibia, 1 Femur.
    cv::Mat labelMask(Label label) const; // Binary 0/255 membership, including overlaps.
    cv::Mat monaiAnatomyMask() const; // Independent Femur/Tibia BGR channels, with blue cleared.
    const cv::Mat& edgeMap() const { return edgeMap_; }
    const cv::Mat& seeds() const { return seeds_; }
    AIFillState& aiFill() { return aiFill_; }
    const AIFillState& aiFill() const { return aiFill_; }
    void applyAIResult(); // Transfer preview foreground into the editable mask, with undo.
    void paintAIPrompt(cv::Point a, cv::Point b, bool erase, int brushSize);

    // Paint a brush stroke segment (thick line) between two points using the
    // given label. Background (id 0) erases.
    void paintLine(cv::Point a, cv::Point b, Label label, int brushSize, const cv::Mat& allowed = {});
    void eraseLabelLine(cv::Point a, cv::Point b, Label label, int brushSize, const cv::Mat& allowed = {});
    void fillPolygon(const std::vector<cv::Point>& points, Label label, bool recordHistory = true, const cv::Mat& allowed = {});
    bool fillEnclosedAt(cv::Point seed, Label label, const cv::Mat& allowed = {});
    // Binary connected region bounded by source-image edges; zero when seed lies on an edge.
    cv::Mat edgeRegion(cv::Point seed, int threshold) const;

    // Run the selected click-seed fill algorithm from a seed click.
    void fill(cv::Point seed, Label label, FillAlgorithm algo,
              int intensityThreshold, int edgePenaltyThreshold);

    void clearMask();

    // --- Seed-competition (scribble) support ---
    // Paint a seed stroke into the seed layer with the given label. Unlike
    // paintLine, Background (id 0) is a real seed here, not an eraser.
    void paintSeedLine(cv::Point a, cv::Point b, Label label, int brushSize);
    void clearSeeds();
    bool hasSeeds() const;
    bool hasSeedForLabel(Label label) const;
    int  seedLabelCount() const;   // number of distinct labels among the seeds

    // Run a scribble algorithm (GrowCut/RandomWalker/GraphCut) over the current
    // seeds, writing the result into the mask. `foreground` is the active label
    // (only used by GraphCut). Runs at a capped working resolution for speed.
    // Creates its own undo snapshot on success. Failure leaves mask/history
    // unchanged (including when downscaling loses a seed label). GraphCut
    // requires a bone foreground label; all methods need two seed labels.
    bool runSeedSegmentation(FillAlgorithm algo, Label foreground, double beta);

    // Undo support: caller pushes a snapshot before a mutating gesture begins,
    // except runSeedSegmentation(), which records its own successful mutation.
    // A snapshot captures both the mask and the seed layer.
    void pushHistory();
    void clearUndoHistory() { history_.clear(); }
    bool canUndo() const { return !history_.empty(); }
    void undo();

private:
    struct Snapshot { cv::Mat channels; cv::Mat seeds; };

    void syncIndexed(const cv::Rect& area);
    void applyRegion(const cv::Mat& region, Label label, bool eraseOnly = false);
    void stroke(cv::Point a, cv::Point b, Label label, int brushSize, bool eraseOnly, const cv::Mat& allowed);

    std::vector<unsigned char> originalPng_;
    std::string sourcePath_;
    cv::Mat sourceGray_;   // CV_8UC1
    cv::Mat sourceColor_;  // CV_8UC3 (BGR) for display
    cv::Mat channels_;     // CV_8UC3, BGR discrete labels; overlap is independent.
    cv::Mat mask_;         // CV_8UC1 compatibility view (Fibula > Tibia > Femur).
    cv::Mat seeds_;        // CV_8UC1 seed layer (label id or kNoSeed)
    cv::Mat edgeMap_;      // CV_8UC1, precomputed once per image
    std::deque<Snapshot> history_;
    AIFillState aiFill_;
};

} // namespace orthoseg
