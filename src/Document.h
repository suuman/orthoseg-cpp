#pragma once
#include "Labels.h"
#include "SegmentationEngine.h"
#include <opencv2/core.hpp>
#include <deque>
#include <string>

namespace orthoseg {

// Holds the source X-ray, the indexed label mask, and a capped undo history.
// Knows nothing about Qt so it stays unit-testable.
class Document {
public:
    static constexpr size_t kMaxHistory = 20;

    bool loadImage(const std::string& path);
    bool exportMask(const std::string& path) const;

    bool hasImage() const { return !sourceGray_.empty(); }
    int  width()  const { return sourceGray_.cols; }
    int  height() const { return sourceGray_.rows; }

    const cv::Mat& sourceGray() const { return sourceGray_; }
    const cv::Mat& sourceColor() const { return sourceColor_; }
    const cv::Mat& mask() const { return mask_; }
    const cv::Mat& edgeMap() const { return edgeMap_; }
    const cv::Mat& seeds() const { return seeds_; }

    // Paint a brush stroke segment (thick line) between two points using the
    // given label. Background (id 0) erases.
    void paintLine(cv::Point a, cv::Point b, Label label, int brushSize);

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
    // Returns false if it could not run (no image/seeds, or missing fg/bg).
    bool runSeedSegmentation(FillAlgorithm algo, Label foreground, double beta);

    // Undo support: caller pushes a snapshot before a mutating gesture begins.
    // A snapshot captures both the mask and the seed layer.
    void pushHistory();
    bool canUndo() const { return !history_.empty(); }
    void undo();

private:
    struct Snapshot { cv::Mat mask; cv::Mat seeds; };

    cv::Mat sourceGray_;   // CV_8UC1
    cv::Mat sourceColor_;  // CV_8UC3 (BGR) for display
    cv::Mat mask_;         // CV_8UC1 indexed label ids
    cv::Mat seeds_;        // CV_8UC1 seed layer (label id or kNoSeed)
    cv::Mat edgeMap_;      // CV_8UC1, precomputed once per image
    std::deque<Snapshot> history_;
};

} // namespace orthoseg
