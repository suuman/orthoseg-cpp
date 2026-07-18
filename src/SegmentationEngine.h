#pragma once
#include "Labels.h"
#include <opencv2/core.hpp>

namespace orthoseg {

// The mask is a single-channel CV_8U indexed image: each pixel holds a Label
// id (0..3). Rendering to color happens in the UI layer. Writing the
// Background id (0) over a pixel therefore erases whatever label was there.

// 3x3 Sobel gradient magnitude of a grayscale image, clamped to 8-bit.
// Returns a CV_8U single-channel edge-strength map.
cv::Mat computeEdgeMap(const cv::Mat& graySource);

// A. Standard region growing (4-way BFS).
// Grows from `seed` while |intensity - seedIntensity| <= intensityThreshold,
// writing `label` into `mask`.
void regionGrowStandard(const cv::Mat& graySource, cv::Mat& mask,
                        cv::Point seed, Label label, int intensityThreshold);

// B. Embedded boundary info. Same as Standard but halts growth at any pixel
// whose edge magnitude exceeds edgePenaltyThreshold. `edgeMap` may be passed in
// to avoid recomputation; if empty it is computed internally.
void regionGrowEdgeEmbedded(const cv::Mat& graySource, cv::Mat& mask,
                            cv::Point seed, Label label,
                            int intensityThreshold, int edgePenaltyThreshold,
                            const cv::Mat& edgeMap = cv::Mat());

// C. Split-and-merge with contours over a block grid + union-find.
void regionGrowSplitMerge(const cv::Mat& graySource, cv::Mat& mask,
                          cv::Point seed, Label label,
                          int intensityThreshold, int edgePenaltyThreshold,
                          int blockSize = 4,
                          const cv::Mat& edgeMap = cv::Mat());

// -------------------------------------------------------------------------
// Seed-competition (scribble) algorithms.
//
// `seeds` is a CV_8UC1 layer the same size as the source: each pixel holds a
// Label id (0..3) where the user scribbled, or kNoSeed elsewhere. Background
// (id 0) is a legitimate competing seed here (not an eraser), which is what
// lets you force a clean split at the knee/hip by seeding between two bones.
constexpr uchar kNoSeed = 255;

// D. GrowCut — synchronous cellular automaton (Vezhnevets & Konouchine 2005).
// Each cell has (label, strength in [0,1]); seeds start at strength 1. Every
// iteration a neighbour q attacks p with strength g(Ip,Iq)*strength_q where
// g = exp(-beta*(Ip-Iq)^2); p is converted if the attack exceeds its own
// strength. Iterates to convergence. Writes a full labeling into `outMask`.
void growCutFromSeeds(const cv::Mat& graySource, const cv::Mat& seeds,
                      cv::Mat& outMask, double beta, int maxIters = -1);

// E. Random walker — for each seeded label, solve the weighted Dirichlet
// (harmonic) problem with that label's seeds held at 1 and all other seeds at
// 0; edge weights w = exp(-beta*(Ip-Iq)^2). The value at a pixel is the
// probability a random walker from it reaches a same-label seed first; each
// pixel is assigned the argmax label. Solved by SOR relaxation. Full labeling.
void randomWalkerFromSeeds(const cv::Mat& graySource, const cv::Mat& seeds,
                           cv::Mat& outMask, double beta,
                           int maxIters = 500, double tol = 1e-3);

// F. Graph cut — OpenCV grabCut (color GMMs + min-cut/max-flow). The active
// `foreground` label's scribbles become hard foreground, every other scribble
// becomes hard background, unseeded pixels are probable background. On return,
// pixels the cut assigns to foreground get `foreground`; pixels it assigns to
// background that previously held `foreground` are cleared. Other labels in
// `mask` are left untouched. Returns false if foreground or background seeds
// are missing (grabCut needs both). `mask` is modified in place.
bool graphCutFromSeeds(const cv::Mat& colorSource, const cv::Mat& seeds,
                       cv::Mat& mask, Label foreground, int iterations = 3);

} // namespace orthoseg
