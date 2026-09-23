#include "Document.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <stdexcept>

using namespace orthoseg;

static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { \
    std::printf("FAIL: %s\n", msg); ++failures; } \
    else std::printf("ok  : %s\n", msg); } while (0)

// Keep image I/O tests isolated, including when separate builds run concurrently.
struct TempDirectory {
    std::filesystem::path path;
    TempDirectory() {
        auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
               ("orthoseg-document-" + std::to_string(stamp));
        if (!std::filesystem::create_directory(path))
            throw std::runtime_error("Cannot create test directory");
    }
    ~TempDirectory() { std::filesystem::remove_all(path); }
};

static bool same(const cv::Mat& a, const cv::Mat& b) {
    return a.size() == b.size() && a.type() == b.type() &&
           cv::norm(a, b, cv::NORM_INF) == 0;
}

int main() {
    TempDirectory tmp;
    const auto input = (tmp.path / "source.png").string();
    cv::Mat source(64, 1024, CV_8UC3, cv::Scalar(30, 30, 30));
    cv::rectangle(source, cv::Rect(256, 16, 256, 32),
                  cv::Scalar(210, 210, 210), cv::FILLED);
    if (!cv::imwrite(input, source)) return 1;

    {
        Document doc;
        CHECK(doc.loadImage(input), "load source image");
        doc.paintLine({20, 20}, {20, 20}, Label::Femur, 2);
        doc.paintLine({30, 20}, {30, 20}, Label::Fibula, 2);
        const auto beforeAutomatic = doc.mask().clone();
        doc.aiFill().resultMask = cv::Mat::zeros(doc.mask().size(), CV_8UC1);
        doc.aiFill().resultMask.at<uchar>(25, 25) = static_cast<uchar>(Label::Tibia);
        doc.aiFill().resultReplacesAnatomy = true;
        doc.applyAIResult();
        CHECK(doc.mask().at<uchar>(20, 20) == 0 && doc.mask().at<uchar>(25, 25) == 2 &&
              doc.mask().at<uchar>(20, 30) == 3 && doc.aiFill().resultMask.empty(),
              "automatic AI preview replaces anatomy while preserving Fibula");
        doc.undo();
        CHECK(same(doc.mask(), beforeAutomatic), "automatic AI result applies in one undo step");
    }

    {
        Document doc;
        CHECK(doc.loadImage(input), "load source image");
        doc.paintLine({51, 5}, {51, 5}, Label::Tibia, 2);
        doc.paintLine({800, 51}, {810, 51}, Label::Fibula, 2);
        doc.paintLine({901, 11}, {901, 11}, Label::Femur, 2);
        doc.paintSeedLine({382, 30}, {382, 30}, Label::Femur, 2);
        doc.paintSeedLine({383, 31}, {383, 31}, Label::Background, 2);
        doc.paintSeedLine({51, 5}, {51, 5}, Label::Background, 2);
        // A narrow background correction inside the bright foreground object.
        doc.paintSeedLine({451, 31}, {451, 31}, Label::Background, 2);
        const cv::Mat before = doc.mask().clone();
        const cv::Mat seeds = doc.seeds().clone();
        CHECK(doc.runSeedSegmentation(FillAlgorithm::GraphCut, Label::Femur, 0.003),
              "graph cut runs on a downscaled working image");
        CHECK(same(doc.mask() == 2, before == 2) && same(doc.mask() == 3, before == 3),
              "graph cut preserves other labels outside foreground pixel-for-pixel");
        CHECK(doc.mask().at<uchar>(11, 901) == 0,
              "graph cut retracts the old active label outside foreground");
        CHECK(cv::countNonZero((doc.seeds() == 0) & (doc.mask() == 1)) == 0,
              "graph cut honors full-resolution background seed strokes");
        CHECK(cv::countNonZero((doc.seeds() == 1) & (doc.mask() != 1)) == 0,
              "graph cut honors full-resolution foreground seed strokes");
        CHECK(doc.canUndo(), "successful segmentation creates an undo snapshot");
        doc.undo();
        CHECK(same(doc.mask(), before) && same(doc.seeds(), seeds),
              "undo segmentation restores the exact mask and seed layer");
    }

    for (auto algo : {FillAlgorithm::GrowCut, FillAlgorithm::RandomWalker}) {
        Document doc;
        CHECK(doc.loadImage(input), "load image for multi-label segmentation");
        doc.paintSeedLine({382, 30}, {382, 30}, Label::Femur, 2);
        doc.paintSeedLine({51, 5}, {51, 5}, Label::Tibia, 2);
        // Adjacent strokes overlap in the reduced grid but still have distinct
        // full-resolution pixels that must retain their explicitly seeded label.
        doc.paintSeedLine({383, 31}, {383, 31}, Label::Background, 2);
        CHECK(doc.runSeedSegmentation(algo, Label::Femur, 0.003),
              "multi-label segmentation runs on a downscaled image");
        CHECK(cv::countNonZero((doc.seeds() != kNoSeed) &
                               (doc.mask() != doc.seeds())) == 0,
              "multi-label segmentation preserves every full-resolution seed");
    }

    // Large reduction maps these distinct strokes into a single working pixel.
    const auto wideInput = (tmp.path / "wide.png").string();
    cv::imwrite(wideInput, cv::Mat(32, 8192, CV_8UC3, cv::Scalar(100, 100, 100)));
    for (auto algo : {FillAlgorithm::GrowCut, FillAlgorithm::RandomWalker,
                      FillAlgorithm::GraphCut}) {
        Document doc;
        CHECK(doc.loadImage(wideInput), "load wide image");
        doc.paintLine({400, 10}, {420, 10}, Label::Fibula, 2);
        doc.paintSeedLine({3, 3}, {3, 3}, Label::Femur, 2);
        doc.paintSeedLine({10, 10}, {10, 10}, Label::Background, 2);
        const cv::Mat before = doc.mask().clone();
        CHECK(!doc.runSeedSegmentation(algo, Label::Femur, 0.003),
              "reject a run when resizing removes an entire seed label");
        CHECK(same(doc.mask(), before) && !doc.canUndo(),
              "failed segmentation leaves annotations and undo history untouched");
    }

    {
        Document doc;
        CHECK(doc.loadImage(input), "load image for validation and export");
        doc.paintLine({40, 40}, {45, 40}, Label::Tibia, 2);
        doc.paintSeedLine({383, 31}, {383, 31}, Label::Femur, 2);
        const cv::Mat before = doc.mask().clone();
        CHECK(!doc.runSeedSegmentation(FillAlgorithm::RandomWalker, Label::Femur, 0.003),
              "document rejects a segmentation with only one seed label");
        CHECK(same(doc.mask(), before), "rejected one-label run keeps the mask");
        doc.paintSeedLine({51, 5}, {51, 5}, Label::Background, 2);
        CHECK(!doc.runSeedSegmentation(FillAlgorithm::GraphCut, Label::Background, 0.003),
              "graph cut requires a bone as the active foreground label");

        bool exported = true;
        bool threw = false;
        try {
            exported = doc.exportMask((tmp.path / "mask.unsupported").string());
        } catch (const cv::Exception&) {
            threw = true;
        }
        CHECK(!threw && !exported, "unsupported export returns failure without throwing");
        const auto output = (tmp.path / "mask.png").string();
        CHECK(doc.exportMask(output), "PNG export succeeds");
        const cv::Mat saved = cv::imread(output);
        CHECK(saved.size() == source.size() && saved.type() == CV_8UC3 &&
              saved.at<cv::Vec3b>(40, 40) == labelInfo(Label::Tibia).colorBGR,
              "export retains source dimensions and exact label colors");
        const cv::Mat maskBeforeLoad = doc.mask().clone();
        const cv::Mat seedsBeforeLoad = doc.seeds().clone();
        CHECK(!doc.loadImage((tmp.path / "missing.png").string()),
              "invalid input reports a load failure");
        CHECK(same(doc.mask(), maskBeforeLoad) && same(doc.seeds(), seedsBeforeLoad),
              "failed image load retains the current annotations");
    }

    std::printf("\n%d document test failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
