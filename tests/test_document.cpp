#include "Document.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <utility>

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
        CHECK(doc.loadImage(input), "load edge editing source");
        doc.paintLine({300, 30}, {300, 30}, Label::Femur, 1);
        CHECK(cv::countNonZero(doc.labelMask(Label::Femur)) == 1, "size 1 brush paints exactly one pixel");
        doc.eraseLabelLine({300, 30}, {300, 30}, Label::Femur, 1);
        CHECK(cv::countNonZero(doc.labelMask(Label::Femur)) == 0, "size 1 eraser removes one pixel");
        const auto region = doc.edgeRegion({300, 30}, 30);
        CHECK(region.at<uchar>(30, 300) && !region.at<uchar>(30, 600), "image edges bound connected editing region");
        doc.paintLine({300, 30}, {600, 30}, Label::Femur, 7, region);
        CHECK(doc.mask().at<uchar>(30, 300) == 1 && doc.mask().at<uchar>(30, 600) == 0,
              "brush stops at source edge even when cursor crosses it");
        doc.paintLine({300, 30}, {600, 30}, Label::Tibia, 7);
        doc.eraseLabelLine({300, 30}, {600, 30}, Label::Tibia, 7, region);
        CHECK(doc.maskChannels().at<cv::Vec3b>(30, 300)[2] == 1 &&
              doc.maskChannels().at<cv::Vec3b>(30, 300)[1] == 0 &&
              doc.maskChannels().at<cv::Vec3b>(30, 600)[1] == 2,
              "edge constrained eraser preserves other bones and labels beyond edge");
        doc.clearMask();
        const std::vector<cv::Point> polygon{{300, 25}, {600, 25}, {600, 40}, {300, 40}};
        doc.fillPolygon(polygon, Label::Femur, true, region);
        CHECK(doc.mask().at<uchar>(30, 350) == 1 && doc.mask().at<uchar>(30, 550) == 0,
              "lasso and draw-fill polygons are clipped to edge region");
        doc.undo();
        CHECK(cv::countNonZero(doc.mask()) == 0, "constrained fill undo restores mask");
        doc.fillPolygon(polygon, Label::Femur);
        CHECK(doc.mask().at<uchar>(30, 550) == 1, "unconstrained fill crosses source edges");
        CHECK(cv::countNonZero(doc.edgeRegion({300, 30}, 255)) == doc.width()*doc.height(),
              "maximum threshold allows every pixel");
        CHECK(cv::countNonZero(doc.edgeRegion({256, 30}, 30)) == 0, "starting on edge cannot leak to another region");
        doc.clearMask();
        CHECK(doc.fillEnclosedAt({300, 30}, Label::Femur, region) && doc.mask().at<uchar>(30, 600) == 0,
              "enclosed-hole fill respects image edge constraint");
    }

    {
        Document doc;
        CHECK(doc.loadImage(input), "load source image");
        doc.paintLine({20, 20}, {20, 20}, Label::Femur, 2);
        doc.paintLine({30, 20}, {30, 20}, Label::Fibula, 2);
        const auto beforeAutomatic = doc.mask().clone();
        doc.aiFill().resultMask = cv::Mat::zeros(doc.mask().size(), CV_8UC1);
        doc.aiFill().resultMask.at<uchar>(25, 25) = static_cast<uchar>(Label::Tibia);
        doc.aiFill().resultMask.at<uchar>(20, 30) = static_cast<uchar>(Label::Tibia);
        doc.aiFill().resultReplacesAnatomy = true;
        doc.applyAIResult();
        CHECK(doc.mask().at<uchar>(20, 20) == 0 && doc.mask().at<uchar>(25, 25) == 2 &&
              doc.mask().at<uchar>(20, 30) == 3 &&
              doc.maskChannels().at<cv::Vec3b>(20, 30) == cv::Vec3b(3, 2, 0) &&
              doc.aiFill().resultMask.empty(),
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
              saved.at<cv::Vec3b>(40, 40) == cv::Vec3b(0, 2, 0),
              "export retains source dimensions and exact discrete channel IDs");
        const cv::Mat maskBeforeLoad = doc.mask().clone();
        const cv::Mat seedsBeforeLoad = doc.seeds().clone();
        CHECK(!doc.loadImage((tmp.path / "missing.png").string()),
              "invalid input reports a load failure");
        CHECK(same(doc.mask(), maskBeforeLoad) && same(doc.seeds(), seedsBeforeLoad),
              "failed image load retains the current annotations");
    }

    {
        Document doc;
        CHECK(doc.loadImage(input), "load image for overlapping labels");
        const cv::Point all(100, 10), fem(110, 10), tib(120, 10), fib(130, 10);
        const cv::Point femTib(140, 10), femFib(150, 10), tibFib(160, 10);
        auto put = [&doc](cv::Point point, Label label) { doc.paintLine(point, point, label, 1); };
        for (auto point : {all, fem, femTib, femFib}) put(point, Label::Femur);
        for (auto point : {all, tib, femTib, tibFib}) put(point, Label::Tibia);
        for (auto point : {all, fib, femFib, tibFib}) put(point, Label::Fibula);
        // Keep all seven non-background combinations distinct in the source channels.
        const auto output = (tmp.path / "overlap.png").string();
        CHECK(doc.exportMask(output), "overlapping channel mask exports as PNG");
        const auto saved = cv::imread(output, cv::IMREAD_UNCHANGED);
        CHECK(saved.type() == CV_8UC3 && saved.at<cv::Vec3b>(all) == cv::Vec3b(3, 2, 1) &&
              saved.at<cv::Vec3b>(fem) == cv::Vec3b(0, 0, 1) &&
              saved.at<cv::Vec3b>(tib) == cv::Vec3b(0, 2, 0) &&
              saved.at<cv::Vec3b>(fib) == cv::Vec3b(3, 0, 0) &&
              saved.at<cv::Vec3b>(femTib) == cv::Vec3b(0, 2, 1) &&
              saved.at<cv::Vec3b>(femFib) == cv::Vec3b(3, 0, 1) &&
              saved.at<cv::Vec3b>(tibFib) == cv::Vec3b(3, 2, 0),
              "export stores independent B=3, G=2, R=1 values at overlaps");
        CHECK(doc.labelMask(Label::Femur).at<uchar>(all) == 255 &&
              doc.labelMask(Label::Tibia).at<uchar>(all) == 255 &&
              doc.labelMask(Label::Fibula).at<uchar>(all) == 255 &&
              doc.monaiAnatomyMask().at<cv::Vec3b>(all) == cv::Vec3b(0, 2, 1),
              "MONAI training receives both overlapping channels with Fibula omitted");
        Document imported;
        CHECK(imported.loadImage(input) && imported.importMask(output) &&
              same(imported.maskChannels(), doc.maskChannels()),
              "RGB mask round-trips every overlapping channel");
        imported.pushHistory();
        imported.eraseLabelLine(all, all, Label::Tibia, 1);
        CHECK(imported.maskChannels().at<cv::Vec3b>(all) == cv::Vec3b(3, 0, 1),
              "eraser removes only the selected label channel");
        imported.undo();
        CHECK(imported.maskChannels().at<cv::Vec3b>(all) == cv::Vec3b(3, 2, 1),
              "undo restores all overlapping channels");
        cv::Mat grayscale(source.size(), CV_8UC1, cv::Scalar(0));
        grayscale.at<uchar>(2, 2) = 1;
        grayscale.at<uchar>(3, 3) = 2;
        grayscale.at<uchar>(4, 4) = 3;
        const auto grayPath = (tmp.path / "grayscale-mask.png").string();
        cv::imwrite(grayPath, grayscale);
        CHECK(imported.importMask(grayPath) &&
              imported.maskChannels().at<cv::Vec3b>(2, 2) == cv::Vec3b(0, 0, 1) &&
              imported.maskChannels().at<cv::Vec3b>(3, 3) == cv::Vec3b(0, 2, 0) &&
              imported.maskChannels().at<cv::Vec3b>(4, 4) == cv::Vec3b(3, 0, 0),
              "single-channel 0–3 mask maps to its corresponding bone channel");
        cv::Mat rgbGray;
        cv::cvtColor(grayscale, rgbGray, cv::COLOR_GRAY2BGR);
        const auto rgbGrayPath = (tmp.path / "rgb-grayscale-mask.png").string();
        cv::imwrite(rgbGrayPath, rgbGray);
        CHECK(imported.importMask(rgbGrayPath) &&
              imported.maskChannels().at<cv::Vec3b>(4, 4) == cv::Vec3b(3, 0, 0),
              "RGB images with equal channels decode as grayscale label IDs");
        auto malformed = rgbGray.clone();
        malformed.at<cv::Vec3b>(5, 5) = cv::Vec3b(1, 2, 3);
        const auto badPath = (tmp.path / "invalid-mask.png").string();
        cv::imwrite(badPath, malformed);
        const auto beforeBad = imported.maskChannels().clone();
        CHECK(!imported.importMask(badPath) && same(imported.maskChannels(), beforeBad),
              "invalid discrete channel values are rejected without changing the mask");
        imported.fillPolygon({{200, 10}, {220, 10}, {220, 30}, {200, 30}}, Label::Femur);
        imported.fillPolygon({{210, 10}, {230, 10}, {230, 30}, {210, 30}}, Label::Tibia);
        CHECK(imported.maskChannels().at<cv::Vec3b>(20, 215) == cv::Vec3b(0, 2, 1),
              "polygon fills add the chosen label without replacing overlaps");
        imported.undo();
        CHECK(imported.maskChannels().at<cv::Vec3b>(20, 215) == cv::Vec3b(0, 0, 1),
              "polygon fill is independently undoable");
        for (const auto& edge : {std::pair{cv::Point(300, 10), cv::Point(320, 10)},
                                 std::pair{cv::Point(320, 10), cv::Point(320, 30)},
                                 std::pair{cv::Point(320, 30), cv::Point(300, 30)},
                                 std::pair{cv::Point(300, 30), cv::Point(300, 10)}})
            imported.paintLine(edge.first, edge.second, Label::Femur, 1);
        CHECK(imported.fillEnclosedAt({310, 20}, Label::Femur) &&
              imported.maskChannels().at<cv::Vec3b>(20, 310)[2] == 1 &&
              !imported.fillEnclosedAt({400, 20}, Label::Femur),
              "Draw & Fill hole mode fills only enclosed regions");
    }

    {
        Document refined;
        CHECK(refined.loadImage(input), "load local refinement regression source");
        refined.paintLine({10, 10}, {12, 10}, Label::Femur, 1);
        refined.paintLine({12, 10}, {12, 10}, Label::Tibia, 1);
        refined.paintLine({12, 10}, {12, 10}, Label::Fibula, 1);
        const auto before = refined.maskChannels().clone();
        refined.aiFill().resultMask = cv::Mat::zeros(refined.mask().size(), CV_8UC1);
        refined.aiFill().resultMask.at<uchar>(10, 10) = 1;
        refined.aiFill().resultLabels = {1};
        refined.applyAIResult();
        CHECK(refined.maskChannels().at<cv::Vec3b>(10, 12) == cv::Vec3b(3, 2, 0),
              "local refinement removes old Femur while retaining overlapping Tibia and Fibula");
        CHECK(refined.maskChannels().at<cv::Vec3b>(10, 10)[2] == 1, "local refinement retains predicted foreground");
        refined.undo();
        CHECK(same(before, refined.maskChannels()), "local refinement undo restores every bone channel");
        refined.aiFill().resultMask = cv::Mat::zeros(refined.mask().size(), CV_8UC1);
        refined.aiFill().resultLabels = {1};
        refined.applyAIResult();
        CHECK(cv::countNonZero(refined.labelMask(Label::Femur)) == 0 &&
              cv::countNonZero(refined.labelMask(Label::Tibia)) == 1,
              "empty local prediction clears the prompted bone only");
    }

    std::printf("\n%d document test failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
