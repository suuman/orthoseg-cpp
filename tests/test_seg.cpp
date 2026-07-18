// Headless tests for the segmentation core (no Qt).
#include "SegmentationEngine.h"
#include "Labels.h"
#include <opencv2/imgproc.hpp>
#include <cassert>
#include <cstdio>

using namespace orthoseg;

static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { \
    std::printf("FAIL: %s\n", msg); ++failures; } \
    else std::printf("ok  : %s\n", msg); } while (0)

// A 100x100 image split into two flat halves with a hard vertical edge at x=50.
static cv::Mat makeTwoHalves() {
    cv::Mat img(100, 100, CV_8UC1, cv::Scalar(50));
    img(cv::Rect(50, 0, 50, 100)).setTo(200);
    return img;
}

static int countLabel(const cv::Mat& mask, Label l) {
    return cv::countNonZero(mask == static_cast<uchar>(l));
}

int main() {
    // --- Edge map: strong response along the x=50 boundary. ---
    {
        cv::Mat img = makeTwoHalves();
        cv::Mat edges = computeEdgeMap(img);
        CHECK(edges.type() == CV_8UC1, "edge map is 8-bit single channel");
        CHECK(edges.at<uchar>(50, 49) > 100 || edges.at<uchar>(50, 50) > 100,
              "edge map fires at the vertical boundary");
        CHECK(edges.at<uchar>(10, 10) == 0, "edge map is flat in uniform region");
    }

    // --- Standard region growing stays within the similar-intensity half. ---
    {
        cv::Mat img = makeTwoHalves();
        cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);
        regionGrowStandard(img, mask, cv::Point(10, 10), Label::Femur, 10);
        int femur = countLabel(mask, Label::Femur);
        CHECK(femur > 4000 && femur < 5200,
              "standard fill covers ~left half only");
        CHECK(mask.at<uchar>(50, 80) == 0,
              "standard fill does not cross into the bright half");
    }

    // --- Edge-embedded fill halts at strong boundaries. ---
    // The barrier is a closed rectangle away from the image border because,
    // matching the reference, the edge map's 1-px border is zero and never
    // blocks growth there.
    {
        cv::Mat img(100, 100, CV_8UC1, cv::Scalar(50));
        cv::rectangle(img, cv::Point(30, 30), cv::Point(70, 70),
                      cv::Scalar(200), 3);
        cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);
        // Permissive intensity threshold so only the edge penalty limits growth.
        regionGrowEdgeEmbedded(img, mask, cv::Point(50, 50), Label::Tibia,
                               255, 60);
        CHECK(mask.at<uchar>(50, 50) == static_cast<uchar>(Label::Tibia),
              "edge-embedded fill labels the seed region");
        CHECK(mask.at<uchar>(10, 10) == 0 && mask.at<uchar>(90, 90) == 0,
              "edge-embedded fill halts at the enclosing boundary");
        int n = countLabel(mask, Label::Tibia);
        CHECK(n > 900 && n < 2000,
              "edge-embedded fill covers ~the enclosed interior");
    }
    // Border-leak parity with the reference: a barrier touching the image
    // border can be bypassed along the zero-edge border rows.
    {
        cv::Mat img = makeTwoHalves();
        cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);
        regionGrowEdgeEmbedded(img, mask, cv::Point(10, 10), Label::Tibia,
                               255, 60);
        CHECK(mask.at<uchar>(0, 80) == static_cast<uchar>(Label::Tibia),
              "edge map border is zero, matching the reference");
    }

    // --- Split-and-merge groups the uniform half via union-find. ---
    {
        cv::Mat img = makeTwoHalves();
        cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);
        regionGrowSplitMerge(img, mask, cv::Point(10, 10), Label::Fibula,
                             10, 60, 4);
        CHECK(countLabel(mask, Label::Fibula) > 3000,
              "split-merge fills the seed's component");
        CHECK(mask.at<uchar>(50, 90) == 0,
              "split-merge does not leak into the other half");
    }

    // --- Background label acts as an eraser id (0). ---
    {
        cv::Mat img(20, 20, CV_8UC1, cv::Scalar(100));
        cv::Mat mask(20, 20, CV_8UC1, cv::Scalar(static_cast<uchar>(Label::Femur)));
        regionGrowStandard(img, mask, cv::Point(5, 5), Label::Background, 10);
        CHECK(countLabel(mask, Label::Femur) == 0,
              "background fill erases existing labels");
    }

    // ----- Seed-competition algorithms -----
    const double beta = 0.003; // matches the UI's default β

    // Helper: two seeds, one per half, competing across the hard boundary.
    auto twoSeedLayer = [](int w, int h) {
        cv::Mat seeds(h, w, CV_8UC1, cv::Scalar(kNoSeed));
        seeds.at<uchar>(10, 10) = static_cast<uchar>(Label::Femur); // left half
        seeds.at<uchar>(90, 90) = static_cast<uchar>(Label::Tibia); // right half
        return seeds;
    };

    // --- GrowCut: labels compete and split at the intensity boundary. ---
    {
        cv::Mat img = makeTwoHalves();
        cv::Mat seeds = twoSeedLayer(100, 100);
        cv::Mat out;
        growCutFromSeeds(img, seeds, out, beta);
        CHECK(out.at<uchar>(10, 10) == static_cast<uchar>(Label::Femur),
              "growcut keeps the femur seed's region");
        CHECK(out.at<uchar>(90, 90) == static_cast<uchar>(Label::Tibia),
              "growcut keeps the tibia seed's region");
        CHECK(out.at<uchar>(50, 10) == static_cast<uchar>(Label::Femur) &&
              out.at<uchar>(50, 90) == static_cast<uchar>(Label::Tibia),
              "growcut splits cleanly at the boundary");
        CHECK(countLabel(out, Label::Femur) > 4000 &&
              countLabel(out, Label::Tibia) > 4000,
              "growcut assigns roughly half to each label");
    }

    // --- Random walker: argmax of the harmonic fields splits the same way. ---
    {
        cv::Mat img = makeTwoHalves();
        cv::Mat seeds = twoSeedLayer(100, 100);
        cv::Mat out;
        randomWalkerFromSeeds(img, seeds, out, beta);
        CHECK(out.at<uchar>(20, 20) == static_cast<uchar>(Label::Femur),
              "random walker assigns the left half to femur");
        CHECK(out.at<uchar>(80, 80) == static_cast<uchar>(Label::Tibia),
              "random walker assigns the right half to tibia");
        CHECK(countLabel(out, Label::Femur) > 4000 &&
              countLabel(out, Label::Tibia) > 4000,
              "random walker splits roughly in half");
    }

    // --- Random walker forces a split via a background seed between regions. ---
    {
        // Uniform image: without contrast, the split is driven purely by the
        // seeds — a background stripe should separate two femur/tibia blobs.
        cv::Mat img(100, 100, CV_8UC1, cv::Scalar(120));
        cv::Mat seeds(100, 100, CV_8UC1, cv::Scalar(kNoSeed));
        seeds.at<uchar>(50, 20) = static_cast<uchar>(Label::Femur);
        seeds.at<uchar>(50, 80) = static_cast<uchar>(Label::Tibia);
        for (int y = 0; y < 100; ++y)
            seeds.at<uchar>(y, 50) = static_cast<uchar>(Label::Background);
        cv::Mat out;
        randomWalkerFromSeeds(img, seeds, out, beta);
        CHECK(out.at<uchar>(50, 20) == static_cast<uchar>(Label::Femur) &&
              out.at<uchar>(50, 80) == static_cast<uchar>(Label::Tibia),
              "background seed keeps femur and tibia on their own sides");
        CHECK(out.at<uchar>(50, 50) == static_cast<uchar>(Label::Background),
              "background seed line stays background");
    }

    // --- Graph cut (grabCut): a bright bar becomes foreground, dark stays bg. ---
    {
        cv::Mat color(100, 100, CV_8UC3, cv::Scalar(30, 30, 30));
        cv::rectangle(color, cv::Rect(30, 30, 40, 40),
                      cv::Scalar(210, 210, 210), cv::FILLED);
        cv::Mat seeds(100, 100, CV_8UC1, cv::Scalar(kNoSeed));
        seeds.at<uchar>(50, 50) = static_cast<uchar>(Label::Femur);      // fg
        seeds.at<uchar>(5, 5)   = static_cast<uchar>(Label::Background); // bg
        cv::Mat mask = cv::Mat::zeros(100, 100, CV_8UC1);
        bool ok = graphCutFromSeeds(color, seeds, mask, Label::Femur);
        CHECK(ok, "graph cut runs with fg and bg seeds");
        CHECK(mask.at<uchar>(50, 50) == static_cast<uchar>(Label::Femur),
              "graph cut labels the bright bar as femur");
        CHECK(mask.at<uchar>(5, 5) == 0,
              "graph cut leaves the dark background unlabeled");
        int femur = countLabel(mask, Label::Femur);
        CHECK(femur > 1000 && femur < 2600,
              "graph cut foreground ~ the bright bar area");
    }

    // --- Graph cut refuses to run without both fg and bg seeds. ---
    {
        cv::Mat color(50, 50, CV_8UC3, cv::Scalar(80, 80, 80));
        cv::Mat seeds(50, 50, CV_8UC1, cv::Scalar(kNoSeed));
        seeds.at<uchar>(25, 25) = static_cast<uchar>(Label::Femur); // fg only
        cv::Mat mask = cv::Mat::zeros(50, 50, CV_8UC1);
        bool ok = graphCutFromSeeds(color, seeds, mask, Label::Femur);
        CHECK(!ok, "graph cut returns false when background seeds are missing");
    }

    if (failures == 0) std::printf("\nALL TESTS PASSED\n");
    else std::printf("\n%d TEST(S) FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
