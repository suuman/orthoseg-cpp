#include "segmentor.hpp"
#include <opencv2/imgproc.hpp>
#include <cstdio>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    cv::Mat image(512, 128, CV_8UC3, cv::Scalar(0, 0, 0));
    image(cv::Rect(20, 50, 85, 410)).setTo(cv::Scalar(120, 120, 120));
    try {
        auto segmentor = xray_ocv::XRaySegmentor::create(argv[1], "cpu", 0.4f, false);
        const auto result = segmentor->predict(image);
        if (result.infer_height != 2048 || result.infer_width != 512 ||
            result.label_mask.size() != image.size() || result.label_mask.type() != CV_8UC1 ||
            cv::countNonZero(result.label_mask > 2)) {
            std::fprintf(stderr, "Native nnUNet returned invalid geometry or class IDs\n");
            return 1;
        }
        std::printf("Native nnUNet OpenCV %d.%d: 2048-high inference and source-size mask passed\n",
                    CV_VERSION_MAJOR, CV_VERSION_MINOR);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Native nnUNet failed: %s\n", e.what());
        return 1;
    }
}
