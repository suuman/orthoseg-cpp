#include "AIFillController.h"
#include <QCoreApplication>
#include <QEventLoop>
#include <QTimer>
#include <cstdio>
#include <limits>
#include <filesystem>

using namespace orthoseg;
static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("FAIL: %s\n", m); ++failures; } \
    else std::printf("ok  : %s\n", m); } while (0)

template<class F> bool rejects(F f) {
    try { f(); } catch (const std::exception&) { return true; }
    return false;
}

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    const cv::Size size(100, 200);
    Box b = validatedBox({80, 150, 10, 20}, size);
    CHECK(b.x0 == 10 && b.y0 == 20 && b.x1 == 80, "reverse box normalized");
    b = validatedBox({-20, -10, 120, 250}, size);
    CHECK(b.x0 == 0 && b.y0 == 0 && b.x1 == 99 && b.y1 == 199, "partial box clamped");
    CHECK(rejects([&] { validatedBox({4, 5, 4, 8}, size); }), "zero-width box rejected");
    CHECK(rejects([&] { validatedBox({-20, 5, -2, 8}, size); }), "outside box rejected");
    CHECK(rejects([&] { validatedBox({0, 0, 10, 10}, {}); }), "invalid image size rejected");
    CHECK(rejects([&] { validatedBox({0, 0, std::numeric_limits<float>::infinity(), 10}, size); }),
          "nonfinite coordinates rejected");
    AIFillRequest request;
    request.imageBGR = cv::Mat(size, CV_8UC3, cv::Scalar(20, 40, 80));
    request.box = Box{10, 20, 80, 150};
    auto prompt = MedSAM2Inference::preparePrompt(request);
    CHECK(prompt.class_id == 1 && !prompt.has_mask && prompt.box.x0 == 10,
          "bounding-box adapter preserves ocv source coordinates and femur label");
    cv::Mat result = cv::Mat::zeros(size, CV_8UC1);
    CHECK(!rejects([&] { validateAIResult(result, size, Label::Femur); }), "source-sized result accepted");
    CHECK(rejects([&] { validateAIResult(result, {20, 20}, Label::Femur); }), "wrong result size rejected");
    result.at<uchar>(0, 0) = 2;
    CHECK(rejects([&] { validateAIResult(result, size, Label::Femur); }), "unexpected result label rejected");

    cv::Mat mask = cv::Mat::zeros(size, CV_16UC1);
    mask(cv::Rect(10, 20, 70, 130)).setTo(4096);
    auto binary = binaryPromptMask(mask, size);
    CHECK(binary.type() == CV_8UC1 && binary.at<uchar>(30, 30) == 255 &&
          binary.at<uchar>(0, 0) == 0, "16-bit prompt converted to binary without losing low values");
    CHECK(rejects([&] { binaryPromptMask(mask, {20, 20}); }), "wrong prompt dimensions rejected");
    CHECK(rejects([&] { binaryPromptMask(cv::Mat::zeros(size, CV_8UC1), size); }), "empty foreground rejected");
    request.type = AIFillPromptType::PaintedMask;
    request.target = Label::Tibia;
    request.promptMask = binary;
    prompt = MedSAM2Inference::preparePrompt(request);
    CHECK(prompt.has_mask && prompt.class_id == 2 && prompt.box.x0 == 10 &&
          prompt.box.x1 == 79 && prompt.box.y1 == 149,
          "mask adapter uses ocv tight inclusive box and tibia target");
    const auto expected = makeMaskLogits(binary, 255, computePadInfo(100, 200));
    CHECK(prompt.mask_logits.dims == 4 && prompt.mask_logits.size[2] == 256 &&
          cv::norm(prompt.mask_logits, expected, cv::NORM_INF) == 0,
          "mask logits use existing ocv preprocessing unchanged");

    cv::Mat colorMask(size, CV_8UC4, cv::Scalar(0, 0, 0, 255));
    colorMask.at<cv::Vec4b>(2, 3)[0] = 1;
    const auto colorBinary = binaryPromptMask(colorMask, size);
    CHECK(cv::countNonZero(colorBinary) == 1 && colorBinary.at<uchar>(2, 3) == 255,
          "RGB channels convert to binary without treating alpha as foreground");
    CHECK(rejects([&] { binaryPromptMask(cv::Mat(size, CV_32FC1), size); }),
          "unsupported prompt depth rejected");
    request.type = AIFillPromptType::LoadedMask;
    prompt = MedSAM2Inference::preparePrompt(request);
    CHECK(prompt.has_mask && cv::norm(prompt.mask_logits, expected, cv::NORM_INF) == 0,
          "loaded and painted prompts share the same ocv mask path");

    // Real worker, deterministic missing-model error; no simulated inference.
    AIFillController controller;
    request.modelDirectory = "/nonexistent/orthoseg-models";
    QEventLoop loop;
    bool failed = false, uiThread = false, heartbeat = false;
    QObject::connect(&controller, &AIFillController::failed, &loop, [&](const QString& error) {
        failed = error.contains("Model directory");
        uiThread = QThread::currentThread() == app.thread();
        loop.quit();
    });
    CHECK(controller.run(request), "first inference request accepted");
    CHECK(!controller.run(request), "duplicate request rejected while busy");
    QTimer::singleShot(0, &loop, [&] { heartbeat = true; });
    QTimer::singleShot(5000, &loop, &QEventLoop::quit);
    loop.exec();
    CHECK(failed && uiThread && !controller.running(), "worker errors return safely to UI thread");
    CHECK(heartbeat, "UI event loop remains responsive");

    // Exercise the real CUDA preflight when the checkout's models are available.
    if (cv::cuda::getCudaEnabledDeviceCount() <= 0 &&
        std::filesystem::is_regular_file(std::filesystem::path(ORTHOSEG_MODEL_DIR) /
                                        "medsam2_image_encoder.onnx")) {
        request.modelDirectory = ORTHOSEG_MODEL_DIR;
        bool cudaError = false;
        try {
            MedSAM2Inference inference;
            inference.run(request);
        } catch (const std::exception& error) {
            cudaError = std::string(error.what()).find("CUDA is unavailable") != std::string::npos;
        }
        CHECK(cudaError, "real models with no CUDA produce an explicit error, not CPU fallback");
    }
    std::printf("%d AI test failure(s)\n", failures);
    return failures ? 1 : 0;
}
