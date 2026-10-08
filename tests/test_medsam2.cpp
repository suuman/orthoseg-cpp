#include "medsam2_ocv.h"
#include "medsam2_onnx.h"
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>

namespace {
int failures = 0;
#define CHECK(condition, message) do { \
    if (!(condition)) { std::printf("FAIL: %s\n", message); ++failures; } \
    else std::printf("ok  : %s\n", message); } while (0)

// Small ONNX fixtures encoded directly to keep the tests OpenCV-only.
void varint(std::string& result, uint64_t value) {
    while (value > 127) { result.push_back((value & 127) | 128); value >>= 7; }
    result.push_back(value);
}
void integer(std::string& result, int field, uint64_t value) {
    varint(result, field << 3); varint(result, value);
}
void bytes(std::string& result, int field, const std::string& value) {
    varint(result, (field << 3) | 2); varint(result, value.size()); result.append(value);
}
std::string info(const std::string& name, int type, const std::vector<int>& dims) {
    std::string shape;
    for (int dimension : dims) {
        std::string d; integer(d, 1, dimension); bytes(shape, 1, d);
    }
    std::string tensor, valueType, result;
    integer(tensor, 1, type); bytes(tensor, 2, shape);
    bytes(valueType, 1, tensor); bytes(result, 1, name); bytes(result, 2, valueType);
    return result;
}
std::string stringAttribute(const std::string& name, const std::string& value) {
    std::string result;
    bytes(result, 1, name); bytes(result, 4, value); integer(result, 20, 3);
    return result;
}
std::string tensor(const std::string& name, int type, const std::vector<int>& dims,
                   const char* raw, size_t count) {
    std::string result;
    for (int d : dims) integer(result, 1, d);
    integer(result, 2, type); bytes(result, 8, name); bytes(result, 9, std::string(raw, count));
    return result;
}
std::string model(const std::string& graph) {
    std::string result, opset;
    integer(result, 1, 8); bytes(result, 7, graph);
    integer(opset, 2, 17); bytes(result, 8, opset);
    return result;
}
void write(const std::filesystem::path& path, const std::string& data) {
    std::ofstream out(path, std::ios::binary); out.write(data.data(), data.size());
    if (!out) throw std::runtime_error("Cannot write ONNX test fixture");
}
std::string read(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
}

int main(int argc, char** argv) {
    try {
        const auto fixture = std::filesystem::current_path() / "test_medsam2_fixture.onnx";
        std::string resize;
        for (const char* input : {"image", "", "", "sizes"}) bytes(resize, 1, input);
        bytes(resize, 2, "result"); bytes(resize, 4, "Resize");
        bytes(resize, 5, stringAttribute("mode", "cubic"));
        bytes(resize, 5, stringAttribute("coordinate_transformation_mode", "half_pixel"));
        int64_t sizes[] = {1, 1, 1, 4};
        std::string graph;
        bytes(graph, 1, resize); bytes(graph, 2, "cubic-regression");
        bytes(graph, 5, tensor("sizes", 7, {4}, reinterpret_cast<char*>(sizes), sizeof(sizes)));
        bytes(graph, 11, info("image", 1, {1, 1, 1, 2}));
        bytes(graph, 12, info("result", 1, {1, 1, 1, 4}));
        const auto original = model(graph);
        write(fixture, original);
        auto net = readMedSAM2ONNX(fixture.string());
        int dimensions[] = {1, 1, 1, 2};
        float pixels[] = {1, 2};
        net.setInput(cv::Mat(4, dimensions, CV_32F, pixels), "image");
        const auto output = net.forward("result");
        const float expected[] = {0.89453125f, 1.2265625f, 1.7734375f, 2.10546875f};
        bool correct = output.total() == 4;
        for (size_t i = 0; correct && i < 4; ++i)
            correct = std::abs(output.ptr<float>()[i] - expected[i]) < 1e-6f;
        CHECK(correct, "native cubic Resize preserves the bicubic kernel and edge values");
        CHECK(read(fixture) == original, "model import leaves the ONNX file unchanged");

#if CV_VERSION_MAJOR >= 5
        std::string abs;
        bytes(abs, 1, "values"); bytes(abs, 2, "result"); bytes(abs, 4, "Abs");
        int64_t values[] = {-5, 0, 7};
        graph.clear(); bytes(graph, 1, abs); bytes(graph, 2, "integer-abs-regression");
        bytes(graph, 5, tensor("values", 7, {3}, reinterpret_cast<char*>(values), sizeof(values)));
        bytes(graph, 12, info("result", 7, {3}));
        write(fixture, model(graph));
        net = readMedSAM2ONNX(fixture.string());
        const auto absolute = net.forward("result");
        CHECK(absolute.type() == CV_64S && absolute.total() == 3 &&
              absolute.ptr<int64_t>()[0] == 5 && absolute.ptr<int64_t>()[1] == 0 &&
              absolute.ptr<int64_t>()[2] == 7, "integer shape Abs stays exact");
#endif
        write(fixture, std::string("\x3a\x80", 2));
        bool rejected = false;
        try { readMedSAM2ONNX(fixture.string()); } catch (const cv::Exception&) { rejected = true; }
        CHECK(rejected, "truncated ONNX input is rejected");
        std::filesystem::remove(fixture);

        if (argc > 1) {
            const auto directory = std::filesystem::path(argv[1]);
            const auto encoder = directory / "medsam2_image_encoder.onnx";
            const auto decoder = directory / "medsam2_mask_decoder.onnx";
            if (std::filesystem::is_regular_file(encoder) && std::filesystem::is_regular_file(decoder)) {
                MedSAM2Engine engine(encoder.string(), decoder.string(), "cpu");
                cv::Mat image(160, 80, CV_8UC3, cv::Scalar(100, 100, 100));
                Prompt box; box.box = {10, 10, 65, 140};
                Prompt mask = box; mask.class_id = 2; mask.has_mask = true;
                cv::Mat labels = cv::Mat::zeros(image.size(), CV_8UC1);
                labels(cv::Rect(20, 20, 30, 100)).setTo(2);
                mask.mask_logits = makeMaskLogits(labels, 2, computePadInfo(image.cols, image.rows));
                const auto result = engine.segment(image, {box, mask});
                CHECK(result.label_map.size() == image.size() && result.label_map.type() == CV_8UC1,
                      "real MedSAM2 box and mask inference restores source dimensions");
                CHECK(cv::countNonZero(result.label_map > 2) == 0 && result.iou_scores.size() == 2 &&
                      std::isfinite(result.iou_scores.at(1)) && std::isfinite(result.iou_scores.at(2)),
                      "real model outputs contain valid labels and finite IoU scores");
            } else std::puts("SKIP: real MedSAM2 ONNX files are not present");
        }
    } catch (const std::exception& error) {
        std::printf("FAIL: %s\n", error.what()); ++failures;
    }
    return failures ? 1 : 0;
}
