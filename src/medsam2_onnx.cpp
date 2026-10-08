#include "medsam2_onnx.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn/layer.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <mutex>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <string_view>

namespace {
#if CV_VERSION_MAJOR >= 5
class IntegerAbsLayer final : public cv::dnn::Layer {
public:
    explicit IntegerAbsLayer(const cv::dnn::LayerParams& params)
        : cv::dnn::Layer(params), builtin_(cv::dnn::AbsLayer::create(params)) {}
    bool supportBackend(int backend) override { return builtin_->supportBackend(backend); }
    bool getMemoryShapes(const std::vector<cv::MatShape>& inputs, int,
                         std::vector<cv::MatShape>& outputs,
                         std::vector<cv::MatShape>& internals) const override {
        outputs = inputs; internals.clear(); return false;
    }
    void forward(cv::InputArrayOfArrays inputArrays, cv::OutputArrayOfArrays outputArrays,
                 cv::OutputArrayOfArrays internals) override {
        std::vector<cv::Mat> inputs, outputs;
        inputArrays.getMatVector(inputs); outputArrays.getMatVector(outputs);
        CV_Assert(inputs.size() == outputs.size());
        if (inputs.empty() || inputs[0].type() != CV_64S) {
            builtin_->forward(inputArrays, outputArrays, internals);
            return;
        }
        for (size_t n = 0; n < inputs.size(); ++n) {
            const auto& input = inputs[n]; auto& output = outputs[n];
            if (input.type() == CV_64S) {
                CV_Assert(input.isContinuous() && output.isContinuous() && output.type() == CV_64S);
                const int64_t* src = input.ptr<int64_t>(); int64_t* dst = output.ptr<int64_t>();
                for (size_t i = 0; i < input.total(); ++i) {
                    CV_Assert(src[i] != std::numeric_limits<int64_t>::min());
                    dst[i] = src[i] < 0 ? -src[i] : src[i];
                }
            } else CV_Error(cv::Error::StsUnsupportedFormat, "Mixed integer Abs input types");
        }
    }
    cv::Ptr<cv::dnn::BackendNode> initCUDA(void* context,
            const std::vector<cv::Ptr<cv::dnn::BackendWrapper>>& inputs,
            const std::vector<cv::Ptr<cv::dnn::BackendWrapper>>& outputs) override {
        return builtin_->initCUDA(context, inputs, outputs);
    }
private:
    cv::Ptr<cv::dnn::AbsLayer> builtin_;
};
cv::Ptr<cv::dnn::Layer> createAbs(cv::dnn::LayerParams& params) {
    return cv::makePtr<IntegerAbsLayer>(params);
}
#endif
// The exporter omits shape metadata on one If branch output. OpenCV 5 requires
// that field to exist before it performs its own shape inference. Add an empty
// shape placeholder in memory; pass all weights and operations through verbatim.
struct Field {
    unsigned number, wire;
    uint64_t integer = 0;
    std::string_view data, encoded;
};

uint64_t varint(std::string_view bytes, size_t& cursor) {
    uint64_t value = 0;
    for (unsigned shift = 0; shift < 64 && cursor < bytes.size(); shift += 7) {
        const auto byte = static_cast<unsigned char>(bytes[cursor++]);
        value |= uint64_t(byte & 127) << shift;
        if (!(byte & 128)) return value;
    }
    CV_Error(cv::Error::StsParseError, "Invalid ONNX protobuf varint");
}

std::vector<Field> fields(std::string_view bytes) {
    std::vector<Field> result;
    size_t cursor = 0;
    while (cursor < bytes.size()) {
        const size_t start = cursor;
        const auto tag = varint(bytes, cursor);
        Field f{static_cast<unsigned>(tag >> 3), static_cast<unsigned>(tag & 7)};
        size_t length = 0;
        if (f.wire == 0) f.integer = varint(bytes, cursor);
        else if (f.wire == 2) length = varint(bytes, cursor);
        else if (f.wire == 1) length = 8;
        else if (f.wire == 5) length = 4;
        else CV_Error(cv::Error::StsParseError, "Unsupported ONNX protobuf wire type");
        if (!f.number || length > bytes.size() - cursor)
            CV_Error(cv::Error::StsParseError, "Truncated ONNX protobuf field");
        f.data = bytes.substr(cursor, length);
        cursor += length;
        f.encoded = bytes.substr(start, cursor - start);
        result.push_back(f);
    }
    return result;
}

void appendVarint(std::string& out, uint64_t value) {
    while (value > 127) { out.push_back(static_cast<char>((value & 127) | 128)); value >>= 7; }
    out.push_back(static_cast<char>(value));
}
void appendBytes(std::string& out, unsigned number, std::string_view value) {
    appendVarint(out, (uint64_t(number) << 3) | 2);
    appendVarint(out, value.size()); out.append(value);
}
std::string_view bytesField(const std::vector<Field>& fs, unsigned number) {
    for (const auto& f : fs) if (f.number == number) return f.data;
    return {};
}
std::string completeOutputType(std::string_view output) {
    std::string result;
    for (const auto& f : fields(output)) {
        if (f.number != 2) { result.append(f.encoded); continue; }
        std::string type;
        for (const auto& t : fields(f.data)) {
            if (t.number != 1) { type.append(t.encoded); continue; }
            std::string tensorType(t.data);
            const auto tf = fields(t.data);
            const bool hasShape = std::any_of(tf.begin(), tf.end(), [](const Field& shape) { return shape.number == 2; });
            // OpenCV infers the actual output dimensions from the branch operations.
            if (!hasShape) appendBytes(tensorType, 2, "");
            appendBytes(type, 1, tensorType);
        }
        appendBytes(result, 2, type);
    }
    return result;
}

std::string normalizeGraph(std::string_view graph);
std::string normalizeBranches(std::string_view node) {
    std::string result;
    for (const auto& f : fields(node)) {
        if (f.number != 5) { result.append(f.encoded); continue; }
        std::string attribute;
        for (const auto& a : fields(f.data)) {
            if (a.number == 6) appendBytes(attribute, 6, normalizeGraph(a.data));
            else attribute.append(a.encoded);
        }
        appendBytes(result, 5, attribute);
    }
    return result;
}

std::string normalizeGraph(std::string_view graph) {
    const auto fs = fields(graph);
    std::string result;
    for (const auto& f : fs) {
        if (f.number == 1) {
            appendBytes(result, 1, normalizeBranches(f.data));
            continue;
        }
        if (f.number == 12) appendBytes(result, 12, completeOutputType(f.data));
        else result.append(f.encoded);
    }
    return result;
}
} // namespace

cv::dnn::Net readMedSAM2ONNX(const std::string& path) {
#if CV_VERSION_MAJOR >= 5
    static std::once_flag registered;
    std::call_once(registered, [] { cv::dnn::LayerFactory::registerLayer("AbsVal", createAbs); });
#endif
    std::ifstream file(path, std::ios::binary);
    if (!file) CV_Error(cv::Error::StsError, "Cannot open MedSAM2 ONNX model: " + path);
    const std::string source{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    std::string normalized;
    for (const auto& f : fields(source)) {
        if (f.number == 7) appendBytes(normalized, 7, normalizeGraph(f.data));
        else normalized.append(f.encoded);
    }
    return cv::dnn::readNetFromONNX(normalized.data(), normalized.size());
}
