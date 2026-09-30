#include "Document.h"
#include "XrayImage.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iterator>

namespace orthoseg {

bool Document::loadImage(const std::string& path) {
    try {
        cv::Mat original = cv::imread(path, cv::IMREAD_UNCHANGED);
        cv::Mat color = displayXray(original);
        if (color.empty()) return false;

        // Prepare all layers before replacing the current document, so a
        // decoder or processing failure leaves the previous annotations intact.
        cv::Mat gray;
        // Use the channel mean, as in the web reference, rather than luma weights.
        cv::transform(color, gray, cv::Matx13f(1.f / 3, 1.f / 3, 1.f / 3));
        cv::Mat edges = computeEdgeMap(gray);
        cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8UC1);
        cv::Mat channels = cv::Mat::zeros(gray.size(), CV_8UC3);
        cv::Mat seeds(gray.size(), CV_8UC1, cv::Scalar(kNoSeed));

        // Retain exact PNG bytes independently of the unchanged display path.
        std::vector<unsigned char> originalBytes;
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (file && file.tellg() > 0 && file.tellg() <= 32 * 1024 * 1024) {
            file.seekg(0);
            originalBytes.assign(std::istreambuf_iterator<char>(file), {});
            const unsigned char signature[] = {137, 80, 78, 71, 13, 10, 26, 10};
            if (originalBytes.size() < 8 || !std::equal(std::begin(signature), std::end(signature), originalBytes.begin()))
                originalBytes.clear();
        }
        originalPng_ = std::move(originalBytes);
        sourceOriginal_ = original;
        sourcePath_ = path;
        sourceColor_ = color;
        sourceGray_ = gray;
        edgeMap_ = edges;
        mask_ = mask;
        channels_ = channels;
        seeds_ = seeds;
        history_.clear();
        aiFill_ = AIFillState{};
        return true;
    } catch (const cv::Exception&) {
        return false;
    }
}

bool Document::exportMask(const std::string& path) const {
    if (channels_.empty()) return false;
    // PNG stores RGB channel values exactly: R=1, G=2, B=3. Lossy formats
    // would corrupt these discrete IDs, so never silently write one.
    const auto dot = path.find_last_of('.');
    if (dot == std::string::npos) return false;
    std::string ext = path.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    if (ext != ".png") return false;
    try {
        return cv::imwrite(path, channels_);
    } catch (const cv::Exception&) {
        // Missing/unsupported extensions and encoder failures can throw rather
        // than return false. Let the UI show its export error in either case.
        return false;
    }
}

bool Document::importMask(const std::string& path) {
    if (!hasImage()) return false;
    cv::Mat input;
    try { input = cv::imread(path, cv::IMREAD_UNCHANGED); }
    catch (const cv::Exception&) { return false; }
    if (input.empty() || input.size() != mask_.size() || input.depth() != CV_8U ||
        (input.channels() != 1 && input.channels() != 3 && input.channels() != 4)) return false;
    cv::Mat decoded = cv::Mat::zeros(input.size(), CV_8UC3);
    bool grayscale = input.channels() == 1;
    if (!grayscale) {
        grayscale = true;
        for (int y = 0; y < input.rows && grayscale; ++y) {
            for (int x = 0; x < input.cols; ++x) {
                const auto* p = input.ptr<uchar>(y) + x * input.channels();
                if (p[0] != p[1] || p[1] != p[2]) { grayscale = false; break; }
            }
        }
    }
    for (int y = 0; y < input.rows; ++y) {
        auto* out = decoded.ptr<cv::Vec3b>(y);
        for (int x = 0; x < input.cols; ++x) {
            const auto* p = input.ptr<uchar>(y) + x * input.channels();
            if (grayscale) {
                const uchar id = p[0];
                if (id > 3) return false;
                if (id) out[x][3 - id] = id;
            } else {
                if ((p[0] != 0 && p[0] != 3) || (p[1] != 0 && p[1] != 2) ||
                    (p[2] != 0 && p[2] != 1)) return false;
                out[x] = cv::Vec3b(p[0], p[1], p[2]);
            }
        }
    }
    pushHistory();
    channels_ = std::move(decoded);
    syncIndexed(cv::Rect(0, 0, width(), height()));
    return true;
}

void Document::syncIndexed(const cv::Rect& area) {
    for (int y = area.y; y < area.y + area.height; ++y) {
        const auto* src = channels_.ptr<cv::Vec3b>(y);
        auto* dst = mask_.ptr<uchar>(y);
        for (int x = area.x; x < area.x + area.width; ++x)
            dst[x] = src[x][0] ? 3 : src[x][1] ? 2 : src[x][2] ? 1 : 0;
    }
}

cv::Mat Document::labelMask(Label label) const {
    if (channels_.empty()) return {};
    cv::Mat out(channels_.size(), CV_8UC1, cv::Scalar(0));
    if (label == Label::Background) return mask_ == 0;
    const int channel = 3 - static_cast<int>(label);
    cv::extractChannel(channels_, out, channel);
    return out != 0;
}

cv::Mat Document::monaiAnatomyMask() const {
    if (channels_.empty()) return {};
    cv::Mat out = channels_.clone();
    cv::insertChannel(cv::Mat::zeros(channels_.size(), CV_8UC1), out, 0);
    return out;
}

void Document::applyRegion(const cv::Mat& region, Label label, bool eraseOnly) {
    CV_Assert(region.type() == CV_8UC1 && region.size() == channels_.size());
    const int id = static_cast<int>(label);
    for (int y = 0; y < region.rows; ++y) {
        const auto* r = region.ptr<uchar>(y);
        auto* dst = channels_.ptr<cv::Vec3b>(y);
        for (int x = 0; x < region.cols; ++x) {
            if (!r[x]) continue;
            if (id == 0) dst[x] = cv::Vec3b(0, 0, 0);
            else dst[x][3 - id] = eraseOnly ? 0 : static_cast<uchar>(id);
        }
    }
    syncIndexed(cv::Rect(0, 0, width(), height()));
}

bool Document::replaceAnatomyMask(const cv::Mat& mapped) {
    const int bg = static_cast<int>(Label::Background);
    const int femur = static_cast<int>(Label::Femur);
    const int tibia = static_cast<int>(Label::Tibia);
    if (!hasImage() || mapped.size() != mask_.size()) return false;
    if (mapped.type() == CV_8UC3) {
        std::vector<cv::Mat> channels;
        cv::split(mapped, channels);
        if (cv::countNonZero(channels[0]) || cv::countNonZero(channels[2] > 1) ||
            cv::countNonZero((channels[1] != 0) & (channels[1] != 2))) return false;
    } else if (mapped.type() != CV_8UC1 ||
        cv::countNonZero((mapped != bg) & (mapped != femur) & (mapped != tibia))) return false;
    pushHistory();
    for (int y = 0; y < height(); ++y) {
        auto* dst = channels_.ptr<cv::Vec3b>(y);
        const auto* src = mapped.ptr<uchar>(y);
        for (int x = 0; x < width(); ++x) {
            dst[x][2] = mapped.type() == CV_8UC3 ? mapped.at<cv::Vec3b>(y, x)[2] : src[x] == femur ? 1 : 0;
            dst[x][1] = mapped.type() == CV_8UC3 ? mapped.at<cv::Vec3b>(y, x)[1] : src[x] == tibia ? 2 : 0;
        }
    }
    syncIndexed(cv::Rect(0, 0, width(), height()));
    return true;
}

void Document::applyAIResult() {
    if (aiFill_.resultMask.empty()) return;
    const cv::Mat& result = aiFill_.resultMask;
    if (result.size() != mask_.size()) return;
    if (aiFill_.resultReplacesAnatomy) {
        if (!replaceAnatomyMask(result)) return;
    } else {
        if (result.type() != CV_8UC1 || cv::countNonZero(result > 3)) return;
        auto labels = aiFill_.resultLabels;
        if (labels.empty()) {
            for (int id = 1; id <= 3; ++id)
                if (cv::countNonZero(result == id)) labels.push_back(id);
        }
        for (int id : labels) if (id < 1 || id > 3) return;
        pushHistory();
        for (int y = 0; y < height(); ++y) {
            const auto* src = result.ptr<uchar>(y);
            auto* dst = channels_.ptr<cv::Vec3b>(y);
            for (int x = 0; x < width(); ++x)
                for (int id : labels)
                    dst[x][3 - id] = src[x] == id ? id : 0;
        }
        syncIndexed(cv::Rect(0, 0, width(), height()));
    }
    aiFill_.resultMask.release();
    aiFill_.resultLabels.clear();
    aiFill_.resultReplacesAnatomy = false;
}

void Document::paintAIPrompt(cv::Point a, cv::Point b, bool erase, int brushSize) {
    if (!hasImage()) return;
    if (aiFill_.promptMask.empty())
        aiFill_.promptMask = cv::Mat::zeros(sourceGray_.size(), CV_8UC1);
    const cv::Scalar value(erase ? 0 : 255);
    cv::line(aiFill_.promptMask, a, b, value, brushSize, cv::LINE_8);
    const int radius = std::max(0, brushSize / 2);
    cv::circle(aiFill_.promptMask, a, radius, value, cv::FILLED);
    cv::circle(aiFill_.promptMask, b, radius, value, cv::FILLED);
}

cv::Mat Document::edgeRegion(cv::Point seed, int threshold) const {
    if (!hasImage()) return {};
    cv::Mat region = edgeMap_ <= threshold;
    if (seed.x < 0 || seed.y < 0 || seed.x >= width() || seed.y >= height() ||
        !region.at<uchar>(seed)) return cv::Mat::zeros(edgeMap_.size(), CV_8UC1);
    cv::floodFill(region, seed, cv::Scalar(128), nullptr, cv::Scalar(), cv::Scalar(), 4);
    return region == 128;
}

void Document::paintLine(cv::Point a, cv::Point b, Label label, int brushSize, const cv::Mat& allowed) {
    stroke(a, b, label, brushSize, false, allowed);
}

void Document::eraseLabelLine(cv::Point a, cv::Point b, Label label, int brushSize, const cv::Mat& allowed) {
    stroke(a, b, label, brushSize, true, allowed);
}

void Document::stroke(cv::Point a, cv::Point b, Label label, int brushSize, bool eraseOnly, const cv::Mat& allowed) {
    if (channels_.empty()) return;
    const int radius = std::max(0, brushSize / 2);
    const int pad = radius + std::max(1, brushSize) + 1;
    const cv::Rect area = cv::Rect(std::min(a.x, b.x) - pad, std::min(a.y, b.y) - pad,
        std::abs(a.x - b.x) + 2 * pad + 1, std::abs(a.y - b.y) + 2 * pad + 1) &
        cv::Rect(0, 0, width(), height());
    if (area.empty()) return;
    cv::Mat strokeMask(area.size(), CV_8UC1, cv::Scalar(0));
    const cv::Point origin(area.x, area.y);
    cv::line(strokeMask, a - origin, b - origin, cv::Scalar(255), std::max(1, brushSize), cv::LINE_8);
    cv::circle(strokeMask, a - origin, radius, cv::Scalar(255), cv::FILLED);
    cv::circle(strokeMask, b - origin, radius, cv::Scalar(255), cv::FILLED);
    if (!allowed.empty()) cv::bitwise_and(strokeMask, allowed(area), strokeMask);
    const int id = static_cast<int>(label);
    for (int y = 0; y < area.height; ++y) {
        const auto* r = strokeMask.ptr<uchar>(y);
        auto* dst = channels_.ptr<cv::Vec3b>(area.y + y);
        for (int x = 0; x < area.width; ++x) {
            if (!r[x]) continue;
            auto& pixel = dst[area.x + x];
            if (id == 0) pixel = cv::Vec3b(0, 0, 0);
            else pixel[3 - id] = eraseOnly ? 0 : static_cast<uchar>(id);
        }
    }
    syncIndexed(area);
}

void Document::fillPolygon(const std::vector<cv::Point>& points, Label label, bool recordHistory, const cv::Mat& allowed) {
    if (channels_.empty() || points.size() < 3) return;
    cv::Mat region = cv::Mat::zeros(mask_.size(), CV_8UC1);
    std::vector<std::vector<cv::Point>> polygons{points};
    cv::fillPoly(region, polygons, cv::Scalar(255));
    if (!allowed.empty()) cv::bitwise_and(region, allowed, region);
    if (recordHistory) pushHistory();
    applyRegion(region, label);
}

bool Document::fillEnclosedHoles(Label label, int maxPixels) {
    if (channels_.empty() || label == Label::Background) return false;
    const auto active = labelMask(label);
    // Padding gives every image-border background region a common exterior seed.
    cv::Mat background;
    cv::copyMakeBorder(active == 0, background, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(255));
    // Diagonal connections to the exterior count as openings, not enclosed holes.
    cv::floodFill(background, {0, 0}, cv::Scalar(128), nullptr, cv::Scalar(), cv::Scalar(), 8);
    cv::Mat holes = background(cv::Rect(1, 1, width(), height())) == 255;
    if(maxPixels>0) {
        cv::Mat components,stats,centers;
        int count=cv::connectedComponentsWithStats(holes,components,stats,centers,8);
        for(int id=1;id<count;++id)if(stats.at<int>(id,cv::CC_STAT_AREA)>maxPixels)holes.setTo(0,components==id);
    }
    if (!cv::countNonZero(holes)) return false;
    pushHistory();
    applyRegion(holes, label);
    return true;
}

bool Document::fillEnclosedAt(cv::Point seed, Label label, const cv::Mat& allowed) {
    if (channels_.empty() || label == Label::Background || seed.x < 0 || seed.y < 0 ||
        seed.x >= width() || seed.y >= height()) return false;
    cv::Mat active = labelMask(label);
    if (active.at<uchar>(seed)) return false;
    cv::Mat region = active == 0;
    if (!allowed.empty()) cv::bitwise_and(region, allowed, region);
    if (!region.at<uchar>(seed)) return false;
    cv::floodFill(region, seed, cv::Scalar(128), nullptr, cv::Scalar(), cv::Scalar(), 4);
    if (cv::countNonZero(region.row(0) == 128) || cv::countNonZero(region.row(height() - 1) == 128) ||
        cv::countNonZero(region.col(0) == 128) || cv::countNonZero(region.col(width() - 1) == 128))
        return false;
    pushHistory();
    applyRegion(region == 128, label);
    return true;
}

void Document::fill(cv::Point seed, Label label, FillAlgorithm algo,
                    int intensityThreshold, int edgePenaltyThreshold) {
    if (mask_.empty()) return;
    cv::Mat region = cv::Mat::zeros(mask_.size(), CV_8UC1);
    const Label marker = label == Label::Background ? Label::Femur : label;
    switch (algo) {
        case FillAlgorithm::Standard:
            regionGrowStandard(sourceGray_, region, seed, marker,
                               intensityThreshold);
            break;
        case FillAlgorithm::EdgeEmbedded:
            regionGrowEdgeEmbedded(sourceGray_, region, seed, marker,
                                   intensityThreshold, edgePenaltyThreshold,
                                   edgeMap_);
            break;
        case FillAlgorithm::SplitMerge:
            regionGrowSplitMerge(sourceGray_, region, seed, marker,
                                 intensityThreshold, edgePenaltyThreshold,
                                 4, edgeMap_);
            break;
        default:
            return; // Scribble algorithms require runSeedSegmentation().
    }
    // Background uses a temporary foreground marker, then clears all channels.
    applyRegion(region, label);
}

void Document::clearMask() {
    if (mask_.empty()) return;
    channels_.setTo(cv::Scalar(0, 0, 0));
    mask_.setTo(cv::Scalar(0));
}

void Document::paintSeedLine(cv::Point a, cv::Point b, Label label, int brushSize) {
    if (seeds_.empty()) return;
    const uchar id = static_cast<uchar>(label); // Background (0) is a real seed
    cv::line(seeds_, a, b, cv::Scalar(id), brushSize, cv::LINE_8);
    int r = std::max(0, brushSize / 2);
    cv::circle(seeds_, a, r, cv::Scalar(id), cv::FILLED);
    cv::circle(seeds_, b, r, cv::Scalar(id), cv::FILLED);
}

void Document::clearSeeds() {
    if (seeds_.empty()) return;
    seeds_.setTo(cv::Scalar(kNoSeed));
}

bool Document::hasSeeds() const {
    if (seeds_.empty()) return false;
    return cv::countNonZero(seeds_ != kNoSeed) > 0;
}

bool Document::hasSeedForLabel(Label label) const {
    if (seeds_.empty()) return false;
    return cv::countNonZero(seeds_ == static_cast<uchar>(label)) > 0;
}

int Document::seedLabelCount() const {
    if (seeds_.empty()) return 0;
    int count = 0;
    for (int l = 0; l < 4; ++l)
        if (cv::countNonZero(seeds_ == static_cast<uchar>(l)) > 0) ++count;
    return count;
}

// Stamp seeds into the working grid so thin strokes are not skipped. Labels
// can collide; the caller checks for lost classes and restores the original
// hard constraints after upsampling the result.
static cv::Mat downscaleSeeds(const cv::Mat& seeds, cv::Size dst) {
    cv::Mat out(dst, CV_8UC1, cv::Scalar(kNoSeed));
    double sx = static_cast<double>(dst.width)  / seeds.cols;
    double sy = static_cast<double>(dst.height) / seeds.rows;
    for (int y = 0; y < seeds.rows; ++y) {
        const uchar* s = seeds.ptr<uchar>(y);
        for (int x = 0; x < seeds.cols; ++x) {
            if (s[x] == kNoSeed) continue;
            int ox = std::min(dst.width - 1,  static_cast<int>(x * sx));
            int oy = std::min(dst.height - 1, static_cast<int>(y * sy));
            out.at<uchar>(oy, ox) = s[x];
        }
    }
    return out;
}

bool Document::runSeedSegmentation(FillAlgorithm algo, Label foreground,
                                   double beta, const std::function<bool()>& cancelled, bool restrictToSeeds) {
    if (sourceGray_.empty() || seedLabelCount() < 2) return false;
    if (!isScribbleAlgorithm(algo)) return false;
    if (algo == FillAlgorithm::GraphCut &&
        (foreground == Label::Background || !hasSeedForLabel(foreground)))
        return false;

    // Cap the working resolution so the iterative solvers stay interactive.
    constexpr int kMaxWorkDim = 512;
    cv::Rect area(0,0,width(),height());
    if(restrictToSeeds) {
        const auto bounds=cv::boundingRect(seeds_ != kNoSeed);
        area=cv::Rect(bounds.x-32,bounds.y-32,bounds.width+64,bounds.height+64)&area;
    }
    const auto localGray=sourceGray_(area), localColor=sourceColor_(area), localSeeds=seeds_(area);
    const int w = area.width, h = area.height;
    const double scale = std::min(1.0, static_cast<double>(kMaxWorkDim) /
                                       std::max(w, h));
    const bool down = scale < 1.0;
    cv::Size ws(std::max(1, static_cast<int>(std::lround(w * scale))),
                std::max(1, static_cast<int>(std::lround(h * scale))));

    cv::Mat wgray, wcolor, wseeds, wout;
    if (down) {
        cv::resize(localGray,  wgray,  ws, 0, 0, cv::INTER_AREA);
        cv::resize(localColor, wcolor, ws, 0, 0, cv::INTER_AREA);
        wseeds = downscaleSeeds(localSeeds, ws);
        // Never silently run a different competition after an entire seed
        // class disappears into another label's working pixel.
        for (const auto& info : labels()) {
            if (hasSeedForLabel(info.id) &&
                cv::countNonZero(wseeds == static_cast<uchar>(info.id)) == 0)
                return false;
        }
    } else {
        wgray = localGray;
        wcolor = localColor;
        wseeds = localSeeds;
    }
    // Graph Cut produces a foreground selection here; merge it into the
    // original mask later so unrelated labels never take a resizing round trip.
    wout = cv::Mat::zeros(ws, CV_8UC1);

    bool ok = true;
    switch (algo) {
        case FillAlgorithm::GrowCut:
            growCutFromSeeds(wgray, wseeds, wout, beta, -1, cancelled);
            break;
        case FillAlgorithm::RandomWalker:
            randomWalkerFromSeeds(wgray, wseeds, wout, beta, 4000, 1e-8, cancelled);
            break;
        case FillAlgorithm::GraphCut:
            ok = graphCutFromSeeds(wcolor, wseeds, wout, foreground);
            break;
        default:
            return false;
    }
    if (!ok || (cancelled && cancelled())) return false;

    cv::Mat result;
    if (down)
        cv::resize(wout, result, area.size(), 0, 0, cv::INTER_NEAREST);
    else
        result = wout;

    cv::Mat fullResult=cv::Mat::zeros(sourceGray_.size(),CV_8U);
    result.copyTo(fullResult(area));result=fullResult;
    if (algo == FillAlgorithm::GraphCut) {
        const uchar fg = static_cast<uchar>(foreground);
        cv::Mat selected = result == fg;
        selected.setTo(0, (seeds_ != kNoSeed) & (seeds_ != fg));
        selected.setTo(255, seeds_ == fg);
        pushHistory();
        for (int y = area.y; y < area.y+area.height; ++y) {
            const auto* s = selected.ptr<uchar>(y);
            auto* dst = channels_.ptr<cv::Vec3b>(y);
            for (int x = area.x; x < area.x+area.width; ++x)
                dst[x][3 - fg] = s[x] ? fg : 0;
        }
    } else {
        // Working-grid collisions must not override explicit full-size seeds.
        seeds_.copyTo(result, seeds_ != kNoSeed);
        const bool updateFemur = hasSeedForLabel(Label::Femur);
        const bool updateTibia = hasSeedForLabel(Label::Tibia);
        const bool updateFibula = hasSeedForLabel(Label::Fibula);
        pushHistory();
        for (int y = area.y; y < area.y+area.height; ++y) {
            const auto* src = result.ptr<uchar>(y);
            auto* dst = channels_.ptr<cv::Vec3b>(y);
            for (int x = area.x; x < area.x+area.width; ++x) {
                if (updateFemur) dst[x][2] = src[x] == 1 ? 1 : 0;
                if (updateTibia) dst[x][1] = src[x] == 2 ? 2 : 0;
                if (updateFibula) dst[x][0] = src[x] == 3 ? 3 : 0;
            }
        }
    }
    syncIndexed(cv::Rect(0, 0, width(), height()));
    return true;
}

Document Document::segmentationCopy() const {
    Document copy = *this;
    copy.channels_ = channels_.clone(); copy.mask_ = mask_.clone(); copy.seeds_ = seeds_.clone();
    copy.history_.clear(); copy.aiFill_ = AIFillState{};
    return copy;
}
void Document::applySegmentationChannels(const cv::Mat& channels) {
    CV_Assert(channels.type()==CV_8UC3 && channels.size()==channels_.size());
    pushHistory(); channels_=channels.clone();syncIndexed(cv::Rect(0,0,width(),height()));
}

void Document::pushHistory() {
    if (mask_.empty()) return;
    history_.push_back({channels_.clone(), seeds_.clone()});
    while (history_.size() > kMaxHistory) history_.pop_front();
}

void Document::undo() {
    if (history_.empty()) return;
    // Each snapshot is the state captured just before a mutation; restoring the
    // most recent one returns mask + seeds to before the last edit.
    Snapshot s = history_.back();
    history_.pop_back();
    s.channels.copyTo(channels_);
    syncIndexed(cv::Rect(0, 0, width(), height()));
    s.seeds.copyTo(seeds_);
}

} // namespace orthoseg
