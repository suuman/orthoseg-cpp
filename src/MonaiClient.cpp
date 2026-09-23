#include "MonaiClient.h"
#include <QCryptographicHash>
#include <QHttpMultiPart>
#include <QJsonDocument>
#include <QNetworkProxy>
#include <QNetworkReply>
#include <QTimer>
#include <QRegularExpression>
#include <QtEndian>
#include <opencv2/imgcodecs.hpp>
#include <memory>
#include <algorithm>
#include <stdexcept>

namespace orthoseg {
namespace {
void validate(const cv::Mat& mask) {
    if (mask.empty() || mask.type() != CV_8UC1 || cv::countNonZero(mask > 2))
        throw std::runtime_error("MONAI mask must be single-channel uint8 with only labels 0, 1, 2.");
}
void validateLabels(MonaiLabels l) {
    if (l.background < 0 || l.background > 255 || l.femur < 0 || l.femur > 255 ||
        l.tibia < 0 || l.tibia > 255 || l.background == l.femur ||
        l.background == l.tibia || l.femur == l.tibia)
        throw std::runtime_error("AI Segment requires distinct Background, Femur and Tibia labels.");
}
QJsonObject parseJson(const QByteArray& bytes) {
    QJsonParseError error;
    auto json = QJsonDocument::fromJson(bytes, &error);
    if (error.error != QJsonParseError::NoError || !json.isObject())
        throw std::runtime_error("MONAI returned invalid JSON.");
    return json.object();
}
void validateOriginal(const QByteArray& bytes) {
    if (!bytes.startsWith(QByteArray::fromHex("89504e470d0a1a0a")) || bytes.size() > 32 * 1024 * 1024)
        throw std::runtime_error("MONAI requires the original PNG (up to 32 MiB). Open a PNG X-ray first.");
}
}
cv::Mat decodeMonaiMask(const QByteArray& png, cv::Size expected) {
    // Check IHDR first: never silently convert RGB, palette or 16-bit labels.
    if (png.size() < 33 || !png.startsWith(QByteArray::fromHex("89504e470d0a1a0a")) ||
        png.mid(12, 4) != "IHDR" || static_cast<unsigned char>(png[24]) != 8 || png[25] != 0)
        throw std::runtime_error("MONAI response is not an 8-bit grayscale class-ID PNG.");
    auto width = qFromBigEndian<quint32>(reinterpret_cast<const uchar*>(png.constData() + 16));
    auto height = qFromBigEndian<quint32>(reinterpret_cast<const uchar*>(png.constData() + 20));
    if (expected.width <= 0 || expected.height <= 0 || width != quint32(expected.width) || height != quint32(expected.height))
        throw std::runtime_error("MONAI mask dimensions do not match the original X-ray. No annotation was changed.");
    std::vector<uchar> bytes(png.begin(), png.end());
    auto mask = cv::imdecode(bytes, cv::IMREAD_UNCHANGED);
    validate(mask);
    if (mask.size() != expected) throw std::runtime_error("Decoded mask dimensions differ from the X-ray.");
    return mask;
}
cv::Mat fromMonaiLabels(const cv::Mat& canonical, MonaiLabels labels) {
    validate(canonical);
    validateLabels(labels);
    cv::Mat mapped(canonical.size(), CV_8UC1, cv::Scalar(labels.background));
    mapped.setTo(labels.femur, canonical == 1);
    mapped.setTo(labels.tibia, canonical == 2);
    return mapped;
}
cv::Mat toMonaiLabels(const cv::Mat& current, MonaiLabels labels) {
    validateLabels(labels);
    if (current.empty() || current.type() != CV_8UC1)
        throw std::runtime_error("Current annotation must be a single-channel uint8 mask.");
    cv::Mat canonical(current.size(), CV_8UC1, cv::Scalar(0));
    canonical.setTo(1, current == labels.femur);
    canonical.setTo(2, current == labels.tibia);
    return canonical;
}
QByteArray encodeMonaiMask(const cv::Mat& canonical) {
    validate(canonical);
    std::vector<uchar> bytes;
    if (!cv::imencode(".png", canonical, bytes)) throw std::runtime_error("Could not encode training mask.");
    return QByteArray(reinterpret_cast<const char*>(bytes.data()), qsizetype(bytes.size()));
}
QByteArray monaiCaseId(const QByteArray& original) {
    return QCryptographicHash::hash(original, QCryptographicHash::Sha256).toHex();
}
QUrl MonaiClient::configuredUrl() {
    return QUrl(qEnvironmentVariable("MONAI_BACKEND_URL", "http://127.0.0.1:8000"));
}
MonaiClient::MonaiClient(QObject* parent, QUrl base, int timeoutMs)
    : QObject(parent), base_(std::move(base)), timeoutMs_(timeoutMs) {
    network_.setProxy(QNetworkProxy::NoProxy);
}
void MonaiClient::request(const QString& path, const QByteArray& original, const QByteArray& mask,
                          const QString& filename, const QString& version, ReplyCallback done, const QByteArray& jsonBody,
                          const QByteArray& femurBox, const QByteArray& tibiaBox) {
    const auto host = base_.host().toLower();
    if (!base_.isValid() || base_.scheme() != "http" ||
        (host != "127.0.0.1" && host != "localhost" && host != "::1") ||
        !base_.userInfo().isEmpty() || base_.hasQuery() || base_.hasFragment()) {
        QTimer::singleShot(0, this, [done] { done({}, {}, "MONAI_BACKEND_URL must be a local http URL."); });
        return;
    }
    auto url = base_;
    QString prefix = url.path();
    while (prefix.endsWith('/')) prefix.chop(1);
    url.setPath(prefix + path);
    QNetworkRequest req(url);
    req.setAttribute(QNetworkRequest::RedirectPolicyAttribute, QNetworkRequest::ManualRedirectPolicy);
    const int deadline = (path == "/health" || path == "/management/status") ?
        std::min(timeoutMs_, 5000) : timeoutMs_;
    req.setTransferTimeout(deadline);
    QNetworkReply* reply;
    if (!jsonBody.isNull()) {
        req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
        reply = network_.post(req, jsonBody);
    } else if (original.isEmpty()) reply = network_.get(req);
    else {
        auto* multipart = new QHttpMultiPart(QHttpMultiPart::FormDataType);
        auto part = [multipart](const char* name, const QByteArray& data, bool file) {
            QHttpPart p;
            p.setHeader(QNetworkRequest::ContentDispositionHeader,
                        QString("form-data; name=\"%1\"%2").arg(name, file ? "; filename=\"input.png\"" : ""));
            if (file) p.setHeader(QNetworkRequest::ContentTypeHeader, "image/png");
            p.setBody(data);
            multipart->append(p);
        };
        part("image", original, true);
        part("case_id", monaiCaseId(original), false);
        if (!femurBox.isEmpty()) part("femur_box", femurBox, false);
        if (!tibiaBox.isEmpty()) part("tibia_box", tibiaBox, false);
        if (!mask.isEmpty()) {
            part("mask", mask, true);
            part("original_filename", filename.toUtf8(), false);
            if (!version.isEmpty()) part("model_version", version.toUtf8(), false);
        }
        reply = network_.post(req, multipart);
        multipart->setParent(reply);
    }
    auto* timer = new QTimer(reply);
    timer->setSingleShot(true);
    connect(timer, &QTimer::timeout, reply, [reply] {
        reply->setProperty("monaiTimeout", true);
        reply->abort();
    });
    timer->start(deadline);
    auto body = std::make_shared<QByteArray>();
    connect(reply, &QNetworkReply::readyRead, reply, [reply, body] {
        body->append(reply->readAll());
        if (body->size() > 32 * 1024 * 1024) {
            reply->setProperty("monaiOversize", true);
            reply->abort();
        }
    });
    connect(reply, &QNetworkReply::finished, this, [reply, timer, body, done, path] {
        timer->stop();
        body->append(reply->readAll());
        QString error;
        const int status = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
        if (reply->property("monaiTimeout").toBool()) error = "MONAI request timed out. Try again when the local service is ready.";
        else if (reply->property("monaiOversize").toBool()) error = "MONAI response exceeds the size limit.";
        else if (status && (status < 200 || status >= 300)) {
            error = QString("MONAI HTTP %1").arg(status);
            const auto detail = QJsonDocument::fromJson(*body).object().value("detail");
            if (detail.isString()) error += ": " + detail.toString().left(1000);
        } else if (reply->error() != QNetworkReply::NoError) {
            error = "MONAI AI Segment service is unavailable. Start the local MONAI backend and try again.\n" + reply->errorString();
        } else if ((path == "/segment" || path == "/segment/prompted" || path == "/segment/nnunet") &&
                   reply->header(QNetworkRequest::ContentTypeHeader).toString().section(';', 0, 0).trimmed() != "image/png") {
            error = "MONAI did not return image/png.";
        }
        const auto versionHeader = reply->rawHeader("X-Model-Version");
        reply->deleteLater();
        done(*body, versionHeader, error);
    });
}
void MonaiClient::adminRequest(const QString& path, bool post, JsonCallback done) {
    request(path, {}, {}, {}, {}, [done](const QByteArray& bytes, const QByteArray&, const QString& error) {
        if (!error.isEmpty()) { done({}, error); return; }
        QJsonObject result;
        try { result = parseJson(bytes); }
        catch (const std::exception& e) { done({}, QString::fromUtf8(e.what())); return; }
        done(result, {});
    }, post ? QByteArray("{}") : QByteArray());
}
void MonaiClient::managementStatus(JsonCallback done) { adminRequest("/management/status", false, std::move(done)); }
void MonaiClient::startTraining(JsonCallback done) { adminRequest("/training/start", true, std::move(done)); }
void MonaiClient::trainingJob(const QString& id, JsonCallback done) {
    if (!QRegularExpression("^[a-f0-9]{32}$").match(id).hasMatch()) { done({}, "Invalid job identifier."); return; }
    adminRequest("/training/jobs/" + id, false, std::move(done));
}
void MonaiClient::promoteCandidate(const QString& version, JsonCallback done) {
    if (!QRegularExpression("^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$").match(version).hasMatch()) { done({}, "Invalid candidate version."); return; }
    adminRequest("/models/" + version + "/promote", true, std::move(done));
}

void MonaiClient::health(JsonCallback done) {
    request("/health", {}, {}, {}, {}, [done](const QByteArray& bytes, const QByteArray&, const QString& error) {
        if (!error.isEmpty()) { done({}, error); return; }
        QJsonObject result;
        try { result = parseJson(bytes); }
        catch (const std::exception& e) { done({}, QString::fromUtf8(e.what())); return; }
        done(result, {});
    });
}
bool MonaiClient::segment(const QByteArray& original, cv::Size expected, SegmentCallback done) {
    if (segmentRunning_) return false;
    validateOriginal(original);
    segmentRunning_ = true;
    request("/segment", original, {}, {}, {}, [this, expected, done](const QByteArray& bytes, const QByteArray& version, const QString& error) {
        segmentRunning_ = false;
        if (!error.isEmpty()) { done({}, {}, error); return; }
        cv::Mat mask;
        try { mask = decodeMonaiMask(bytes, expected); }
        catch (const std::exception& e) { done({}, {}, QString::fromUtf8(e.what())); return; }
        done(mask, QString::fromUtf8(version), {});
    });
    return true;
}
bool MonaiClient::segmentNnUnet(const QByteArray& original, cv::Size expected, SegmentCallback done) {
    if (segmentRunning_) return false;
    validateOriginal(original);
    segmentRunning_ = true;
    request("/segment/nnunet", original, {}, {}, {},
        [this, expected, done](const QByteArray& bytes, const QByteArray& version, const QString& error) {
            segmentRunning_ = false;
            if (!error.isEmpty()) { done({}, {}, error); return; }
            try { done(decodeMonaiMask(bytes, expected), QString::fromUtf8(version), {}); }
            catch (const std::exception& e) { done({}, {}, QString::fromUtf8(e.what())); }
        });
    return true;
}
bool MonaiClient::segmentPrompted(const QByteArray& original, cv::Size expected, const QJsonArray& femurBox,
                                  const QJsonArray& tibiaBox, SegmentCallback done) {
    if (segmentRunning_) return false;
    validateOriginal(original);
    if (femurBox.isEmpty() && tibiaBox.isEmpty())
        throw std::runtime_error("Draw a Femur or Tibia box before running MONAI SAM2.");
    segmentRunning_ = true;
    request("/segment/prompted", original, {}, {}, {},
        [this, expected, done](const QByteArray& bytes, const QByteArray& version, const QString& error) {
            segmentRunning_ = false;
            if (!error.isEmpty()) { done({}, {}, error); return; }
            try { done(decodeMonaiMask(bytes, expected), QString::fromUtf8(version), {}); }
            catch (const std::exception& e) { done({}, {}, QString::fromUtf8(e.what())); }
        }, {}, femurBox.isEmpty() ? QByteArray() : QJsonDocument(femurBox).toJson(QJsonDocument::Compact),
            tibiaBox.isEmpty() ? QByteArray() : QJsonDocument(tibiaBox).toJson(QJsonDocument::Compact));
    return true;
}
bool MonaiClient::submitTrainingCase(const QByteArray& original, const cv::Mat& canonical,
                                    const QString& filename, const QString& version, JsonCallback done) {
    if (uploadRunning_) return false;
    validateOriginal(original);
    auto png = encodeMonaiMask(canonical);
    if (original.size() < 24 || qFromBigEndian<quint32>(reinterpret_cast<const uchar*>(original.constData()+16)) != quint32(canonical.cols) ||
        qFromBigEndian<quint32>(reinterpret_cast<const uchar*>(original.constData()+20)) != quint32(canonical.rows))
        throw std::runtime_error("Training mask dimensions do not match the original PNG.");
    uploadRunning_ = true;
    request("/training/cases", original, png, filename, version,
        [this, done](const QByteArray& bytes, const QByteArray&, const QString& error) {
            uploadRunning_ = false;
            if (!error.isEmpty()) { done({}, error); return; }
            QJsonObject result;
            try {
                result = parseJson(bytes);
                if (result.value("status").toString() != "accepted") throw std::runtime_error("MONAI did not confirm acceptance of the case.");
            } catch (const std::exception& e) { done({}, QString::fromUtf8(e.what())); return; }
            done(result, {});
        });
    return true;
}
} // namespace orthoseg
