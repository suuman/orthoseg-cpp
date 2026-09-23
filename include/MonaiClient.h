#pragma once
#include <QObject>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QUrl>
#include <opencv2/core.hpp>
#include <functional>

namespace orthoseg {
struct MonaiLabels { int background = 0; int femur = 1; int tibia = 2; };
cv::Mat decodeMonaiMask(const QByteArray& png, cv::Size expected);
cv::Mat fromMonaiLabels(const cv::Mat& canonical, MonaiLabels labels);
cv::Mat toMonaiLabels(const cv::Mat& current, MonaiLabels labels);
QByteArray encodeMonaiMask(const cv::Mat& canonical);
QByteArray monaiCaseId(const QByteArray& original);
class MonaiClient : public QObject {
public:
    using JsonCallback = std::function<void(const QJsonObject&, const QString&)>;
    using SegmentCallback = std::function<void(const cv::Mat&, const QString&, const QString&)>;
    explicit MonaiClient(QObject* parent = nullptr, QUrl base = configuredUrl(), int timeoutMs = 180000);
    static QUrl configuredUrl();
    void health(JsonCallback done);
    void managementStatus(JsonCallback done);
    void startTraining(JsonCallback done);
    void trainingJob(const QString& id, JsonCallback done);
    void promoteCandidate(const QString& version, JsonCallback done);
    bool segment(const QByteArray& original, cv::Size expected, SegmentCallback done);
    bool submitTrainingCase(const QByteArray& original, const cv::Mat& currentCanonical,
                            const QString& filename, const QString& version, JsonCallback done);
    bool segmentRunning() const { return segmentRunning_; }
    bool uploadRunning() const { return uploadRunning_; }
private:
    using ReplyCallback = std::function<void(const QByteArray&, const QByteArray&, const QString&)>;
    void request(const QString& path, const QByteArray& original, const QByteArray& mask,
                 const QString& filename, const QString& version, ReplyCallback done, const QByteArray& jsonBody = {});
    void adminRequest(const QString& path, bool post, JsonCallback done);
    QNetworkAccessManager network_;
    QUrl base_;
    int timeoutMs_;
    bool segmentRunning_ = false;
    bool uploadRunning_ = false;
};
} // namespace orthoseg
