#pragma once
#include "MedSAM2Inference.h"
#include <QObject>
#include <QThread>

namespace orthoseg {

class AIFillController : public QObject {
    Q_OBJECT
public:
    explicit AIFillController(QObject* parent = nullptr);
    ~AIFillController() override;
    bool running() const { return running_; }
    bool run(AIFillRequest request);
signals:
    void completed(const cv::Mat& mask);
    void failed(const QString& message);
private:
    QThread thread_;
    QObject* worker_ = nullptr;
    bool running_ = false; // Accessed only on the UI thread.
};

} // namespace orthoseg
