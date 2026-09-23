#pragma once
#include "MedSAM2Inference.h"
#include <QObject>
#include <QThread>

namespace orthoseg {

#ifdef ORTHOSEG_NATIVE_NNUNET
struct NativeNnUnetRequest {
    cv::Mat imageBGR;
    std::string modelPath;
    std::string device = "auto";
};
#endif

class AIFillController : public QObject {
    Q_OBJECT
public:
    explicit AIFillController(QObject* parent = nullptr);
    ~AIFillController() override;
    bool running() const { return running_; }
    bool run(AIFillRequest request);
#ifdef ORTHOSEG_NATIVE_NNUNET
    bool runNative(NativeNnUnetRequest request);
#endif
signals:
    void completed(const cv::Mat& mask);
    void failed(const QString& message);
#ifdef ORTHOSEG_NATIVE_NNUNET
    void nativeCompleted(const cv::Mat& mask);
    void nativeFailed(const QString& message);
#endif
private:
    QThread thread_;
    QObject* worker_ = nullptr;
    bool running_ = false; // Accessed only on the UI thread.
};

} // namespace orthoseg
