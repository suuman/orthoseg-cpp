#include "AIFillController.h"
#include <QMetaObject>
#ifdef ORTHOSEG_NATIVE_NNUNET
#include "segmentor.hpp"
#endif

namespace orthoseg {
namespace {
struct InferenceWorker : QObject {
    MedSAM2Inference inference;
#ifdef ORTHOSEG_NATIVE_NNUNET
    std::unique_ptr<xray_ocv::XRaySegmentor> native;
    std::string nativePath;
    std::string nativeDevice;
#endif
};
}

AIFillController::AIFillController(QObject* parent) : QObject(parent) {
    worker_ = new InferenceWorker;
    worker_->moveToThread(&thread_);
    connect(&thread_, &QThread::finished, worker_, &QObject::deleteLater);
    thread_.start();
}

AIFillController::~AIFillController() {
    // CUDA forward cannot safely be interrupted; let it finish before freeing
    // the worker/model. Queued UI callbacks are removed when QObject is destroyed.
    thread_.quit();
    thread_.wait();
}

bool AIFillController::run(AIFillRequest request) {
    if (running_) return false;
    running_ = true;
    QMetaObject::invokeMethod(worker_, [this, request = std::move(request)] {
        cv::Mat result;
        QString error;
        try {
            result = static_cast<InferenceWorker*>(worker_)->inference.run(request);
        } catch (const std::exception& e) {
            error = QString::fromUtf8(e.what());
        } catch (...) {
            error = "Unknown MedSAM2 inference failure.";
        }
        QMetaObject::invokeMethod(this, [this, result, error] {
            running_ = false;
            if (error.isEmpty()) emit completed(result);
            else emit failed(error);
        }, Qt::QueuedConnection);
    }, Qt::QueuedConnection);
    return true;
}

#ifdef ORTHOSEG_NATIVE_NNUNET
bool AIFillController::runNative(NativeNnUnetRequest request) {
    if (running_) return false;
    running_ = true;
    QMetaObject::invokeMethod(worker_, [this, request = std::move(request)] {
        cv::Mat result;
        QString error;
        try {
            auto* worker = static_cast<InferenceWorker*>(worker_);
            if (!worker->native || worker->nativePath != request.modelPath ||
                worker->nativeDevice != request.device) {
                worker->native = xray_ocv::XRaySegmentor::create(request.modelPath, request.device,
                                                                  0.4f, false);
                worker->nativePath = request.modelPath;
                worker->nativeDevice = request.device;
            }
            result = worker->native->predict(request.imageBGR).label_mask;
        } catch (const std::exception& e) {
            error = QString::fromUtf8(e.what());
        } catch (...) {
            error = "Unknown native nnUNet inference failure.";
        }
        QMetaObject::invokeMethod(this, [this, result, error] {
            running_ = false;
            if (error.isEmpty()) emit nativeCompleted(result);
            else emit nativeFailed(error);
        }, Qt::QueuedConnection);
    }, Qt::QueuedConnection);
    return true;
}
#endif

} // namespace orthoseg
