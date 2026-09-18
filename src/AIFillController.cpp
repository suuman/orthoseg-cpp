#include "AIFillController.h"
#include <QMetaObject>

namespace orthoseg {
namespace {
struct InferenceWorker : QObject {
    MedSAM2Inference inference;
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

} // namespace orthoseg
