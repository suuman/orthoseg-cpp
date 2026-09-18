#include "CanvasWidget.h"
#include <QPainter>
#include <QMouseEvent>
#include <QWheelEvent>
#include <algorithm>
#include <cmath>

namespace orthoseg {

CanvasWidget::CanvasWidget(Document* doc, QWidget* parent)
    : QWidget(parent), doc_(doc) {
    setMouseTracking(true);
    setCursor(Qt::CrossCursor);
    setMinimumSize(400, 400);
}

void CanvasWidget::rebuildSourceImage() {
    const cv::Mat& c = doc_->sourceColor();
    if (c.empty()) { sourceQt_ = QImage(); return; }
    // OpenCV is BGR; QImage::Format_RGB888 expects RGB. Copy with swap.
    QImage img(c.cols, c.rows, QImage::Format_RGB888);
    for (int y = 0; y < c.rows; ++y) {
        const cv::Vec3b* srow = c.ptr<cv::Vec3b>(y);
        uchar* drow = img.scanLine(y);
        for (int x = 0; x < c.cols; ++x) {
            drow[x * 3 + 0] = srow[x][2];
            drow[x * 3 + 1] = srow[x][1];
            drow[x * 3 + 2] = srow[x][0];
        }
    }
    sourceQt_ = img;
}

void CanvasWidget::refresh() {
    rebuildSourceImage();
    update();
}

QRectF CanvasWidget::imageRect() const {
    if (!doc_->hasImage()) return QRectF();
    const float iw = doc_->width(), ih = doc_->height();
    // Fit-to-widget base scale, then apply zoom, centered.
    float base = std::min(width() / iw, height() / ih);
    float scale = base * zoom_;
    float dw = iw * scale, dh = ih * scale;
    return QRectF((width() - dw) / 2.f + panOffset_.x(),
                  (height() - dh) / 2.f + panOffset_.y(), dw, dh);
}

QPoint CanvasWidget::widgetToImage(const QPointF& p) const {
    QRectF r = imageRect();
    if (r.width() <= 0) return {-1, -1};
    double fx = (p.x() - r.left()) / r.width() * doc_->width();
    double fy = (p.y() - r.top()) / r.height() * doc_->height();
    return QPoint(static_cast<int>(std::floor(fx)),
                  static_cast<int>(std::floor(fy)));
}

void CanvasWidget::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.fillRect(rect(), QColor(15, 23, 42)); // medical-dark background

    if (!doc_->hasImage() || sourceQt_.isNull()) {
        p.setPen(QColor(100, 116, 139));
        p.drawText(rect(), Qt::AlignCenter,
                   "Upload an X-ray to begin segmentation");
        return;
    }

    QRectF dst = imageRect();
    p.setRenderHint(QPainter::SmoothPixmapTransform, false);
    p.drawImage(dst, sourceQt_);

    // Build the color overlay from the indexed mask on the fly.
    const cv::Mat& mask = doc_->mask();
    QImage overlay(mask.cols, mask.rows, QImage::Format_ARGB32);
    overlay.fill(Qt::transparent);
    const int a = static_cast<int>(opacity_ * 255);
    for (int y = 0; y < mask.rows; ++y) {
        const uchar* mrow = mask.ptr<uchar>(y);
        QRgb* orow = reinterpret_cast<QRgb*>(overlay.scanLine(y));
        for (int x = 0; x < mask.cols; ++x) {
            uchar id = mrow[x];
            if (id == 0) { orow[x] = qRgba(0, 0, 0, 0); continue; }
            cv::Vec3b bgr = labelInfo(static_cast<Label>(id)).colorBGR;
            orow[x] = qRgba(bgr[2], bgr[1], bgr[0], a);
        }
    }
    p.drawImage(dst, overlay);

    const auto& ai = doc_->aiFill();
    if (ai.showResult && !ai.resultMask.empty()) {
        overlay.fill(Qt::transparent);
        for (int y = 0; y < ai.resultMask.rows; ++y) {
            auto* row = reinterpret_cast<QRgb*>(overlay.scanLine(y));
            const auto* maskRow = ai.resultMask.ptr<uchar>(y);
            for (int x = 0; x < ai.resultMask.cols; ++x) {
                if (maskRow[x] == 0) continue;
                const auto bgr = labelInfo(static_cast<Label>(maskRow[x])).colorBGR;
                row[x] = qRgba(bgr[2], bgr[1], bgr[0], a);
            }
        }
        p.drawImage(dst, overlay);
    }
    if (tool_ == Tool::AIFill && ai.promptType != AIFillPromptType::BoundingBox && !ai.promptMask.empty()) {
        overlay.fill(Qt::transparent);
        for (int y = 0; y < ai.promptMask.rows; ++y) {
            auto* row = reinterpret_cast<QRgb*>(overlay.scanLine(y));
            const auto* maskRow = ai.promptMask.ptr<uchar>(y);
            for (int x = 0; x < ai.promptMask.cols; ++x)
                if (maskRow[x]) row[x] = qRgba(34, 211, 238, 150);
        }
        p.drawImage(dst, overlay);
    }
    if (tool_ == Tool::AIFill && ai.promptType == AIFillPromptType::BoundingBox && ai.box) {
        const Box& b = *ai.box;
        const double sx = dst.width() / doc_->width(), sy = dst.height() / doc_->height();
        p.setPen(QPen(QColor(250, 204, 21), 2, Qt::DashLine));
        p.drawRect(QRectF(QPointF(dst.left() + b.x0 * sx, dst.top() + b.y0 * sy),
                          QPointF(dst.left() + b.x1 * sx, dst.top() + b.y1 * sy)).normalized());
    }

    // Seed overlay: shown while scribbling seeds (Fill tool + a competition
    // algorithm). Drawn opaque so scribbles stand out over the translucent
    // result. Background seeds (id 0, otherwise invisible) use a slate color.
    if (tool_ == Tool::Fill && isScribbleAlgorithm(fillAlgo_)) {
        const cv::Mat& seeds = doc_->seeds();
        if (!seeds.empty()) {
            QImage sov(seeds.cols, seeds.rows, QImage::Format_ARGB32);
            sov.fill(Qt::transparent);
            for (int y = 0; y < seeds.rows; ++y) {
                const uchar* srow = seeds.ptr<uchar>(y);
                QRgb* orow = reinterpret_cast<QRgb*>(sov.scanLine(y));
                for (int x = 0; x < seeds.cols; ++x) {
                    uchar id = srow[x];
                    if (id == kNoSeed) { orow[x] = qRgba(0, 0, 0, 0); continue; }
                    if (id == 0) { orow[x] = qRgba(148, 163, 184, 255); continue; }
                    cv::Vec3b bgr = labelInfo(static_cast<Label>(id)).colorBGR;
                    orow[x] = qRgba(bgr[2], bgr[1], bgr[0], 255);
                }
            }
            p.drawImage(dst, sov);
        }
    }
}

void CanvasWidget::mousePressEvent(QMouseEvent* e) {
    if (doc_->hasImage() && e->button() == Qt::MiddleButton) {
        panning_ = true;
        lastPanPos_ = e->position();
        return;
    }
    if (!doc_->hasImage() || e->button() != Qt::LeftButton) return;
    QPoint ip = widgetToImage(e->position());
    if (ip.x() < 0 || ip.y() < 0 ||
        ip.x() >= doc_->width() || ip.y() >= doc_->height()) return;

    if (tool_ == Tool::AIFill) {
        if (doc_->aiFill().promptType == AIFillPromptType::LoadedMask) return;
        drawing_ = true;
        if (doc_->aiFill().promptType == AIFillPromptType::BoundingBox) {
            boxStart_ = ip;
            doc_->aiFill().box = Box{float(ip.x()), float(ip.y()), float(ip.x()), float(ip.y())};
        } else {
            lastImgPt_ = ip;
            doc_->paintAIPrompt({ip.x(), ip.y()}, {ip.x(), ip.y()}, aiPromptErase_, brushSize_);
        }
        update();
        return;
    }

    if (tool_ == Tool::Fill && isScribbleAlgorithm(fillAlgo_)) {
        // Scribble a seed stroke; the actual segmentation runs on demand.
        doc_->pushHistory();
        drawing_ = true;
        lastImgPt_ = ip;
        doc_->paintSeedLine(cv::Point(ip.x(), ip.y()), cv::Point(ip.x(), ip.y()),
                            label_, brushSize_);
        update();
        return;
    }

    if (tool_ == Tool::Fill) {
        doc_->pushHistory();
        doc_->fill(cv::Point(ip.x(), ip.y()), label_, fillAlgo_,
                   intensityThreshold_, edgePenalty_);
        emit maskChanged();
        update();
        return;
    }

    // Brush / Eraser: snapshot once at gesture start.
    doc_->pushHistory();
    drawing_ = true;
    lastImgPt_ = ip;
    Label paintLabel = (tool_ == Tool::Eraser) ? Label::Background : label_;
    doc_->paintLine(cv::Point(ip.x(), ip.y()), cv::Point(ip.x(), ip.y()),
                    paintLabel, brushSize_);
    update();
}

void CanvasWidget::mouseMoveEvent(QMouseEvent* e) {
    if (panning_) {
        panOffset_ += e->position() - lastPanPos_;
        lastPanPos_ = e->position();
        update();
        return;
    }
    if (!drawing_) return;
    QPoint ip = widgetToImage(e->position());
    if (tool_ == Tool::AIFill) {
        if (doc_->aiFill().promptType == AIFillPromptType::BoundingBox) {
            ip.setX(std::clamp(ip.x(), 0, doc_->width() - 1));
            ip.setY(std::clamp(ip.y(), 0, doc_->height() - 1));
            doc_->aiFill().box = Box{float(std::min(boxStart_.x(), ip.x())),
                float(std::min(boxStart_.y(), ip.y())), float(std::max(boxStart_.x(), ip.x())),
                float(std::max(boxStart_.y(), ip.y()))};
        } else if (doc_->aiFill().promptType == AIFillPromptType::PaintedMask) {
            doc_->paintAIPrompt({lastImgPt_.x(), lastImgPt_.y()}, {ip.x(), ip.y()},
                               aiPromptErase_, brushSize_);
            lastImgPt_ = ip;
        }
        update();
        return;
    }
    if (tool_ == Tool::Fill && isScribbleAlgorithm(fillAlgo_)) {
        doc_->paintSeedLine(cv::Point(lastImgPt_.x(), lastImgPt_.y()),
                            cv::Point(ip.x(), ip.y()), label_, brushSize_);
    } else {
        Label paintLabel = (tool_ == Tool::Eraser) ? Label::Background : label_;
        doc_->paintLine(cv::Point(lastImgPt_.x(), lastImgPt_.y()),
                        cv::Point(ip.x(), ip.y()), paintLabel, brushSize_);
    }
    lastImgPt_ = ip;
    update();
}

void CanvasWidget::mouseReleaseEvent(QMouseEvent* e) {
    if (e->button() == Qt::MiddleButton) { panning_ = false; return; }
    if (e->button() != Qt::LeftButton) return;
    if (drawing_) {
        if (tool_ == Tool::AIFill) mouseMoveEvent(e);
        drawing_ = false;
        emit maskChanged();
    }
}

void CanvasWidget::wheelEvent(QWheelEvent* e) {
    if (e->angleDelta().y() > 0) zoomIn();
    else zoomOut();
}

void CanvasWidget::zoomIn()    { zoom_ = std::min(zoom_ + 0.25f, 4.0f);  emit zoomChanged(zoom_); update(); }
void CanvasWidget::zoomOut()   { zoom_ = std::max(zoom_ - 0.25f, 0.25f); emit zoomChanged(zoom_); update(); }
void CanvasWidget::zoomReset() { zoom_ = 1.0f; panOffset_ = {}; drawing_ = false; panning_ = false; emit zoomChanged(zoom_); update(); }

} // namespace orthoseg
