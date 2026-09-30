#include "CanvasWidget.h"
#include <QPainter>
#include <QPainterPath>
#include <QMouseEvent>
#include <QWheelEvent>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace orthoseg {

CanvasWidget::CanvasWidget(Document* doc, QWidget* parent)
    : QWidget(parent), doc_(doc) {
    setMouseTracking(true);
    setCursor(panMode_ ? Qt::OpenHandCursor : Qt::CrossCursor);
    setMinimumSize(400, 400);
    setToolTip("Wheel: zoom at pointer. Right-drag, middle-drag, or Ctrl+left-drag: pan.");
}

void CanvasWidget::setClaheEnabled(bool enabled) {
    if (claheEnabled_ != enabled) {
        claheEnabled_ = enabled;
        rebuildSourceImage();
        update();
    }
}

void CanvasWidget::setClaheParams(double clipLimit, int gridSize) {
    claheClipLimit_ = std::max(0.1, clipLimit);
    claheGridSize_ = std::max(1, gridSize);
    if (claheEnabled_) {
        rebuildSourceImage();
        update();
    }
}

void CanvasWidget::rebuildSourceImage() {
    const cv::Mat& c = doc_->sourceColor();
    if (c.empty()) { sourceQt_ = QImage(); return; }

    cv::Mat displayMat;
    if (claheEnabled_) {
        try {
            int gx = std::max(1, std::min(claheGridSize_, c.cols));
            int gy = std::max(1, std::min(claheGridSize_, c.rows));
            auto clahe = cv::createCLAHE(claheClipLimit_, cv::Size(gx, gy));
            if (c.channels() == 1) {
                clahe->apply(c, displayMat);
                cv::cvtColor(displayMat, displayMat, cv::COLOR_GRAY2BGR);
            } else {
                cv::Mat lab;
                cv::cvtColor(c, lab, cv::COLOR_BGR2Lab);
                std::vector<cv::Mat> channels;
                cv::split(lab, channels);
                clahe->apply(channels[0], channels[0]);
                cv::merge(channels, lab);
                cv::cvtColor(lab, displayMat, cv::COLOR_Lab2BGR);
            }
        } catch (const cv::Exception&) {
            displayMat = c;
        }
    } else {
        displayMat = c;
    }

    // OpenCV is BGR; QImage::Format_RGB888 expects RGB. Copy with swap.
    QImage img(displayMat.cols, displayMat.rows, QImage::Format_RGB888);
    for (int y = 0; y < displayMat.rows; ++y) {
        const cv::Vec3b* srow = displayMat.ptr<cv::Vec3b>(y);
        uchar* drow = img.scanLine(y);
        for (int x = 0; x < displayMat.cols; ++x) {
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
    const double iw = doc_->width(), ih = doc_->height();
    // Fit-to-widget base scale, then apply zoom and the user's pan offset.
    const double base = std::min(width() / iw, height() / ih);
    const double dw = iw * base * zoom_, dh = ih * base * zoom_;
    return QRectF((width() - dw) / 2.0 + panOffset_.x(),
                  (height() - dh) / 2.0 + panOffset_.y(), dw, dh);
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

    // Each structure has an independent discrete channel. Rendering binary
    // occupancy in primary colors makes overlaps additive (R+G=yellow, etc.).
    const cv::Mat& mask = doc_->maskChannels();
    const auto& ai = doc_->aiFill();
    const bool validPreview = ai.showResult &&
        !ai.resultMask.empty() && ai.resultMask.size() == mask.size() &&
        (ai.resultMask.type() == CV_8UC1 || ai.resultMask.type() == CV_8UC3);
    QImage overlay(mask.cols, mask.rows, QImage::Format_ARGB32);
    overlay.fill(Qt::transparent);
    const int a = static_cast<int>(opacity_ * 255);
    for (int y = 0; y < mask.rows; ++y) {
        const auto* mrow = mask.ptr<cv::Vec3b>(y);
        QRgb* orow = reinterpret_cast<QRgb*>(overlay.scanLine(y));
        for (int x = 0; x < mask.cols; ++x) {
            bool femur = mrow[x][2] != 0;
            bool tibia = mrow[x][1] != 0;
            bool fibula = mrow[x][0] != 0;
            if (validPreview && ai.resultReplacesAnatomy) {
                if (ai.resultMask.type() == CV_8UC3) {
                    const auto predicted = ai.resultMask.at<cv::Vec3b>(y, x);
                    femur = predicted[2] == 1;
                    tibia = predicted[1] == 2;
                } else {
                    const uchar predicted = ai.resultMask.at<uchar>(y, x);
                    femur = predicted == 1;
                    tibia = predicted == 2;
                }
            } else if (validPreview) {
                const uchar predicted = ai.resultMask.at<uchar>(y, x);
                femur |= predicted == 1;
                tibia |= predicted == 2;
                fibula |= predicted == 3;
            }
            if (isolatedView_) {
                femur &= label_ == Label::Femur;
                tibia &= label_ == Label::Tibia;
                fibula &= label_ == Label::Fibula;
            }
            orow[x] = qRgba(femur ? 255 : 0, tibia ? 255 : 0, fibula ? 255 : 0,
                              femur || tibia || fibula ? a : 0);
        }
    }
    p.drawImage(dst, overlay);

    if (tool_ == Tool::AIFill && ai.showPrompt && ai.promptType != AIFillPromptType::BoundingBox && !ai.promptMask.empty()) {
        overlay.fill(Qt::transparent);
        const cv::Vec3b activeBgr = labelInfo(label_).colorBGR;
        for (int y = 0; y < ai.promptMask.rows; ++y) {
            auto* row = reinterpret_cast<QRgb*>(overlay.scanLine(y));
            const auto* maskRow = ai.promptMask.ptr<uchar>(y);
            for (int x = 0; x < ai.promptMask.cols; ++x) {
                uchar val = maskRow[x];
                if (val == 0) continue;
                if (val == 1 || val == 2 || val == 3) {
                    const auto bgr = labelInfo(static_cast<Label>(val)).colorBGR;
                    row[x] = qRgba(bgr[2], bgr[1], bgr[0], a);
                } else {
                    row[x] = qRgba(activeBgr[2], activeBgr[1], activeBgr[0], a);
                }
            }
        }
        p.drawImage(dst, overlay);
    }
    if (tool_ == Tool::AIFill && ai.showPrompt && ai.promptType == AIFillPromptType::BoundingBox) {
        const double sx = dst.width() / doc_->width(), sy = dst.height() / doc_->height();

        auto drawBoxWithBadge = [&](const Box& b, const QColor& color, const QString& badge) {
            QRectF br(QPointF(dst.left() + b.x0 * sx, dst.top() + b.y0 * sy),
                      QPointF(dst.left() + b.x1 * sx, dst.top() + b.y1 * sy));
            br = br.normalized();
            p.setPen(QPen(color, 2, Qt::SolidLine));
            p.setBrush(QColor(0x22, 0xc5, 0x5e, 70)); // Prominent green fill for AI Fill box
            p.drawRect(br);

            QFont f = p.font();
            f.setPointSize(9);
            f.setBold(true);
            p.setFont(f);
            QFontMetrics fm(f);
            int bw = fm.horizontalAdvance(badge) + 8;
            int bh = fm.height() + 4;
            QRect badgeRect(static_cast<int>(br.left()),
                            static_cast<int>(std::max(dst.top(), br.top() - bh)),
                            bw, bh);
            p.fillRect(badgeRect, color);
            p.setPen(Qt::white);
            p.drawText(badgeRect, Qt::AlignCenter, badge);
        };

        if (ai.femurBox) {
            drawBoxWithBadge(*ai.femurBox, QColor(0xef, 0x44, 0x44), "Femur");
        }
        if (ai.tibiaBox) {
            drawBoxWithBadge(*ai.tibiaBox, QColor(0x22, 0xc5, 0x5e), "Tibia");
        }
        if (ai.showSecondaryBoxes && ai.femurBox2)
            drawBoxWithBadge(*ai.femurBox2, QColor(0xef, 0x44, 0x44), "Femur 2");
        if (ai.showSecondaryBoxes && ai.tibiaBox2)
            drawBoxWithBadge(*ai.tibiaBox2, QColor(0x22, 0xc5, 0x5e), "Tibia 2");
        if (!ai.femurBox && !ai.tibiaBox && !ai.femurBox2 && !ai.tibiaBox2 && ai.box) {
            QColor col = (label_ == Label::Tibia) ? QColor(0x22, 0xc5, 0x5e) : QColor(0xef, 0x44, 0x44);
            QString name = (label_ == Label::Tibia) ? "Tibia" : "Femur";
            drawBoxWithBadge(*ai.box, col, name);
        }
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
    if ((tool_ == Tool::DrawFill || tool_ == Tool::Lasso) && !outline_.empty()) {
        const double sx = dst.width() / doc_->width();
        const double sy = dst.height() / doc_->height();
        QPainterPath path;
        path.moveTo(dst.left() + outline_[0].x * sx, dst.top() + outline_[0].y * sy);
        for (size_t i = 1; i < outline_.size(); ++i)
            path.lineTo(dst.left() + outline_[i].x * sx, dst.top() + outline_[i].y * sy);
        if (tool_ == Tool::Lasso && outline_.size() >= 3) path.closeSubpath();
        p.setPen(QPen(QColor(56, 189, 248), 2, Qt::DashLine));
        p.setBrush(Qt::NoBrush);
        p.drawPath(path);
    }
}

bool CanvasWidget::applyPendingAIResult() {
    if (doc_->aiFill().resultMask.empty()) return true;
    doc_->applyAIResult();
    if (!doc_->aiFill().resultMask.empty()) return false;
    emit aiResultApplied();
    emit maskChanged();
    update();
    return true;
}

void CanvasWidget::undoLastEdit() {
    const bool pendingLasso = tool_ == Tool::Lasso && !outline_.empty();
    drawing_ = false;
    panning_ = false;
    outline_.clear();
    gestureRegion_.release();
    if (!pendingLasso) doc_->undo();
    emit maskChanged();
    update();
}

bool CanvasWidget::event(QEvent* event) {
    if (event->type() == QEvent::WindowDeactivate || event->type() == QEvent::UngrabMouse) {
        if (drawing_) { drawing_ = false; emit maskChanged(); }
        panning_ = false;
    }
    return QWidget::event(event);
}

void CanvasWidget::fillClosedAreas() {
    if (!doc_->hasImage() || label_ == Label::Background) return;
    if (!applyPendingAIResult()) return;
    outline_.clear();
    if (doc_->fillEnclosedHoles(label_, maxHolePixels_)) emit maskChanged();
    update();
}

void CanvasWidget::fillCurrentOutline() {
    if (outline_.size() < 3 || !doc_->hasImage()) return;
    if (!applyPendingAIResult()) return;
    doc_->fillPolygon(outline_, label_, true,
        edgeConstrained_ && tool_ == Tool::Lasso ? doc_->edgeRegion(outline_.front(), edgePenalty_) : cv::Mat());
    outline_.clear();
    emit maskChanged();
    update();
}

void CanvasWidget::mousePressEvent(QMouseEvent* e) {
    if (doc_->hasImage() && (e->button() == Qt::MiddleButton ||
        e->button() == Qt::RightButton ||
        (e->button() == Qt::LeftButton && (panMode_ || (e->modifiers() & Qt::ControlModifier))))) {
        panning_ = true;
        panButton_ = e->button();
        lastPanPos_ = e->position();
        setCursor(Qt::ClosedHandCursor);
        e->accept();
        return;
    }
    if (!doc_->hasImage() || e->button() != Qt::LeftButton || tool_ == Tool::None ||
        (tool_ == Tool::AIFill && !aiPromptEditing_)) return;
    QPoint ip = widgetToImage(e->position());
    if (ip.x() < 0 || ip.y() < 0 ||
        ip.x() >= doc_->width() || ip.y() >= doc_->height()) return;

    if (tool_ == Tool::AIFill) {
        if (doc_->aiFill().showSecondaryBoxes &&
            label_ != Label::Femur && label_ != Label::Tibia) return;
        if (doc_->aiFill().promptType == AIFillPromptType::LoadedMask ||
            doc_->aiFill().promptType == AIFillPromptType::NormalFillMask) return;
        drawing_ = true;
        doc_->aiFill().showPrompt = true;
        if (doc_->aiFill().promptType == AIFillPromptType::BoundingBox) {
            boxStart_ = ip;
            Box b{float(ip.x()), float(ip.y()), float(ip.x()), float(ip.y())};
            doc_->aiFill().box = b;
            if (label_ == Label::Femur) {
                if (doc_->aiFill().showSecondaryBoxes && doc_->aiFill().activeBoxNumber == 2)
                    doc_->aiFill().femurBox2 = b;
                else doc_->aiFill().femurBox = b;
            } else if (label_ == Label::Tibia) {
                if (doc_->aiFill().showSecondaryBoxes && doc_->aiFill().activeBoxNumber == 2)
                    doc_->aiFill().tibiaBox2 = b;
                else doc_->aiFill().tibiaBox = b;
            } else {
                doc_->aiFill().femurBox = b;
            }
        } else {
            // When editing paint prompt, if promptMask is empty and AI resultMask exists,
            // initialize promptMask with resultMask so edits directly modify the AI mask!
            if (doc_->aiFill().promptMask.empty() && !doc_->aiFill().resultMask.empty()) {
                if (doc_->aiFill().resultMask.type() == CV_8UC3 && label_ != Label::Background) {
                    cv::Mat bone;
                    cv::extractChannel(doc_->aiFill().resultMask, bone, 3 - static_cast<int>(label_));
                    doc_->aiFill().promptMask = bone != 0;
                } else if (doc_->aiFill().resultMask.type() == CV_8UC1) {
                    doc_->aiFill().promptMask = doc_->aiFill().resultMask.clone();
                }
            }
            lastImgPt_ = ip;
            doc_->paintAIPrompt({ip.x(), ip.y()}, {ip.x(), ip.y()}, aiPromptErase_, brushSize_);
        }
        update();
        return;
    }

    if (!applyPendingAIResult()) return;

    gestureRegion_ = edgeConstrained_ && tool_ != Tool::Fill
        ? doc_->edgeRegion({ip.x(), ip.y()}, edgePenalty_) : cv::Mat();

    if (tool_ == Tool::DrawFill || tool_ == Tool::Lasso) {
        if (tool_ == Tool::DrawFill && drawFillHoles_) {
            fillClosedAreas();
            return;
        }
        drawing_ = true;
        outline_.clear();
        outline_.push_back(cv::Point(ip.x(), ip.y()));
        lastImgPt_ = ip;
        if (tool_ == Tool::DrawFill) {
            doc_->pushHistory();
            doc_->paintLine(cv::Point(ip.x(), ip.y()), cv::Point(ip.x(), ip.y()), label_, brushSize_, gestureRegion_);
            emit maskChanged();
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
    if (tool_ == Tool::Eraser)
        doc_->eraseLabelLine(cv::Point(ip.x(), ip.y()), cv::Point(ip.x(), ip.y()), label_, brushSize_, gestureRegion_);
    else
        doc_->paintLine(cv::Point(ip.x(), ip.y()), cv::Point(ip.x(), ip.y()), label_, brushSize_, gestureRegion_);
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
    if (e->type() == QEvent::MouseMove && !(e->buttons() & Qt::LeftButton)) {
        // A lost release must not connect a later hover to the last drawn point.
        drawing_ = false;
        emit maskChanged();
        update();
        return;
    }
    QPoint ip = widgetToImage(e->position());
    if (tool_ == Tool::DrawFill || tool_ == Tool::Lasso) {
        ip.setX(std::clamp(ip.x(), 0, doc_->width() - 1));
        ip.setY(std::clamp(ip.y(), 0, doc_->height() - 1));
        if (ip == lastImgPt_) return;
        if (tool_ == Tool::DrawFill)
            doc_->paintLine(cv::Point(lastImgPt_.x(), lastImgPt_.y()),
                cv::Point(ip.x(), ip.y()), label_, brushSize_, gestureRegion_);
        outline_.push_back(cv::Point(ip.x(), ip.y()));
        lastImgPt_ = ip;
        update();
        return;
    }
    if (tool_ == Tool::AIFill) {
        if (doc_->aiFill().promptType == AIFillPromptType::BoundingBox) {
            ip.setX(std::clamp(ip.x(), 0, doc_->width() - 1));
            ip.setY(std::clamp(ip.y(), 0, doc_->height() - 1));
            Box b{float(std::min(boxStart_.x(), ip.x())),
                  float(std::min(boxStart_.y(), ip.y())),
                  float(std::max(boxStart_.x(), ip.x())),
                  float(std::max(boxStart_.y(), ip.y()))};
            doc_->aiFill().box = b;
            if (label_ == Label::Femur) {
                if (doc_->aiFill().showSecondaryBoxes && doc_->aiFill().activeBoxNumber == 2)
                    doc_->aiFill().femurBox2 = b;
                else doc_->aiFill().femurBox = b;
            } else if (label_ == Label::Tibia) {
                if (doc_->aiFill().showSecondaryBoxes && doc_->aiFill().activeBoxNumber == 2)
                    doc_->aiFill().tibiaBox2 = b;
                else doc_->aiFill().tibiaBox = b;
            } else {
                doc_->aiFill().femurBox = b;
            }
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
        if (tool_ == Tool::Eraser)
            doc_->eraseLabelLine(cv::Point(lastImgPt_.x(), lastImgPt_.y()),
                cv::Point(ip.x(), ip.y()), label_, brushSize_, gestureRegion_);
        else
            doc_->paintLine(cv::Point(lastImgPt_.x(), lastImgPt_.y()),
                            cv::Point(ip.x(), ip.y()), label_, brushSize_, gestureRegion_);
    }
    lastImgPt_ = ip;
    update();
}

void CanvasWidget::mouseReleaseEvent(QMouseEvent* e) {
    if (panning_ && e->button() == panButton_) {
        panning_ = false;
        panButton_ = Qt::NoButton;
        setCursor(panMode_ ? Qt::OpenHandCursor : Qt::CrossCursor);
        e->accept();
        return;
    }
    if (e->button() != Qt::LeftButton) return;
    if (drawing_) {
        if (tool_ == Tool::AIFill || tool_ == Tool::DrawFill || tool_ == Tool::Lasso)
            mouseMoveEvent(e);
        drawing_ = false;
        if ((tool_ == Tool::DrawFill || tool_ == Tool::Lasso) && autoFillOutline_ && outline_.size() >= 3) {
            doc_->fillPolygon(outline_, label_, tool_ != Tool::DrawFill,
                tool_ == Tool::Lasso ? gestureRegion_ : cv::Mat());
            outline_.clear();
        }
        emit maskChanged();
    }
}

void CanvasWidget::wheelEvent(QWheelEvent* e) {
    const int angle = e->angleDelta().y();
    const int pixels = e->pixelDelta().y();
    if (angle == 0 && pixels == 0) return;
    const double factor = angle != 0 ? std::pow(1.2, angle / 120.0)
                                     : std::exp(pixels / 600.0);
    const QRectF current = imageRect();
    const QPointF anchor = current.contains(e->position())
        ? e->position() : QPointF(width() / 2.0, height() / 2.0);
    zoomAt(anchor, factor);
    e->accept();
}

void CanvasWidget::zoomAt(const QPointF& anchor, double factor) {
    const QRectF before = imageRect();
    if (before.isEmpty()) return;
    const double next = std::clamp(static_cast<double>(zoom_) * factor, 0.1, 32.0);
    if (next == zoom_) return;

    // Preserve the source point under the pointer as the image grows or shrinks.
    const double imageX = (anchor.x() - before.left()) / before.width();
    const double imageY = (anchor.y() - before.top()) / before.height();
    zoom_ = static_cast<float>(next);
    const QRectF after = imageRect();
    panOffset_ += anchor - QPointF(after.left() + imageX * after.width(),
                                   after.top() + imageY * after.height());
    emit zoomChanged(zoom_);
    update();
}

void CanvasWidget::setPanMode(bool enabled) {
    if (panMode_ == enabled) return;
    panMode_ = enabled;
    drawing_ = false;
    setCursor(enabled ? Qt::OpenHandCursor : Qt::CrossCursor);
    emit panModeChanged(enabled);
}

void CanvasWidget::panBy(const QPointF& delta) {
    if (!doc_->hasImage()) return;
    panOffset_ += delta;
    update();
}

void CanvasWidget::centerImageEnd(bool bottom) {
    if (!doc_->hasImage()) return;
    const QRectF image = imageRect();
    panBy(QPointF(width() / 2.0, height() / 2.0) -
          QPointF(image.center().x(), bottom ? image.bottom() : image.top()));
}

void CanvasWidget::zoomIn()  { zoomAt(QPointF(width() / 2.0, height() / 2.0), 1.2); }
void CanvasWidget::zoomOut() { zoomAt(QPointF(width() / 2.0, height() / 2.0), 1.0 / 1.2); }
void CanvasWidget::zoomReset() {
    zoom_ = 1.0f;
    panOffset_ = {};
    drawing_ = false;
    panning_ = false;
    panButton_ = Qt::NoButton;
    setCursor(panMode_ ? Qt::OpenHandCursor : Qt::CrossCursor);
    emit zoomChanged(zoom_);
    update();
}

} // namespace orthoseg
