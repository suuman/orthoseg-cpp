#pragma once
#include "Document.h"
#include "Labels.h"
#include <QWidget>
#include <QImage>
#include <QPoint>

namespace orthoseg {

// Renders the source X-ray with the label mask blended on top at maskOpacity,
// and translates mouse gestures into Document edits.
class CanvasWidget : public QWidget {
    Q_OBJECT
public:
    explicit CanvasWidget(Document* doc, QWidget* parent = nullptr);

    void setActiveTool(Tool t)          { tool_ = t; }
    void setActiveLabel(Label l)        { label_ = l; }
    void setBrushSize(int s)            { brushSize_ = s; }
    void setMaskOpacity(float o)        { opacity_ = o; update(); }
    void setFillAlgorithm(FillAlgorithm a) { fillAlgo_ = a; }
    void setIntensityThreshold(int t)   { intensityThreshold_ = t; }
    void setEdgePenaltyThreshold(int t) { edgePenalty_ = t; }

    void refresh();  // rebuild cached QImages from the Document and repaint

    float zoom() const { return zoom_; }
    void  zoomIn();
    void  zoomOut();
    void  zoomReset();

signals:
    void maskChanged();   // emitted after any edit so the window can refresh UI
    void zoomChanged(float z);

protected:
    void paintEvent(QPaintEvent*) override;
    void mousePressEvent(QMouseEvent*) override;
    void mouseMoveEvent(QMouseEvent*) override;
    void mouseReleaseEvent(QMouseEvent*) override;
    void wheelEvent(QWheelEvent*) override;

private:
    QPoint widgetToImage(const QPoint& p) const;
    QRectF imageRect() const;  // where the image is drawn in widget coords
    void rebuildSourceImage();

    Document* doc_;
    QImage    sourceQt_;       // cached BGR->RGB source

    Tool          tool_       = Tool::Brush;
    Label         label_      = Label::Femur;
    int           brushSize_  = 20;
    float         opacity_    = 0.5f;
    FillAlgorithm fillAlgo_   = FillAlgorithm::Standard;
    int           intensityThreshold_ = 5;
    int           edgePenalty_        = 30;

    float   zoom_ = 1.0f;
    bool    drawing_ = false;
    QPoint  lastImgPt_;
};

} // namespace orthoseg
