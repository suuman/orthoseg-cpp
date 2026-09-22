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

    void setActiveTool(Tool t)          { drawing_ = false; tool_ = t; update(); }
    void setActiveLabel(Label l)        { label_ = l; }
    void setBrushSize(int s)            { brushSize_ = s; }
    void setAIPromptErase(bool erase)   { aiPromptErase_ = erase; }
    void setMaskOpacity(float o)        { opacity_ = o; update(); }
    void setFillAlgorithm(FillAlgorithm a) { fillAlgo_ = a; update(); }
    void setIntensityThreshold(int t)   { intensityThreshold_ = t; }
    void setEdgePenaltyThreshold(int t) { edgePenalty_ = t; }

    bool claheEnabled() const           { return claheEnabled_; }
    void setClaheEnabled(bool enabled);
    double claheClipLimit() const       { return claheClipLimit_; }
    int claheGridSize() const           { return claheGridSize_; }
    void setClaheParams(double clipLimit, int gridSize);
    const QImage& displayImage() const  { return sourceQt_; }

    void refresh();  // rebuild cached QImages from the Document and repaint

    float zoom() const { return zoom_; }
    void  zoomIn();
    void  zoomOut();
    void  zoomReset();

    // Qt mouse positions and painting use logical pixels (including HiDPI).
    QPoint widgetToImage(const QPointF& p) const;
    QRectF imageRect() const;

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
    QPoint boxStart_;
    QPointF panOffset_;
    QPointF lastPanPos_;
    bool panning_ = false;
    bool aiPromptErase_ = false;

    bool   claheEnabled_   = false;
    double claheClipLimit_ = 2.0;
    int    claheGridSize_  = 8;
};

} // namespace orthoseg
