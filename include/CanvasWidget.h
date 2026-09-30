#pragma once
#include "Document.h"
#include "Labels.h"
#include <QWidget>
#include <QImage>
#include <QPoint>
#include <vector>

namespace orthoseg {

// Renders the source X-ray with the label mask blended on top at maskOpacity,
// and translates mouse gestures into Document edits.
class CanvasWidget : public QWidget {
    Q_OBJECT
public:
    explicit CanvasWidget(Document* doc, QWidget* parent = nullptr);

    void setActiveTool(Tool t)          { drawing_ = false; outline_.clear(); tool_ = t; update(); }
    void setActiveLabel(Label l)        { label_ = l; update(); }
    void setIsolatedView(bool enabled)  { isolatedView_ = enabled; update(); }
    bool isolatedView() const           { return isolatedView_; }
    void setDrawFillHoles(bool enabled) { drawFillHoles_ = enabled; outline_.clear(); update(); }
    void setAutoFillOutline(bool enabled) { autoFillOutline_ = enabled; }
    void fillCurrentOutline();
    bool applyPendingAIResult();
    void undoLastEdit();
    void fillClosedAreas();
    void setMaxHolePixels(int value) { maxHolePixels_=value; }
    void clearOutline() { outline_.clear(); update(); }
    void setBrushSize(int s)            { brushSize_ = s; }
    void setAIPromptEditing(bool enabled) { aiPromptEditing_ = enabled; drawing_ = false; }
    void setAIPromptErase(bool erase)   { aiPromptErase_ = erase; }
    void setMaskOpacity(float o)        { opacity_ = o; update(); }
    void setFillAlgorithm(FillAlgorithm a) { fillAlgo_ = a; update(); }
    void setIntensityThreshold(int t)   { intensityThreshold_ = t; }
    void setEdgePenaltyThreshold(int t) { edgePenalty_ = t; }

    void setEdgeConstrained(bool enabled) { edgeConstrained_ = enabled; }
    void setPanMode(bool enabled);
    void panBy(const QPointF& delta);
    void centerImageEnd(bool bottom);

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
    void aiResultApplied();
    void panModeChanged(bool enabled);
    void maskChanged();   // emitted after any edit so the window can refresh UI
    void zoomChanged(float z);

protected:
    bool event(QEvent*) override;
    void paintEvent(QPaintEvent*) override;
    void mousePressEvent(QMouseEvent*) override;
    void mouseMoveEvent(QMouseEvent*) override;
    void mouseReleaseEvent(QMouseEvent*) override;
    void wheelEvent(QWheelEvent*) override;

private:
    void rebuildSourceImage();
    void zoomAt(const QPointF& anchor, double factor);

    Document* doc_;
    QImage    sourceQt_;       // cached BGR->RGB source

    Tool          tool_       = Tool::None;
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
    bool panMode_ = false;
    bool edgeConstrained_ = false;
    cv::Mat gestureRegion_;
    Qt::MouseButton panButton_ = Qt::NoButton;
    bool aiPromptErase_ = false;
    bool aiPromptEditing_ = true;
    bool isolatedView_ = false;
    int maxHolePixels_ = 0;
    bool drawFillHoles_ = false;
    bool autoFillOutline_ = false;
    std::vector<cv::Point> outline_;

    bool   claheEnabled_   = false;
    double claheClipLimit_ = 2.0;
    int    claheGridSize_  = 8;
};

} // namespace orthoseg
