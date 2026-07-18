#pragma once
#include "Document.h"
#include "CanvasWidget.h"
#include <QMainWindow>
#include <memory>

class QPushButton;
class QSlider;
class QComboBox;
class QLabel;
class QStackedWidget;
class QWidget;

namespace orthoseg {

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    MainWindow();

    // Test/screenshot hook: switch to the Fill tool with the given algorithm.
    void showFillAlgorithm(int comboIndex);

private slots:
    void onUpload();
    void onExport();
    void onClear();
    void onUndo();
    void onRunSegmentation();
    void onClearSeeds();
    void selectLabel(Label l);
    void selectTool(Tool t);

private:
    QWidget* buildSidebar();
    QWidget* buildTopBar();
    void updateSettingsVisibility();
    void updateUndoState();
    void updateStatus();

    std::unique_ptr<Document> doc_;
    CanvasWidget* canvas_ = nullptr;

    // Sidebar controls kept for state updates.
    QWidget* labelButtons_[4] = {nullptr, nullptr, nullptr, nullptr};
    QWidget* toolButtons_[3]  = {nullptr, nullptr, nullptr};
    QWidget* brushPanel_ = nullptr;
    QWidget* fillPanel_  = nullptr;
    QWidget* intensityPanel_ = nullptr;
    QWidget* edgePanel_  = nullptr;
    QWidget* betaPanel_  = nullptr;   // scribble algos: edge sensitivity
    QWidget* seedPanel_  = nullptr;   // scribble algos: hint + Run/Clear Seeds
    QLabel*  graphCutNote_ = nullptr;
    QSlider* brushSlider_ = nullptr;
    QSlider* intensitySlider_ = nullptr;
    QSlider* edgeSlider_ = nullptr;
    QSlider* betaSlider_ = nullptr;
    QSlider* opacitySlider_ = nullptr;
    QComboBox* algoCombo_ = nullptr;
    QLabel* brushValue_ = nullptr;
    QLabel* intensityValue_ = nullptr;
    QLabel* edgeValue_ = nullptr;
    QLabel* betaValue_ = nullptr;
    QLabel* opacityValue_ = nullptr;

    // Effective beta for the edge-weight kernel exp(-beta*dI^2); the slider is
    // 1..100 and maps to a small positive constant.
    double currentBeta() const;

    QPushButton* undoBtn_ = nullptr;
    QLabel* zoomLabel_ = nullptr;
    QLabel* dimLabel_ = nullptr;

    Tool  activeTool_ = Tool::Brush;
    Label activeLabel_ = Label::Femur;
};

} // namespace orthoseg
