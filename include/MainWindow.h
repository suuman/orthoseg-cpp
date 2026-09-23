#pragma once
#include "Document.h"
#include "CanvasWidget.h"
#include "AIFillController.h"
#include "MonaiClient.h"
#include <QSet>
#include <QHash>
#include <QMainWindow>
#include <memory>

class QPushButton;
class QSlider;
class QComboBox;
class QLabel;
class QStackedWidget;
class QWidget;
class QLineEdit;
class QCheckBox;

namespace orthoseg {

class ModelManagementDialog;

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    MainWindow();
    ~MainWindow() override;

    // Test/screenshot hook: switch to the Fill tool with the given algorithm.
    void showFillAlgorithm(int comboIndex);
    Document* document() const { return doc_.get(); }
    CanvasWidget* canvas() const { return canvas_; }
    QCheckBox* claheCheckBox() const { return claheCheck_; }
    QDialog* claheDialog() const { return claheDialog_; }

private slots:
    void onUpload();
    void onMonaiSegment();
    void onExport();
    void onClear();
    void onUndo();
    void onRunSegmentation();
    void onClearSeeds();
    void onRunAIFill();
    void onLoadPromptMask();
    void onOpenModelDirDialog();
    void onOpenClaheDialog();
    void selectLabel(Label l);
    void selectTool(Tool t);

private:
    void offerMonaiTraining();
    QWidget* buildSidebar();
    QWidget* buildTopBar();
    QWidget* buildAIPanel();
    void updateSettingsVisibility();
    void updateUndoState();
    void updateStatus();
    void updateAIPromptStatus();

    ModelManagementDialog* modelManagement_ = nullptr;
    std::unique_ptr<Document> doc_;
    std::unique_ptr<MonaiClient> monai_;
    QPushButton* monaiSegment_ = nullptr;
    QHash<QByteArray, QString> monaiVersions_;
    QSet<QByteArray> monaiPrompted_;
    CanvasWidget* canvas_ = nullptr;

    // Sidebar controls kept for state updates.
    QWidget* labelButtons_[4] = {nullptr, nullptr, nullptr, nullptr};
    QWidget* toolButtons_[4]  = {nullptr, nullptr, nullptr, nullptr};
    QWidget* aiPanel_ = nullptr;
    QComboBox* aiPromptCombo_ = nullptr;
    QLineEdit* aiModels_ = nullptr;
    QPushButton* modelDirBtn_ = nullptr;
    QCheckBox* claheCheck_ = nullptr;
    QPushButton* claheSettingsBtn_ = nullptr;
    QDialog* claheDialog_ = nullptr;
    QLabel* aiStatus_ = nullptr;
    QLabel* aiPromptHint_ = nullptr;
    QPushButton* aiRun_ = nullptr;
    QCheckBox* aiShow_ = nullptr;
    QCheckBox* aiShowPrompt_ = nullptr;
    QCheckBox* aiErase_ = nullptr;
    QPushButton* aiLoadMask_ = nullptr;
    QPushButton* useResultAsPromptBtn_ = nullptr;
    QWidget* aiBoxControls_ = nullptr;
    QLabel* femurBoxStatus_ = nullptr;
    QPushButton* clearFemurBoxBtn_ = nullptr;
    QLabel* tibiaBoxStatus_ = nullptr;
    QPushButton* clearTibiaBoxBtn_ = nullptr;
    QWidget* normalMaskControls_ = nullptr;
    QLabel* normalMaskStatus_ = nullptr;
    QPushButton* copyNormalMaskBtn_ = nullptr;
    std::unique_ptr<AIFillController> aiController_;
    unsigned long imageGeneration_ = 0;
    unsigned long submittedGeneration_ = 0;
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
