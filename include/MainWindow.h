#pragma once
#include "Document.h"
#include "CanvasWidget.h"
#include "AIFillController.h"
#include "MonaiClient.h"
#include <QSet>
#include <QHash>
#include <QMainWindow>
#include <QString>
#include <memory>
#include <vector>

class QPushButton;
class QSlider;
class QComboBox;
class QLabel;
class QStackedWidget;
class QWidget;
class QLineEdit;
class QCheckBox;
class QTimer;
class QAction;

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
    void offerFineTuningReminder();
    void onUpload();
    void onBatchMode();
    void onImportMask();
    void onMonaiSegment();
    void onExport();
    void onClear();
    void onUndo();
    void onRunSegmentation();
    void onClearSeeds();
    void onRunAIFill();
#ifdef ORTHOSEG_NATIVE_NNUNET
    void onRunNativeNnUnet();
#endif
    void onLoadPromptMask();
    void onOpenModelDirDialog();
    void onOpenClaheDialog();
    void selectLabel(Label l);
    void selectTool(Tool t);

private:
    bool confirmOpenProcessed(const QString& path);
    void markProcessed(const QString& maskPath);
    QHash<QString, QString> processedFiles_;
    struct ToolSettings { int size = 20; int edge = 30; bool constrained = false; };
    std::array<ToolSettings, 7> toolSettings_;
    void recordAppliedMonaiResult();
    void offerMonaiTraining();
    void startMonaiSegment(bool prompted, bool nnunet = false);
    void checkMonaiStatus();
    void openModelManagement();
    QWidget* buildSidebar();
    QWidget* buildTopBar();
    QWidget* buildAIPanel();
    void updateSettingsVisibility();
    void updateUndoState();
    void updateStatus();
    void updateAIPromptStatus();
    void finishImageLoad();
    bool loadBatchItem();
    void endBatchMode();
    void updateBatchUi();

    struct BatchItem {
        QString imagePath;
        QString labelPath;
        QString outputPath;
    };
    std::vector<BatchItem> batchItems_;
    size_t batchIndex_ = 0;
    bool batchActive_ = false;
    QString batchImagesDir_;
    QString batchLabelsDir_;
    QString batchOutputDir_;

    ModelManagementDialog* modelManagement_ = nullptr;
    std::unique_ptr<Document> doc_;
    std::unique_ptr<MonaiClient> monai_;
    QPushButton* monaiSegment_ = nullptr;
    QComboBox* aiModelSelector_ = nullptr;
    QLineEdit* monaiUrl_ = nullptr;
    QLabel* monaiStatus_ = nullptr;
    QLabel* monaiModel_ = nullptr;
    QPushButton* managementButton_ = nullptr;
    QAction* managementAction_ = nullptr;
    QWidget* medSamControls_ = nullptr;
    QWidget* monaiControls_ = nullptr;
#ifdef ORTHOSEG_NATIVE_NNUNET
    QWidget* nativeNnUnetControls_ = nullptr;
    QString nativeNnUnetModelPath_;
    QString nativeNnUnetDeviceKey_ = "auto";
    cv::Mat nativeNnUnetBefore_;
    unsigned long nativeNnUnetGeneration_ = 0;
#endif
    QWidget* aiResultControls_ = nullptr;
    QTimer* monaiCheckTimer_ = nullptr;
    unsigned monaiCheckSerial_ = 0;
    bool monaiRequestPending_ = false;
    bool monaiReady_ = false;
    bool fineTuningReminderPending_ = false;
    QHash<QByteArray, QString> monaiVersions_;
    QByteArray pendingMonaiVersionKey_;
    QString pendingMonaiVersion_;
    QSet<QByteArray> monaiPrompted_;
    CanvasWidget* canvas_ = nullptr;

    // Sidebar controls kept for state updates.
    QWidget* labelButtons_[4] = {nullptr, nullptr, nullptr, nullptr};
    QWidget* toolButtons_[6]  = {};
    QWidget* aiPanel_ = nullptr;
    QComboBox* aiPromptCombo_ = nullptr;
    QLineEdit* aiModels_ = nullptr;
    QPushButton* modelDirBtn_ = nullptr;
    QPushButton* batchModeBtn_ = nullptr;
    QPushButton* exportMaskBtn_ = nullptr;
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
    QPushButton* nextBoxBtn_ = nullptr;
    QLabel* femurBox2Status_ = nullptr;
    QLabel* tibiaBox2Status_ = nullptr;
    QPushButton* clearFemurBox2Btn_ = nullptr;
    QPushButton* clearTibiaBox2Btn_ = nullptr;
    QWidget* normalMaskControls_ = nullptr;
    QLabel* normalMaskStatus_ = nullptr;
    QPushButton* copyNormalMaskBtn_ = nullptr;
    std::unique_ptr<AIFillController> aiController_;
    unsigned long imageGeneration_ = 0;
    unsigned long submittedGeneration_ = 0;
    std::vector<int> submittedLabels_;
    QWidget* brushPanel_ = nullptr;
    QWidget* fillPanel_  = nullptr;
    QWidget* drawFillPanel_ = nullptr;
    void setFillOutlineHighlight(bool highlighted);
    QPushButton* fillOutlineButton_ = nullptr;
    QPushButton* drawOutlineButton_ = nullptr;
    QPushButton* fillClosedAreaButton_ = nullptr;
    QLabel* lassoHint_ = nullptr;
    QWidget* intensityPanel_ = nullptr;
    QCheckBox* constrainEdges_ = nullptr;
    QWidget* edgePanel_  = nullptr;
    QWidget* betaPanel_  = nullptr;   // scribble algos: edge sensitivity
    QCheckBox* restrictSeedRegion_ = nullptr;
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

    Tool  activeTool_ = Tool::None;
    Label activeLabel_ = Label::Femur;
};

} // namespace orthoseg
