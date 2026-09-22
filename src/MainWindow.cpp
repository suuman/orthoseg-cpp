#include "MainWindow.h"
#include <QApplication>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGridLayout>
#include <QPushButton>
#include <QSlider>
#include <QComboBox>
#include <QLabel>
#include <QFileDialog>
#include <QMessageBox>
#include <QFrame>
#include <QButtonGroup>
#include <QScrollArea>
#include <QLineEdit>
#include <QCheckBox>
#include <QDialog>
#include <QMouseEvent>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>
#include <stdexcept>

namespace orthoseg {

// Accent color from the web reference (#38bdf8) and slate palette.
static const char* kAccent = "#38bdf8";

static QLabel* sectionLabel(const QString& text) {
    auto* l = new QLabel(text.toUpper());
    l->setStyleSheet("color:#64748b; font-size:10px; font-weight:700; "
                     "letter-spacing:2px;");
    return l;
}

MainWindow::MainWindow() : doc_(std::make_unique<Document>()) {
    setWindowTitle("OrthoSeg — Medical Imaging");
    resize(1280, 800);

    aiModels_ = new QLineEdit(qEnvironmentVariable("MEDSAM2_MODEL_DIR", ORTHOSEG_MODEL_DIR));
    aiModels_->setObjectName("aiModelDirectory");
    aiModels_->setToolTip("Directory containing medsam2_image_encoder.onnx and medsam2_mask_decoder.onnx");

    canvas_ = new CanvasWidget(doc_.get());
    connect(canvas_, &CanvasWidget::maskChanged, this, [this] {
        updateUndoState();
        updateAIPromptStatus();
        if (doc_->hasImage() && !doc_->mask().empty() && cv::countNonZero(doc_->mask()) > 0) {
            doc_->aiFill().promptMask = doc_->mask().clone();
            doc_->aiFill().promptType = AIFillPromptType::PaintedMask;
            if (aiPromptCombo_) {
                aiPromptCombo_->blockSignals(true);
                aiPromptCombo_->setCurrentIndex(static_cast<int>(AIFillPromptType::PaintedMask));
                aiPromptCombo_->blockSignals(false);
            }
        }
    });
    connect(canvas_, &CanvasWidget::zoomChanged, this, [this](float) {
        updateStatus();
    });

    auto* central = new QWidget;
    auto* root = new QHBoxLayout(central);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(0);

    root->addWidget(buildSidebar());

    auto* mainArea = new QWidget;
    auto* mainLayout = new QVBoxLayout(mainArea);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(0);
    mainLayout->addWidget(buildTopBar());
    mainLayout->addWidget(canvas_, 1);
    root->addWidget(mainArea, 1);

    setCentralWidget(central);
    central->setStyleSheet("background:#0f172a; color:#e2e8f0;");

    selectTool(Tool::Brush);
    selectLabel(Label::Femur);
    updateSettingsVisibility();
    updateUndoState();
    updateStatus();

    aiController_ = std::make_unique<AIFillController>();
    connect(aiController_.get(), &AIFillController::completed, this, [this](const cv::Mat& result) {
        aiRun_->setEnabled(true);
        if (aiModels_) aiModels_->setEnabled(true);
        if (modelDirBtn_) modelDirBtn_->setEnabled(true);
        if (submittedGeneration_ != imageGeneration_) {
            aiStatus_->setText("Result discarded: image or result was cleared.");
            return;
        }
        doc_->aiFill().resultMask = result;
        doc_->aiFill().showResult = true;
        doc_->aiFill().showPrompt = false; // Hide prompts after segmentation completes
        aiShow_->setChecked(true);
        if (aiShowPrompt_) aiShowPrompt_->setChecked(false);
        if (useResultAsPromptBtn_) useResultAsPromptBtn_->setVisible(true);
        aiStatus_->setText(cv::countNonZero(result) ? "AI Fill complete. Prompts hidden. Apply to edit or export."
                                                  : "AI Fill complete: no foreground found.");
        updateAIPromptStatus();
        canvas_->update();
    });
    connect(aiController_.get(), &AIFillController::failed, this, [this](const QString& error) {
        aiRun_->setEnabled(true);
        if (aiModels_) aiModels_->setEnabled(true);
        if (modelDirBtn_) modelDirBtn_->setEnabled(true);
        aiStatus_->setText("AI Fill failed.");
        QMessageBox::warning(this, "AI Fill", error);
    });
}

MainWindow::~MainWindow() = default;

QWidget* MainWindow::buildSidebar() {
    // The controls live on an inner widget inside a scroll area so that when
    // the contextual panels grow (e.g. the scribble algorithms' seed workflow)
    // nothing gets pushed out of reach — overflow just scrolls.
    auto* side = new QWidget;
    side->setStyleSheet("background:#0f172a;");
    auto* v = new QVBoxLayout(side);
    v->setContentsMargins(24, 24, 24, 24);
    v->setSpacing(20);

    // Header
    auto* header = new QLabel("🦴  OrthoSeg");
    header->setStyleSheet("font-size:18px; font-weight:600; color:#f1f5f9;");
    v->addWidget(header);
    auto* sub = new QLabel("MEDICAL IMAGING");
    sub->setStyleSheet("color:#64748b; font-size:10px; letter-spacing:3px;");
    v->addWidget(sub);

    // --- Anatomy label selector (2x2 grid) ---
    v->addWidget(sectionLabel("Select Anatomy"));
    auto* labelGrid = new QGridLayout;
    labelGrid->setSpacing(8);
    const char* dotColors[4] = {"transparent", "#ef4444", "#22c55e", "#3b82f6"};
    for (int i = 0; i < 4; ++i) {
        Label lid = static_cast<Label>(i);
        auto* btn = new QPushButton(QString("     %1").arg(labelInfo(lid).name));
        btn->setCheckable(true);
        btn->setCursor(Qt::PointingHandCursor);
        btn->setMinimumHeight(40);
        btn->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        // Use an emoji-free color chip via border for background.
        btn->setStyleSheet(QString(
            "QPushButton{ text-align:left; padding-left:10px; border-radius:10px;"
            " border:1px solid transparent; color:#94a3b8; background:transparent;"
            " font-size:12px; }"
            "QPushButton:hover{ background:#1e293b80; }"
            "QPushButton:checked{ background:#1e293b; border:1px solid #334155;"
            " color:#ffffff; }"));
        labelButtons_[i] = btn;
        connect(btn, &QPushButton::clicked, this, [this, lid] { selectLabel(lid); });
        // Colored dot overlaid on the button's left padding.
        auto* chip = new QLabel(btn);
        chip->setFixedSize(10, 10);
        chip->setStyleSheet(QString("background:%1; border-radius:5px; %2")
            .arg(dotColors[i], i == 0 ? "border:1px dashed #64748b;" : ""));
        chip->move(10, 15);
        labelGrid->addWidget(btn, i / 2, i % 2);
    }
    v->addLayout(labelGrid);

    // --- Toolbox ---
    v->addWidget(sectionLabel("Toolbox"));
    auto* toolRow = new QHBoxLayout;
    toolRow->setSpacing(8);
    const char* toolNames[4] = {"Brush", "Fill", "Eraser", "AI Fill"};
    const char* toolIcons[4] = {"🖌", "🪣", "🧽", "AI"};
    for (int i = 0; i < 4; ++i) {
        Tool tid = static_cast<Tool>(i);
        auto* btn = new QPushButton(QString("%1\n%2").arg(toolIcons[i], toolNames[i]));
        btn->setCheckable(true);
        btn->setCursor(Qt::PointingHandCursor);
        btn->setMinimumHeight(56);
        if (i == 3) {
            btn->setObjectName("aiFillToolBtn");
            btn->setStyleSheet(
                "QPushButton{ border-radius:12px; border:1px solid #16a34a;"
                " background:#14532d66; color:#4ade80; font-size:11px; font-weight:600; }"
                "QPushButton:hover{ background:#16a34a4d; border:1px solid #22c55e; color:#86efac; }"
                "QPushButton:checked{ background:#22c55e; border:2px solid #16a34a;"
                " color:#0f172a; font-weight:700; }");
        } else {
            btn->setStyleSheet(QString(
                "QPushButton{ border-radius:12px; border:1px solid #1e293b;"
                " background:#1e293b4d; color:#94a3b8; font-size:11px; }"
                "QPushButton:hover{ border:1px solid #334155; }"
                "QPushButton:checked{ background:%1; border:1px solid %1;"
                " color:#0f172a; font-weight:600; }").arg(kAccent));
        }
        toolButtons_[i] = btn;
        connect(btn, &QPushButton::clicked, this, [this, tid] { selectTool(tid); });
        toolRow->addWidget(btn);
    }
    v->addLayout(toolRow);

    // Divider
    auto* divider = new QFrame;
    divider->setFrameShape(QFrame::HLine);
    divider->setStyleSheet("color:#1e293b;");
    v->addWidget(divider);

    // --- Brush size panel ---
    brushPanel_ = new QWidget;
    {
        auto* bl = new QVBoxLayout(brushPanel_);
        bl->setContentsMargins(0, 0, 0, 0);
        bl->setSpacing(8);
        auto* hdr = new QHBoxLayout;
        hdr->addWidget(sectionLabel("Brush Size"));
        brushValue_ = new QLabel("20px");
        brushValue_->setStyleSheet(QString("color:%1; font-size:12px;").arg(kAccent));
        hdr->addStretch();
        hdr->addWidget(brushValue_);
        bl->addLayout(hdr);
        brushSlider_ = new QSlider(Qt::Horizontal);
        brushSlider_->setRange(2, 100);
        brushSlider_->setValue(20);
        connect(brushSlider_, &QSlider::valueChanged, this, [this](int v) {
            canvas_->setBrushSize(v);
            brushValue_->setText(QString("%1px").arg(v));
        });
        bl->addWidget(brushSlider_);
    }
    v->addWidget(brushPanel_);

    // --- Fill panel ---
    fillPanel_ = new QWidget;
    {
        auto* fl = new QVBoxLayout(fillPanel_);
        fl->setContentsMargins(0, 0, 0, 0);
        fl->setSpacing(8);
        fl->addWidget(sectionLabel("Algorithm"));
        algoCombo_ = new QComboBox;
        algoCombo_->addItem("Standard Growing");     // click-seed
        algoCombo_->addItem("Embedded Boundary");    // click-seed
        algoCombo_->addItem("Split-and-Merge");      // click-seed
        algoCombo_->addItem("Grow from Seeds");      // scribble: GrowCut
        algoCombo_->addItem("Random Walker");        // scribble
        algoCombo_->addItem("Graph Cut");            // scribble: grabCut
        algoCombo_->setStyleSheet(
            "QComboBox{ background:#1e293b; border:1px solid #334155;"
            " border-radius:8px; padding:6px 10px; color:#cbd5e1; }");
        connect(algoCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
                this, [this](int idx) {
            canvas_->setFillAlgorithm(static_cast<FillAlgorithm>(idx));
            updateSettingsVisibility();
        });
        fl->addWidget(algoCombo_);

        intensityPanel_ = new QWidget;
        auto* ipl = new QVBoxLayout(intensityPanel_);
        ipl->setContentsMargins(0, 0, 0, 0);
        ipl->setSpacing(8);
        auto* ihdr = new QHBoxLayout;
        ihdr->addWidget(sectionLabel("Intensity Thresh"));
        intensityValue_ = new QLabel("5");
        intensityValue_->setStyleSheet(QString("color:%1; font-size:12px;").arg(kAccent));
        ihdr->addStretch();
        ihdr->addWidget(intensityValue_);
        ipl->addLayout(ihdr);
        intensitySlider_ = new QSlider(Qt::Horizontal);
        intensitySlider_->setRange(1, 50);
        intensitySlider_->setValue(5);
        connect(intensitySlider_, &QSlider::valueChanged, this, [this](int v) {
            canvas_->setIntensityThreshold(v);
            intensityValue_->setText(QString::number(v));
        });
        ipl->addWidget(intensitySlider_);
        fl->addWidget(intensityPanel_);

        edgePanel_ = new QWidget;
        auto* el = new QVBoxLayout(edgePanel_);
        el->setContentsMargins(0, 0, 0, 0);
        el->setSpacing(8);
        auto* ehdr = new QHBoxLayout;
        ehdr->addWidget(sectionLabel("Edge Penalty"));
        edgeValue_ = new QLabel("30");
        edgeValue_->setStyleSheet(QString("color:%1; font-size:12px;").arg(kAccent));
        ehdr->addStretch();
        ehdr->addWidget(edgeValue_);
        el->addLayout(ehdr);
        edgeSlider_ = new QSlider(Qt::Horizontal);
        edgeSlider_->setRange(1, 255);
        edgeSlider_->setValue(30);
        connect(edgeSlider_, &QSlider::valueChanged, this, [this](int v) {
            canvas_->setEdgePenaltyThreshold(v);
            edgeValue_->setText(QString::number(v));
        });
        el->addWidget(edgeSlider_);
        fl->addWidget(edgePanel_);

        // --- Edge sensitivity (β) for scribble algorithms ---
        betaPanel_ = new QWidget;
        auto* bpl = new QVBoxLayout(betaPanel_);
        bpl->setContentsMargins(0, 0, 0, 0);
        bpl->setSpacing(8);
        auto* bhdr = new QHBoxLayout;
        bhdr->addWidget(sectionLabel("Edge Sensitivity (β)"));
        betaValue_ = new QLabel("30");
        betaValue_->setStyleSheet(QString("color:%1; font-size:12px;").arg(kAccent));
        bhdr->addStretch();
        bhdr->addWidget(betaValue_);
        bpl->addLayout(bhdr);
        betaSlider_ = new QSlider(Qt::Horizontal);
        betaSlider_->setRange(1, 100);
        betaSlider_->setValue(30);
        connect(betaSlider_, &QSlider::valueChanged, this, [this](int v) {
            betaValue_->setText(QString::number(v));
        });
        bpl->addWidget(betaSlider_);
        fl->addWidget(betaPanel_);

        // Note explaining Graph Cut's foreground/background convention.
        graphCutNote_ = new QLabel(
            "Graph Cut segments the active label as foreground; scribble other "
            "labels (or Background) as background.");
        graphCutNote_->setWordWrap(true);
        graphCutNote_->setStyleSheet("color:#64748b; font-size:11px;");
        fl->addWidget(graphCutNote_);

        // --- Seed workflow panel: hint + Run + Clear Seeds ---
        seedPanel_ = new QWidget;
        auto* spl = new QVBoxLayout(seedPanel_);
        spl->setContentsMargins(0, 8, 0, 0);
        spl->setSpacing(8);
        auto* hint = new QLabel(
            "Scribble seeds inside each structure and on the background / "
            "neighbouring bone, then Run.");
        hint->setWordWrap(true);
        hint->setStyleSheet("color:#94a3b8; font-size:11px;");
        spl->addWidget(hint);

        auto* runBtn = new QPushButton("⚡  Run Segmentation");
        runBtn->setCursor(Qt::PointingHandCursor);
        runBtn->setMinimumHeight(40);
        runBtn->setStyleSheet(QString(
            "QPushButton{ background:%1; color:#0f172a; font-weight:600;"
            " border-radius:12px; } QPushButton:hover{ background:#0ea5e9; }")
            .arg(kAccent));
        connect(runBtn, &QPushButton::clicked, this, &MainWindow::onRunSegmentation);
        spl->addWidget(runBtn);

        auto* clearSeedsBtn = new QPushButton("Clear Seeds");
        clearSeedsBtn->setCursor(Qt::PointingHandCursor);
        clearSeedsBtn->setMinimumHeight(36);
        clearSeedsBtn->setStyleSheet(
            "QPushButton{ background:#1e293b; color:#94a3b8;"
            " border:1px solid #334155; border-radius:12px; }"
            "QPushButton:hover{ color:#e2e8f0; border:1px solid #475569; }");
        connect(clearSeedsBtn, &QPushButton::clicked, this, &MainWindow::onClearSeeds);
        spl->addWidget(clearSeedsBtn);
        fl->addWidget(seedPanel_);
    }
    v->addWidget(fillPanel_);
    aiPanel_ = buildAIPanel();
    v->addWidget(aiPanel_);

    // --- Opacity (global) ---
    {
        auto* ohdr = new QHBoxLayout;
        ohdr->addWidget(sectionLabel("Mask Opacity"));
        opacityValue_ = new QLabel("50%");
        opacityValue_->setStyleSheet(QString("color:%1; font-size:12px;").arg(kAccent));
        ohdr->addStretch();
        ohdr->addWidget(opacityValue_);
        v->addLayout(ohdr);
        opacitySlider_ = new QSlider(Qt::Horizontal);
        opacitySlider_->setRange(10, 100);
        opacitySlider_->setValue(50);
        connect(opacitySlider_, &QSlider::valueChanged, this, [this](int v) {
            canvas_->setMaskOpacity(v / 100.0f);
            opacityValue_->setText(QString("%1%").arg(v));
        });
        v->addWidget(opacitySlider_);
    }

    v->addStretch();

    // --- Action footer ---
    auto* exportBtn = new QPushButton("⬇  Export Mask");
    exportBtn->setCursor(Qt::PointingHandCursor);
    exportBtn->setMinimumHeight(44);
    exportBtn->setStyleSheet(QString(
        "QPushButton{ background:%1; color:#0f172a; font-weight:600;"
        " border-radius:12px; } QPushButton:hover{ background:#0ea5e9; }")
        .arg(kAccent));
    connect(exportBtn, &QPushButton::clicked, this, &MainWindow::onExport);
    v->addWidget(exportBtn);

    auto* uploadBtn = new QPushButton("⬆  Upload X-ray");
    uploadBtn->setCursor(Qt::PointingHandCursor);
    uploadBtn->setMinimumHeight(44);
    uploadBtn->setStyleSheet(
        "QPushButton{ background:#1e293b; color:#cbd5e1; border:1px solid #334155;"
        " border-radius:12px; } QPushButton:hover{ background:#334155; }");
    connect(uploadBtn, &QPushButton::clicked, this, &MainWindow::onUpload);
    v->addWidget(uploadBtn);

    auto* clearBtn = new QPushButton("🗑  Clear All");
    clearBtn->setCursor(Qt::PointingHandCursor);
    clearBtn->setMinimumHeight(44);
    clearBtn->setStyleSheet(
        "QPushButton{ background:#1e293b; color:#94a3b8; border:1px solid #334155;"
        " border-radius:12px; } QPushButton:hover{ color:#f87171;"
        " border:1px solid #ef444480; }");
    connect(clearBtn, &QPushButton::clicked, this, &MainWindow::onClear);
    v->addWidget(clearBtn);

    auto* scroll = new QScrollArea;
    scroll->setWidget(side);
    scroll->setWidgetResizable(true);
    scroll->setFixedWidth(288);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    scroll->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    scroll->setStyleSheet(
        "QScrollArea{ background:#0f172a; border:none;"
        " border-right:1px solid #1e293b; }"
        "QScrollBar:vertical{ background:transparent; width:8px; margin:0; }"
        "QScrollBar::handle:vertical{ background:#334155; border-radius:4px;"
        " min-height:30px; }"
        "QScrollBar::handle:vertical:hover{ background:#475569; }"
        "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical{ height:0; }"
        "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical{"
        " background:transparent; }");
    return scroll;
}

QWidget* MainWindow::buildTopBar() {
    auto* bar = new QWidget;
    bar->setFixedHeight(56);
    bar->setStyleSheet("background:#0f172a; border-bottom:1px solid #1e293b;");
    auto* h = new QHBoxLayout(bar);
    h->setContentsMargins(24, 0, 24, 0);

    auto* status = new QLabel("● ACTIVE SESSION");
    status->setStyleSheet("color:#22c55e; font-size:10px; letter-spacing:2px;"
                          " font-weight:700;");
    h->addWidget(status);
    h->addStretch();

    dimLabel_ = new QLabel("NO_IMAGE");
    dimLabel_->setStyleSheet("color:#64748b; font-size:10px; font-family:monospace;");
    h->addWidget(dimLabel_);

    zoomLabel_ = new QLabel("ZOOM 100%");
    zoomLabel_->setStyleSheet("color:#64748b; font-size:10px; font-family:monospace;");
    h->addWidget(zoomLabel_);

    auto makeIconBtn = [](const QString& txt, const QString& tip) {
        auto* b = new QPushButton(txt);
        b->setToolTip(tip);
        b->setCursor(Qt::PointingHandCursor);
        b->setFixedSize(36, 36);
        b->setStyleSheet(
            "QPushButton{ background:transparent; color:#94a3b8;"
            " border-radius:8px; font-size:16px; }"
            "QPushButton:hover{ background:#ffffff0d; }"
            "QPushButton:disabled{ color:#334155; }");
        return b;
    };

    claheCheck_ = new QCheckBox("CLAHE");
    claheCheck_->setObjectName("claheCheck");
    claheCheck_->setToolTip("Toggle Contrast Limited Adaptive Histogram Equalization for enhanced visibility.\nClick to toggle or configure parameters.");
    claheCheck_->setCursor(Qt::PointingHandCursor);
    claheCheck_->setStyleSheet(
        "QCheckBox{ color:#cbd5e1; font-size:11px; font-weight:600; spacing:6px; padding:6px 10px;"
        " background:#1e293b; border:1px solid #334155; border-radius:8px; }"
        "QCheckBox:hover{ background:#334155; border:1px solid #475569; }"
        "QCheckBox::indicator{ width:14px; height:14px; border-radius:3px; border:1px solid #475569; background:#0f172a; }"
        "QCheckBox::indicator:checked{ background:#38bdf8; border:1px solid #38bdf8; }");
    connect(claheCheck_, &QCheckBox::clicked, this, [this](bool checked) {
        canvas_->setClaheEnabled(checked);
        if (checked) {
            onOpenClaheDialog();
        } else {
            if (claheDialog_) claheDialog_->hide();
        }
    });
    h->addWidget(claheCheck_);

    claheSettingsBtn_ = makeIconBtn("⚙", "Configure CLAHE Parameters (Clip Limit, Grid Size)");
    claheSettingsBtn_->setObjectName("claheSettingsBtn");
    connect(claheSettingsBtn_, &QPushButton::clicked, this, &MainWindow::onOpenClaheDialog);
    h->addWidget(claheSettingsBtn_);

    modelDirBtn_ = new QPushButton("🧠 MedSAM2 Models");
    modelDirBtn_->setObjectName("modelDirBtn");
    modelDirBtn_->setToolTip(QString("MedSAM2 Model Directory: %1\nClick to change directory").arg(aiModels_->text()));
    modelDirBtn_->setCursor(Qt::PointingHandCursor);
    modelDirBtn_->setStyleSheet(
        "QPushButton{ background:#1e293b; color:#cbd5e1; border:1px solid #334155;"
        " border-radius:8px; padding:6px 12px; font-size:11px; font-weight:600; }"
        "QPushButton:hover{ background:#334155; color:#ffffff; border:1px solid #475569; }"
        "QPushButton:disabled{ color:#475569; }");
    connect(modelDirBtn_, &QPushButton::clicked, this, &MainWindow::onOpenModelDirDialog);
    h->addWidget(modelDirBtn_);

    undoBtn_ = makeIconBtn("↺", "Undo");
    connect(undoBtn_, &QPushButton::clicked, this, &MainWindow::onUndo);
    h->addWidget(undoBtn_);

    auto* zoomOut = makeIconBtn("−", "Zoom Out");
    connect(zoomOut, &QPushButton::clicked, this, [this] { canvas_->zoomOut(); });
    h->addWidget(zoomOut);

    auto* zoomIn = makeIconBtn("+", "Zoom In");
    connect(zoomIn, &QPushButton::clicked, this, [this] { canvas_->zoomIn(); });
    h->addWidget(zoomIn);

    auto* zoomReset = makeIconBtn("⛶", "Reset Zoom");
    connect(zoomReset, &QPushButton::clicked, this, [this] { canvas_->zoomReset(); });
    h->addWidget(zoomReset);

    return bar;
}

QWidget* MainWindow::buildAIPanel() {
    auto* panel = new QWidget;
    panel->setStyleSheet(
        "QPushButton, QComboBox, QLineEdit{ background:#1e293b; color:#cbd5e1;"
        " border:1px solid #334155; border-radius:8px; padding:6px; font-size:11px; }"
        "QPushButton:hover{ background:#334155; }"
        "QPushButton:disabled, QLineEdit:disabled{ color:#64748b; }"
        "QLabel, QCheckBox{ font-size:11px; }");
    auto* layout = new QVBoxLayout(panel);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(8);
    layout->addWidget(sectionLabel("AI Fill · Prompt Type"));
    aiPromptCombo_ = new QComboBox;
    aiPromptCombo_->setObjectName("aiPromptType");
    aiPromptCombo_->addItem("Bounding Box");
    aiPromptCombo_->addItem("Paint Mask");
    aiPromptCombo_->addItem("Load Mask");
    aiPromptCombo_->addItem("Normal Fill Mask");
    connect(aiPromptCombo_, &QComboBox::currentIndexChanged, this, [this](int index) {
        auto type = static_cast<AIFillPromptType>(index);
        doc_->aiFill().promptType = type;
        if (type == AIFillPromptType::NormalFillMask) {
            if (doc_->hasImage()) {
                doc_->aiFill().promptMask = doc_->mask().empty() ?
                    cv::Mat::zeros(doc_->sourceColor().size(), CV_8UC1) : doc_->mask().clone();
                doc_->aiFill().promptType = AIFillPromptType::PaintedMask;
                doc_->aiFill().showPrompt = true;
                if (aiShowPrompt_) aiShowPrompt_->setChecked(true);
                aiPromptCombo_->blockSignals(true);
                aiPromptCombo_->setCurrentIndex(static_cast<int>(AIFillPromptType::PaintedMask));
                aiPromptCombo_->blockSignals(false);
                if (cv::countNonZero(doc_->aiFill().promptMask) > 0) {
                    aiStatus_->setText("Normal fill mask converted to paint prompt format.");
                } else {
                    aiStatus_->setText("Normal fill mask converted (empty). Paint strokes or fill regions.");
                }
                canvas_->update();
            } else {
                aiStatus_->setText("Load an image first before selecting Normal Fill Mask.");
            }
        }
        if (aiPromptHint_) {
            switch (doc_->aiFill().promptType) {
            case AIFillPromptType::BoundingBox:
                aiPromptHint_->setText("Select Femur (Red) or Tibia (Green) above to draw bounding boxes. "
                                       "Both boxes can be drawn and segmented together.");
                break;
            case AIFillPromptType::PaintedMask:
                aiPromptHint_->setText("Paint prompt strokes with brush or eraser. "
                                       "Uses anatomy color coding (Femur: Red, Tibia: Green).");
                break;
            case AIFillPromptType::LoadedMask:
                aiPromptHint_->setText("Load an external binary or labeled prompt mask from disk.");
                break;
            case AIFillPromptType::NormalFillMask:
                aiPromptHint_->setText("Uses current normal fill annotations as the prompt for MedSAM2 AI refinement.");
                break;
            }
        }
        canvas_->setActiveTool(activeTool_);
        updateSettingsVisibility();
    });
    layout->addWidget(aiPromptCombo_);

    aiPromptHint_ = new QLabel("Select Femur (Red) or Tibia (Green) above to draw bounding boxes. "
                               "Both boxes can be drawn and segmented together.");
    aiPromptHint_->setWordWrap(true);
    aiPromptHint_->setMinimumHeight(48);
    layout->addWidget(aiPromptHint_);

    // Bounding box controls for Femur and Tibia
    aiBoxControls_ = new QWidget;
    auto* bcl = new QVBoxLayout(aiBoxControls_);
    bcl->setContentsMargins(0, 0, 0, 0);
    bcl->setSpacing(6);

    auto* fRow = new QHBoxLayout;
    femurBoxStatus_ = new QLabel("Femur Box (Red): Not set");
    femurBoxStatus_->setStyleSheet("color:#ef4444; font-size:11px; font-weight:600;");
    fRow->addWidget(femurBoxStatus_, 1);
    clearFemurBoxBtn_ = new QPushButton("Clear");
    clearFemurBoxBtn_->setFixedSize(50, 24);
    connect(clearFemurBoxBtn_, &QPushButton::clicked, this, [this] {
        doc_->aiFill().femurBox.reset();
        if (activeLabel_ == Label::Femur) doc_->aiFill().box.reset();
        updateAIPromptStatus();
        canvas_->update();
    });
    fRow->addWidget(clearFemurBoxBtn_);
    bcl->addLayout(fRow);

    auto* tRow = new QHBoxLayout;
    tibiaBoxStatus_ = new QLabel("Tibia Box (Green): Not set");
    tibiaBoxStatus_->setStyleSheet("color:#22c55e; font-size:11px; font-weight:600;");
    tRow->addWidget(tibiaBoxStatus_, 1);
    clearTibiaBoxBtn_ = new QPushButton("Clear");
    clearTibiaBoxBtn_->setFixedSize(50, 24);
    connect(clearTibiaBoxBtn_, &QPushButton::clicked, this, [this] {
        doc_->aiFill().tibiaBox.reset();
        if (activeLabel_ == Label::Tibia) doc_->aiFill().box.reset();
        updateAIPromptStatus();
        canvas_->update();
    });
    tRow->addWidget(clearTibiaBoxBtn_);
    bcl->addLayout(tRow);
    layout->addWidget(aiBoxControls_);

    aiShowPrompt_ = new QCheckBox("Show Prompt (Box / Mask)");
    aiShowPrompt_->setChecked(true);
    connect(aiShowPrompt_, &QCheckBox::toggled, this, [this](bool show) {
        doc_->aiFill().showPrompt = show;
        canvas_->update();
    });
    layout->addWidget(aiShowPrompt_);

    aiErase_ = new QCheckBox("Erase Prompt (unchecked = brush)");
    connect(aiErase_, &QCheckBox::toggled, canvas_, &CanvasWidget::setAIPromptErase);
    layout->addWidget(aiErase_);
    aiLoadMask_ = new QPushButton("Load Prompt Mask");
    connect(aiLoadMask_, &QPushButton::clicked, this, &MainWindow::onLoadPromptMask);
    layout->addWidget(aiLoadMask_);

    // Normal Fill Mask controls
    normalMaskControls_ = new QWidget;
    auto* nml = new QVBoxLayout(normalMaskControls_);
    nml->setContentsMargins(0, 0, 0, 0);
    nml->setSpacing(6);
    normalMaskStatus_ = new QLabel("Uses current normal fill mask as prompt.");
    normalMaskStatus_->setWordWrap(true);
    normalMaskStatus_->setStyleSheet("color:#94a3b8; font-size:11px;");
    nml->addWidget(normalMaskStatus_);
    copyNormalMaskBtn_ = new QPushButton("Copy Normal Mask to Paint Prompt");
    copyNormalMaskBtn_->setToolTip("Copy normal fill annotations to the paint prompt layer for manual editing.");
    connect(copyNormalMaskBtn_, &QPushButton::clicked, this, [this] {
        if (!doc_->hasImage() || doc_->mask().empty()) return;
        doc_->aiFill().promptMask = doc_->mask().clone();
        doc_->aiFill().promptType = AIFillPromptType::PaintedMask;
        doc_->aiFill().showPrompt = true;
        if (aiShowPrompt_) aiShowPrompt_->setChecked(true);
        aiPromptCombo_->setCurrentIndex(static_cast<int>(AIFillPromptType::PaintedMask));
        canvas_->update();
    });
    nml->addWidget(copyNormalMaskBtn_);
    layout->addWidget(normalMaskControls_);

    useResultAsPromptBtn_ = new QPushButton("Use AI Result as Next Prompt");
    useResultAsPromptBtn_->setToolTip("Use the current AI segmentation output as the prompt for the next refinement pass.");
    connect(useResultAsPromptBtn_, &QPushButton::clicked, this, [this] {
        if (doc_->aiFill().resultMask.empty() || cv::countNonZero(doc_->aiFill().resultMask) == 0) {
            QMessageBox::information(this, "AI Fill", "No AI segmentation result to use as prompt.");
            return;
        }
        doc_->aiFill().promptMask = doc_->aiFill().resultMask.clone();
        doc_->aiFill().promptType = AIFillPromptType::PaintedMask;
        doc_->aiFill().showPrompt = true;
        if (aiShowPrompt_) aiShowPrompt_->setChecked(true);
        aiPromptCombo_->setCurrentIndex(static_cast<int>(AIFillPromptType::PaintedMask));
        aiStatus_->setText("AI segmentation result copied to prompt. Ready for refinement.");
        canvas_->update();
    });
    layout->addWidget(useResultAsPromptBtn_);

    auto* clear = new QPushButton("Clear Prompt");
    connect(clear, &QPushButton::clicked, this, [this] {
        doc_->aiFill().box.reset();
        doc_->aiFill().femurBox.reset();
        doc_->aiFill().tibiaBox.reset();
        doc_->aiFill().promptMask.release();
        doc_->aiFill().showPrompt = true;
        if (aiShowPrompt_) aiShowPrompt_->setChecked(true);
        updateAIPromptStatus();
        canvas_->setActiveTool(activeTool_);
    });
    layout->addWidget(clear);

    aiModels_->setVisible(false);
    layout->addWidget(aiModels_);

    aiRun_ = new QPushButton("▶ Run AI Fill");
    aiRun_->setObjectName("runAIFill");
    aiRun_->setMinimumHeight(38);
    aiRun_->setStyleSheet(
        "QPushButton{ background:#22c55e; color:#0f172a; font-weight:700; border:none; border-radius:8px; font-size:12px; letter-spacing:0.5px; }"
        "QPushButton:hover{ background:#16a34a; color:#ffffff; }"
        "QPushButton:disabled{ background:#1e293b; color:#64748b; border:1px solid #334155; }");
    connect(aiRun_, &QPushButton::clicked, this, &MainWindow::onRunAIFill);
    layout->addWidget(aiRun_);
    aiStatus_ = new QLabel("Ready. CUDA required.");
    aiStatus_->setObjectName("aiStatus");
    aiStatus_->setWordWrap(true);
    aiStatus_->setMinimumHeight(42);
    layout->addWidget(aiStatus_);
    aiShow_ = new QCheckBox("Show AI segmentation");
    aiShow_->setChecked(true);
    connect(aiShow_, &QCheckBox::toggled, this, [this](bool show) {
        doc_->aiFill().showResult = show;
        canvas_->update();
    });
    layout->addWidget(aiShow_);
    auto* clearResult = new QPushButton("Clear AI Segmentation");
    connect(clearResult, &QPushButton::clicked, this, [this] {
        ++imageGeneration_; // Also invalidate any pending result.
        doc_->aiFill().resultMask.release();
        if (useResultAsPromptBtn_) useResultAsPromptBtn_->setVisible(false);
        canvas_->update();
    });
    layout->addWidget(clearResult);
    auto* apply = new QPushButton("Apply AI Result to Mask");
    apply->setToolTip("Copy the preview foreground into the editable mask, with undo. Apply before export.");
    connect(apply, &QPushButton::clicked, this, [this] {
        doc_->applyAIResult();
        if (doc_->hasImage() && !doc_->mask().empty() && cv::countNonZero(doc_->mask()) > 0) {
            doc_->aiFill().promptMask = doc_->mask().clone();
            doc_->aiFill().promptType = AIFillPromptType::PaintedMask;
            doc_->aiFill().showPrompt = true;
            if (aiShowPrompt_) aiShowPrompt_->setChecked(true);
            if (aiPromptCombo_) {
                aiPromptCombo_->blockSignals(true);
                aiPromptCombo_->setCurrentIndex(static_cast<int>(AIFillPromptType::PaintedMask));
                aiPromptCombo_->blockSignals(false);
            }
        }
        canvas_->update();
        updateUndoState();
    });
    layout->addWidget(apply);
    return panel;
}

void MainWindow::onRunAIFill() {
    if (aiController_->running()) return;
    try {
        if (!doc_->hasImage()) throw std::runtime_error("Load an image first.");
        const auto& ai = doc_->aiFill();
        AIFillRequest request;
        request.imageBGR = doc_->sourceColor();
        request.target = activeLabel_;
        request.type = ai.promptType;
        request.modelDirectory = aiModels_->text().toStdString();

        if (request.type == AIFillPromptType::BoundingBox) {
            request.femurBox = ai.femurBox;
            request.tibiaBox = ai.tibiaBox;
            request.box = ai.box;
            if (!request.femurBox && !request.tibiaBox && !request.box) {
                if (!ai.promptMask.empty() && cv::countNonZero(ai.promptMask) > 0) {
                    request.type = AIFillPromptType::PaintedMask;
                    request.promptMask = ai.promptMask.clone();
                } else if (!doc_->mask().empty() && cv::countNonZero(doc_->mask()) > 0) {
                    request.type = AIFillPromptType::PaintedMask;
                    request.promptMask = doc_->mask().clone();
                } else if (!ai.resultMask.empty() && cv::countNonZero(ai.resultMask) > 0) {
                    request.type = AIFillPromptType::PaintedMask;
                    request.promptMask = ai.resultMask.clone();
                } else {
                    throw std::runtime_error("Draw at least one bounding box (Femur or Tibia) first.");
                }
            } else {
                if (request.femurBox)
                    request.femurBox = validatedBox(*request.femurBox, request.imageBGR.size());
                if (request.tibiaBox)
                    request.tibiaBox = validatedBox(*request.tibiaBox, request.imageBGR.size());
                if (!request.femurBox && !request.tibiaBox && request.box)
                    request.box = validatedBox(*request.box, request.imageBGR.size());
            }
        } else if (request.type == AIFillPromptType::NormalFillMask) {
            if (cv::countNonZero(doc_->mask()) == 0)
                throw std::runtime_error("Normal fill mask has no annotations yet. Fill or paint some bone regions first.");
            doc_->aiFill().promptMask = doc_->mask().clone();
            doc_->aiFill().promptType = AIFillPromptType::PaintedMask;
            request.promptMask = doc_->aiFill().promptMask.clone();
            request.type = AIFillPromptType::PaintedMask;
            aiPromptCombo_->blockSignals(true);
            aiPromptCombo_->setCurrentIndex(static_cast<int>(AIFillPromptType::PaintedMask));
            aiPromptCombo_->blockSignals(false);
        } else {
            if (activeLabel_ != Label::Femur && activeLabel_ != Label::Tibia)
                throw std::runtime_error("Select Femur or Tibia under Select Anatomy.");
            request.promptMask = ai.promptMask.clone();
            if ((request.promptMask.empty() || cv::countNonZero(request.promptMask) == 0) &&
                !doc_->mask().empty() && cv::countNonZero(doc_->mask()) > 0) {
                request.promptMask = doc_->mask().clone();
            } else if ((request.promptMask.empty() || cv::countNonZero(request.promptMask) == 0) &&
                !ai.resultMask.empty() && cv::countNonZero(ai.resultMask) > 0) {
                request.promptMask = ai.resultMask.clone();
            }
            if (request.promptMask.empty() || cv::countNonZero(request.promptMask) == 0)
                throw std::runtime_error("Paint or load a non-empty prompt mask first.");
        }
        submittedGeneration_ = imageGeneration_;
        aiController_->run(std::move(request));
        aiRun_->setEnabled(false);
        if (aiModels_) aiModels_->setEnabled(false);
        if (modelDirBtn_) modelDirBtn_->setEnabled(false);
        aiStatus_->setText("Running AI Fill… Loading models on first use.");
    } catch (const std::exception& error) {
        QMessageBox::warning(this, "AI Fill", QString::fromUtf8(error.what()));
    }
}

void MainWindow::onLoadPromptMask() {
    if (!doc_->hasImage()) {
        QMessageBox::information(this, "AI Fill", "Load an image first.");
        return;
    }
    const auto path = QFileDialog::getOpenFileName(this, "Load Prompt Mask", {},
        "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)");
    if (path.isEmpty()) return;
    try {
        // Preserve bit depth: a 16-bit label value of 1 is foreground, not zero.
        cv::Mat raw = cv::imread(path.toStdString(), cv::IMREAD_UNCHANGED);
        cv::Mat binary = binaryPromptMask(raw, doc_->sourceColor().size());
        doc_->aiFill().promptMask = std::move(binary);
        doc_->aiFill().showPrompt = true;
        if (aiShowPrompt_) aiShowPrompt_->setChecked(true);
        aiPromptCombo_->setCurrentIndex(static_cast<int>(AIFillPromptType::LoadedMask));
        aiStatus_->setText("Prompt loaded: nonzero pixels mark the selected anatomy.");
        canvas_->update();
    } catch (const std::exception& error) {
        QMessageBox::warning(this, "AI Fill", QString::fromUtf8(error.what()));
    }
}

void MainWindow::onOpenModelDirDialog() {
    if (aiController_->running()) {
        QMessageBox::information(this, "MedSAM2 Models", "Cannot change model directory while AI Fill is running.");
        return;
    }
    QDialog dlg(this);
    dlg.setWindowTitle("MedSAM2 Model Directory");
    dlg.setMinimumWidth(540);
    dlg.setStyleSheet(
        "QDialog { background:#0f172a; color:#e2e8f0; }"
        "QPushButton { background:#1e293b; color:#cbd5e1; border:1px solid #334155; border-radius:8px; padding:6px 14px; font-size:11px; }"
        "QPushButton:hover { background:#334155; color:#ffffff; }"
        "QLineEdit { background:#1e293b; color:#f1f5f9; border:1px solid #334155; border-radius:8px; padding:6px 10px; font-size:11px; }");

    auto* l = new QVBoxLayout(&dlg);
    l->setContentsMargins(20, 20, 20, 20);
    l->setSpacing(12);

    auto* title = new QLabel("🧠 MedSAM2 ONNX Model Configuration", &dlg);
    title->setStyleSheet("font-size:14px; font-weight:700; color:#f1f5f9;");
    l->addWidget(title);

    auto* desc = new QLabel("Select the folder containing medsam2_image_encoder.onnx and medsam2_mask_decoder.onnx:", &dlg);
    desc->setStyleSheet("color:#94a3b8; font-size:11px;");
    desc->setWordWrap(true);
    l->addWidget(desc);

    auto* editRow = new QHBoxLayout;
    auto* pathEdit = new QLineEdit(aiModels_->text(), &dlg);
    editRow->addWidget(pathEdit, 1);

    auto* browseBtn = new QPushButton("Browse…", &dlg);
    browseBtn->setStyleSheet("background:#38bdf8; color:#0f172a; font-weight:600; border-radius:8px; padding:6px 14px;");
    editRow->addWidget(browseBtn);
    l->addLayout(editRow);

    auto* statusCard = new QLabel(&dlg);
    statusCard->setWordWrap(true);
    statusCard->setStyleSheet("background:#1e293b; border:1px solid #334155; border-radius:8px; padding:10px 12px; font-size:11px;");
    l->addWidget(statusCard);

    auto checkDir = [pathEdit, statusCard]() {
        const auto dir = std::filesystem::path(pathEdit->text().toStdString());
        const auto enc = dir / "medsam2_image_encoder.onnx";
        const auto dec = dir / "medsam2_mask_decoder.onnx";
        bool encOk = std::filesystem::is_regular_file(enc);
        bool decOk = std::filesystem::is_regular_file(dec);
        bool cudaOk = cv::cuda::getCudaEnabledDeviceCount() > 0;

        QString msg;
        if (encOk && decOk) {
            msg = QString("<span style='color:#22c55e; font-weight:bold;'>✓ Valid MedSAM2 Models Found</span><br>"
                          "• medsam2_image_encoder.onnx: <span style='color:#22c55e;'>Present</span><br>"
                          "• medsam2_mask_decoder.onnx: <span style='color:#22c55e;'>Present</span><br>"
                          "• Hardware Acceleration: %1")
                  .arg(cudaOk ? "<span style='color:#22c55e;'>CUDA GPU Available</span>"
                              : "<span style='color:#f87171;'>CUDA Unavailable (CUDA required for AI Fill)</span>");
        } else {
            msg = QString("<span style='color:#f87171; font-weight:bold;'>✗ Incomplete Model Directory</span><br>"
                          "• medsam2_image_encoder.onnx: %1<br>"
                          "• medsam2_mask_decoder.onnx: %2")
                  .arg(encOk ? "<span style='color:#22c55e;'>Found</span>" : "<span style='color:#f87171;'>Missing</span>")
                  .arg(decOk ? "<span style='color:#22c55e;'>Found</span>" : "<span style='color:#f87171;'>Missing</span>");
        }
        statusCard->setText(msg);
    };

    checkDir();
    connect(pathEdit, &QLineEdit::textChanged, &dlg, checkDir);

    connect(browseBtn, &QPushButton::clicked, &dlg, [pathEdit, &dlg] {
        const auto dir = QFileDialog::getExistingDirectory(&dlg, "Choose MedSAM2 Model Directory", pathEdit->text());
        if (!dir.isEmpty()) {
            pathEdit->setText(dir);
        }
    });

    auto* btnRow = new QHBoxLayout;
    btnRow->addStretch();
    auto* cancelBtn = new QPushButton("Cancel", &dlg);
    connect(cancelBtn, &QPushButton::clicked, &dlg, &QDialog::reject);
    btnRow->addWidget(cancelBtn);

    auto* saveBtn = new QPushButton("Save Directory", &dlg);
    saveBtn->setStyleSheet("background:#38bdf8; color:#0f172a; font-weight:700; border-radius:8px; padding:6px 16px;");
    connect(saveBtn, &QPushButton::clicked, &dlg, &QDialog::accept);
    btnRow->addWidget(saveBtn);
    l->addLayout(btnRow);

    if (dlg.exec() == QDialog::Accepted) {
        aiModels_->setText(pathEdit->text());
        if (modelDirBtn_) {
            modelDirBtn_->setToolTip(QString("MedSAM2 Model Directory: %1\nClick to change directory").arg(pathEdit->text()));
        }
    }
}

class ClaheDialog : public QDialog {
public:
    bool userMoved = false;

    explicit ClaheDialog(QWidget* parent = nullptr)
        : QDialog(parent, Qt::Dialog | Qt::WindowTitleHint | Qt::WindowCloseButtonHint | Qt::CustomizeWindowHint) {
        setWindowTitle("CLAHE Display Enhancement");
        setObjectName("claheDialog");
        setModal(false);
    }

protected:
    void mousePressEvent(QMouseEvent* event) override {
        if (event->button() == Qt::LeftButton) {
            QWidget* child = childAt(event->position().toPoint());
            if (!child || qobject_cast<QLabel*>(child) || child == this) {
                dragPosition_ = event->globalPosition().toPoint() - frameGeometry().topLeft();
                dragging_ = true;
                event->accept();
                return;
            }
        }
        QDialog::mousePressEvent(event);
    }

    void mouseMoveEvent(QMouseEvent* event) override {
        if (dragging_ && (event->buttons() & Qt::LeftButton)) {
            userMoved = true;
            move(event->globalPosition().toPoint() - dragPosition_);
            event->accept();
            return;
        }
        QDialog::mouseMoveEvent(event);
    }

    void mouseReleaseEvent(QMouseEvent* event) override {
        if (event->button() == Qt::LeftButton) {
            dragging_ = false;
            event->accept();
            return;
        }
        QDialog::mouseReleaseEvent(event);
    }

private:
    QPoint dragPosition_;
    bool dragging_ = false;
};

void MainWindow::onOpenClaheDialog() {
    if (!claheDialog_) {
        claheDialog_ = new ClaheDialog(this);
        claheDialog_->setMinimumWidth(440);
        claheDialog_->setStyleSheet(
            "QDialog { background:#0f172a; color:#e2e8f0; border:1px solid #334155; border-radius:12px; }"
            "QPushButton { background:#1e293b; color:#cbd5e1; border:1px solid #334155; border-radius:8px; padding:6px 14px; font-size:11px; font-weight:600; }"
            "QPushButton:hover { background:#334155; color:#ffffff; border:1px solid #475569; }"
            "QSlider::groove:horizontal { height: 4px; background: #334155; border-radius: 2px; }"
            "QSlider::sub-page:horizontal { background: #38bdf8; border-radius: 2px; }"
            "QSlider::handle:horizontal { background: #f8fafc; border: 1px solid #94a3b8; width: 14px; margin-top: -5px; margin-bottom: -5px; border-radius: 7px; }"
            "QCheckBox { color: #cbd5e1; font-size: 12px; font-weight: 600; spacing: 8px; }");

        auto* l = new QVBoxLayout(claheDialog_);
        l->setContentsMargins(20, 20, 20, 20);
        l->setSpacing(14);

        auto* hdr = new QHBoxLayout;
        auto* title = new QLabel("🌓 CLAHE Contrast Enhancement", claheDialog_);
        title->setStyleSheet("font-size:14px; font-weight:700; color:#f1f5f9;");
        title->setCursor(Qt::SizeAllCursor);
        title->setToolTip("Click and drag anywhere on this window to move to the side");
        hdr->addWidget(title);
        hdr->addStretch();
        auto* dragHint = new QLabel("⠿ Drag to Move", claheDialog_);
        dragHint->setStyleSheet("color:#64748b; font-size:10px; font-weight:600; padding:2px 8px; background:#1e293b; border-radius:4px; border:1px solid #334155;");
        dragHint->setCursor(Qt::SizeAllCursor);
        dragHint->setToolTip("Click and drag anywhere on this window to move to the side");
        hdr->addWidget(dragHint);
        l->addLayout(hdr);

        auto* desc = new QLabel(
            "Contrast Limited Adaptive Histogram Equalization enhances local contrast and bone trabeculae visibility. "
            "This strictly affects visual rendering in the UI; all segmentation algorithms and AI models process the raw original image.", claheDialog_);
        desc->setStyleSheet("color:#94a3b8; font-size:11px;");
        desc->setWordWrap(true);
        l->addWidget(desc);

        auto* enableCb = new QCheckBox("Enable CLAHE Display", claheDialog_);
        enableCb->setObjectName("claheDialogEnableCb");
        enableCb->setChecked(canvas_->claheEnabled());
        connect(enableCb, &QCheckBox::toggled, this, [this](bool en) {
            canvas_->setClaheEnabled(en);
            if (claheCheck_) claheCheck_->setChecked(en);
        });
        l->addWidget(enableCb);

        // Clip Limit slider (0.5 to 10.0, step 0.1, internal slider 5..100)
        auto* clipHdr = new QHBoxLayout;
        auto* clipTitle = new QLabel("Clip Limit", claheDialog_);
        clipTitle->setStyleSheet("font-size:11px; font-weight:600; color:#cbd5e1;");
        auto* clipVal = new QLabel(QString::number(canvas_->claheClipLimit(), 'f', 1), claheDialog_);
        clipVal->setStyleSheet("color:#38bdf8; font-size:12px; font-weight:700;");
        clipHdr->addWidget(clipTitle);
        clipHdr->addStretch();
        clipHdr->addWidget(clipVal);
        l->addLayout(clipHdr);

        auto* clipSlider = new QSlider(Qt::Horizontal, claheDialog_);
        clipSlider->setObjectName("claheClipSlider");
        clipSlider->setRange(5, 100);
        clipSlider->setValue(static_cast<int>(std::round(canvas_->claheClipLimit() * 10.0)));
        connect(clipSlider, &QSlider::valueChanged, this, [this, clipVal](int v) {
            double val = v / 10.0;
            clipVal->setText(QString::number(val, 'f', 1));
            canvas_->setClaheParams(val, canvas_->claheGridSize());
        });
        l->addWidget(clipSlider);

        // Tile Grid Size slider (2 to 32, default 8)
        auto* gridHdr = new QHBoxLayout;
        auto* gridTitle = new QLabel("Tile Grid Size", claheDialog_);
        gridTitle->setStyleSheet("font-size:11px; font-weight:600; color:#cbd5e1;");
        auto* gridVal = new QLabel(QString("%1 × %1").arg(canvas_->claheGridSize()), claheDialog_);
        gridVal->setStyleSheet("color:#38bdf8; font-size:12px; font-weight:700;");
        gridHdr->addWidget(gridTitle);
        gridHdr->addStretch();
        gridHdr->addWidget(gridVal);
        l->addLayout(gridHdr);

        auto* gridSlider = new QSlider(Qt::Horizontal, claheDialog_);
        gridSlider->setObjectName("claheGridSlider");
        gridSlider->setRange(2, 32);
        gridSlider->setValue(canvas_->claheGridSize());
        connect(gridSlider, &QSlider::valueChanged, this, [this, gridVal](int v) {
            gridVal->setText(QString("%1 × %1").arg(v));
            canvas_->setClaheParams(canvas_->claheClipLimit(), v);
        });
        l->addWidget(gridSlider);

        // Bottom buttons
        auto* btnRow = new QHBoxLayout;
        auto* resetBtn = new QPushButton("Reset Defaults", claheDialog_);
        resetBtn->setObjectName("claheResetBtn");
        connect(resetBtn, &QPushButton::clicked, this, [this, clipSlider, gridSlider] {
            clipSlider->setValue(20);
            gridSlider->setValue(8);
            canvas_->setClaheParams(2.0, 8);
        });
        btnRow->addWidget(resetBtn);
        btnRow->addStretch();

        auto* doneBtn = new QPushButton("Done", claheDialog_);
        doneBtn->setObjectName("claheDoneBtn");
        doneBtn->setStyleSheet("background:#38bdf8; color:#0f172a; font-weight:700; border:none; border-radius:8px; padding:6px 16px;");
        connect(doneBtn, &QPushButton::clicked, claheDialog_, &QDialog::hide);
        btnRow->addWidget(doneBtn);
        l->addLayout(btnRow);

        connect(claheDialog_, &QDialog::finished, this, [this] {
            if (claheCheck_) claheCheck_->setChecked(canvas_->claheEnabled());
        });
    }

    auto* customDlg = static_cast<ClaheDialog*>(claheDialog_);
    if (!customDlg->userMoved) {
        // Automatically position dialog towards the top-right of the window so the central X-ray is unobstructed
        int targetX = mapToGlobal(QPoint(0, 0)).x() + width() - claheDialog_->width() - 24;
        int targetY = mapToGlobal(QPoint(0, 0)).y() + 68;
        claheDialog_->move(std::max(10, targetX), std::max(10, targetY));
    }

    if (auto* cb = claheDialog_->findChild<QCheckBox*>("claheDialogEnableCb")) {
        cb->setChecked(canvas_->claheEnabled());
    }
    if (auto* slider = claheDialog_->findChild<QSlider*>("claheClipSlider")) {
        slider->setValue(static_cast<int>(std::round(canvas_->claheClipLimit() * 10.0)));
    }
    if (auto* slider = claheDialog_->findChild<QSlider*>("claheGridSlider")) {
        slider->setValue(canvas_->claheGridSize());
    }

    claheDialog_->show();
    claheDialog_->raise();
    claheDialog_->activateWindow();
}

void MainWindow::showFillAlgorithm(int comboIndex) {
    selectTool(Tool::Fill);
    algoCombo_->setCurrentIndex(comboIndex);
}

void MainWindow::selectLabel(Label l) {
    activeLabel_ = l;
    canvas_->setActiveLabel(l);
    for (int i = 0; i < 4; ++i)
        static_cast<QPushButton*>(labelButtons_[i])
            ->setChecked(i == static_cast<int>(l));
    updateAIPromptStatus();
}

void MainWindow::selectTool(Tool t) {
    activeTool_ = t;
    canvas_->setActiveTool(t);
    for (int i = 0; i < 4; ++i)
        static_cast<QPushButton*>(toolButtons_[i])
            ->setChecked(i == static_cast<int>(t));
    updateSettingsVisibility();
}

void MainWindow::updateSettingsVisibility() {
    bool isFill = (activeTool_ == Tool::Fill);
    bool isAI = (activeTool_ == Tool::AIFill);
    aiPanel_->setVisible(isAI);
    bool isBox = isAI && doc_->aiFill().promptType == AIFillPromptType::BoundingBox;
    bool isPaint = isAI && doc_->aiFill().promptType == AIFillPromptType::PaintedMask;
    bool isLoad = isAI && doc_->aiFill().promptType == AIFillPromptType::LoadedMask;
    bool isNormalMask = isAI && doc_->aiFill().promptType == AIFillPromptType::NormalFillMask;
    if (aiBoxControls_) aiBoxControls_->setVisible(isBox);
    if (aiErase_) aiErase_->setVisible(isPaint);
    if (aiLoadMask_) aiLoadMask_->setVisible(isLoad);
    if (normalMaskControls_) normalMaskControls_->setVisible(isNormalMask);
    if (aiShowPrompt_) aiShowPrompt_->setVisible(isAI);
    if (useResultAsPromptBtn_) useResultAsPromptBtn_->setVisible(isAI && !doc_->aiFill().resultMask.empty());
    brushPanel_->setVisible((!isFill && !isAI) || isPaint);
    fillPanel_->setVisible(isFill);
    updateAIPromptStatus();
    if (!isFill || !algoCombo_) return;

    FillAlgorithm algo = static_cast<FillAlgorithm>(algoCombo_->currentIndex());
    bool scribble = isScribbleAlgorithm(algo);
    brushPanel_->setVisible(scribble);

    // Region-grow params only for the click-seed algorithms.
    intensityPanel_->setVisible(!scribble);
    edgePanel_->setVisible(!scribble && algo != FillAlgorithm::Standard);

    // Scribble params: β for GrowCut/RandomWalker (grabCut uses its own model).
    betaPanel_->setVisible(scribble && algo != FillAlgorithm::GraphCut);
    graphCutNote_->setVisible(algo == FillAlgorithm::GraphCut);
    seedPanel_->setVisible(scribble);
}

void MainWindow::updateAIPromptStatus() {
    if (!doc_) return;
    const auto& ai = doc_->aiFill();
    if (femurBoxStatus_) {
        if (ai.femurBox) {
            femurBoxStatus_->setText(QString("Femur (Red): [%1,%2]-%3x%4")
                .arg(int(ai.femurBox->x0)).arg(int(ai.femurBox->y0))
                .arg(int(ai.femurBox->x1 - ai.femurBox->x0 + 1))
                .arg(int(ai.femurBox->y1 - ai.femurBox->y0 + 1)));
            clearFemurBoxBtn_->setEnabled(true);
        } else {
            femurBoxStatus_->setText("Femur Box (Red): Not set");
            clearFemurBoxBtn_->setEnabled(false);
        }
    }
    if (tibiaBoxStatus_) {
        if (ai.tibiaBox) {
            tibiaBoxStatus_->setText(QString("Tibia (Grn): [%1,%2]-%3x%4")
                .arg(int(ai.tibiaBox->x0)).arg(int(ai.tibiaBox->y0))
                .arg(int(ai.tibiaBox->x1 - ai.tibiaBox->x0 + 1))
                .arg(int(ai.tibiaBox->y1 - ai.tibiaBox->y0 + 1)));
            clearTibiaBoxBtn_->setEnabled(true);
        } else {
            tibiaBoxStatus_->setText("Tibia Box (Green): Not set");
            clearTibiaBoxBtn_->setEnabled(false);
        }
    }
    if (normalMaskStatus_) {
        if (!doc_->hasImage() || doc_->mask().empty()) {
            normalMaskStatus_->setText("No image or mask loaded.");
        } else {
            int femurCount = cv::countNonZero(doc_->mask() == static_cast<int>(Label::Femur));
            int tibiaCount = cv::countNonZero(doc_->mask() == static_cast<int>(Label::Tibia));
            normalMaskStatus_->setText(QString("Normal mask ready: Femur (%1 px), Tibia (%2 px)")
                .arg(femurCount).arg(tibiaCount));
        }
    }
}

double MainWindow::currentBeta() const {
    // Slider 1..100 → small positive constant for exp(-beta*dI^2), dI in [0,255].
    return (betaSlider_ ? betaSlider_->value() : 30) * 1e-4;
}

void MainWindow::updateUndoState() {
    undoBtn_->setEnabled(doc_->canUndo());
}

void MainWindow::updateStatus() {
    zoomLabel_->setText(QString("ZOOM %1%")
        .arg(static_cast<int>(canvas_->zoom() * 100)));
    if (doc_->hasImage())
        dimLabel_->setText(QString("%1x%2").arg(doc_->width()).arg(doc_->height()));
    else
        dimLabel_->setText("NO_IMAGE");
}

void MainWindow::onUpload() {
    QString path = QFileDialog::getOpenFileName(
        this, "Open X-ray", QString(),
        "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)");
    if (path.isEmpty()) return;
    if (!doc_->loadImage(path.toStdString())) {
        QMessageBox::warning(this, "OrthoSeg", "Failed to load image.");
        return;
    }
    canvas_->zoomReset();
    ++imageGeneration_;
    aiPromptCombo_->setCurrentIndex(0);
    aiShow_->setChecked(true);
    canvas_->refresh();
    updateUndoState();
    updateStatus();
}

void MainWindow::onExport() {
    if (!doc_->hasImage()) {
        QMessageBox::information(this, "OrthoSeg", "Load an image first.");
        return;
    }
    QFileDialog dialog(this, "Export Mask");
    dialog.setAcceptMode(QFileDialog::AcceptSave);
    dialog.setNameFilter("PNG (*.png)");
    dialog.setDefaultSuffix("png");
    dialog.selectFile("bone_segmentation_mask.png");
    if (dialog.exec() != QDialog::Accepted || dialog.selectedFiles().isEmpty()) return;
    const QString path = dialog.selectedFiles().first();
    if (!doc_->exportMask(path.toStdString()))
        QMessageBox::warning(this, "OrthoSeg", "Failed to export mask.");
}

void MainWindow::onClear() {
    if (!doc_->hasImage()) return;
    doc_->pushHistory();
    doc_->clearMask();
    doc_->clearSeeds();
    ++imageGeneration_;
    doc_->aiFill().box.reset();
    doc_->aiFill().femurBox.reset();
    doc_->aiFill().tibiaBox.reset();
    doc_->aiFill().promptMask.release();
    doc_->aiFill().resultMask.release();
    doc_->aiFill().showPrompt = true;
    if (aiShowPrompt_) aiShowPrompt_->setChecked(true);
    if (useResultAsPromptBtn_) useResultAsPromptBtn_->setVisible(false);
    updateAIPromptStatus();
    canvas_->update();
    updateUndoState();
}

void MainWindow::onClearSeeds() {
    if (!doc_->hasImage()) return;
    doc_->pushHistory();
    doc_->clearSeeds();
    canvas_->update();
    updateUndoState();
}

void MainWindow::onRunSegmentation() {
    if (!doc_->hasImage()) {
        QMessageBox::information(this, "OrthoSeg", "Load an image first.");
        return;
    }
    FillAlgorithm algo = static_cast<FillAlgorithm>(algoCombo_->currentIndex());
    if (!isScribbleAlgorithm(algo)) return;

    if (!doc_->hasSeeds()) {
        QMessageBox::information(this, "OrthoSeg",
            "Scribble some seed strokes first: pick the Fill tool and draw "
            "inside each structure and on the background.");
        return;
    }

    if (algo == FillAlgorithm::GraphCut) {
        if (activeLabel_ == Label::Background) {
            QMessageBox::information(this, "OrthoSeg",
                "Select the bone you want to segment before running Graph Cut. "
                "Background strokes will be used as background seeds.");
            return;
        }
        if (!doc_->hasSeedForLabel(activeLabel_)) {
            QMessageBox::information(this, "OrthoSeg",
                "Graph Cut needs foreground seeds for the active label. "
                "Scribble inside the structure you want to segment.");
            return;
        }
        if (doc_->seedLabelCount() < 2) {
            QMessageBox::information(this, "OrthoSeg",
                "Graph Cut needs background seeds too. Scribble the Background "
                "label (or another bone) outside the target structure.");
            return;
        }
    } else if (doc_->seedLabelCount() < 2) {
        QMessageBox::information(this, "OrthoSeg",
            "Grow-from-seeds needs at least two labels competing — seed each "
            "structure plus the Background between them.");
        return;
    }

    bool ok = doc_->runSeedSegmentation(algo, activeLabel_, currentBeta());
    if (!ok)
        QMessageBox::warning(this, "OrthoSeg",
            "Segmentation could not run. Try larger, separated seed strokes; "
            "small strokes can overlap when the image is reduced for processing.");
    canvas_->update();
    updateUndoState();
}

void MainWindow::onUndo() {
    doc_->undo();
    canvas_->update();
    updateUndoState();
}

} // namespace orthoseg
