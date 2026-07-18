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

    canvas_ = new CanvasWidget(doc_.get());
    connect(canvas_, &CanvasWidget::maskChanged, this, [this] {
        updateUndoState();
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
}

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
    const char* toolNames[3] = {"Brush", "Fill", "Eraser"};
    const char* toolIcons[3] = {"🖌", "🪣", "🧽"};
    for (int i = 0; i < 3; ++i) {
        Tool tid = static_cast<Tool>(i);
        auto* btn = new QPushButton(QString("%1\n%2").arg(toolIcons[i], toolNames[i]));
        btn->setCheckable(true);
        btn->setCursor(Qt::PointingHandCursor);
        btn->setMinimumHeight(56);
        btn->setStyleSheet(QString(
            "QPushButton{ border-radius:12px; border:1px solid #1e293b;"
            " background:#1e293b4d; color:#94a3b8; font-size:11px; }"
            "QPushButton:hover{ border:1px solid #334155; }"
            "QPushButton:checked{ background:%1; border:1px solid %1;"
            " color:#0f172a; font-weight:600; }").arg(kAccent));
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
}

void MainWindow::selectTool(Tool t) {
    activeTool_ = t;
    canvas_->setActiveTool(t);
    for (int i = 0; i < 3; ++i)
        static_cast<QPushButton*>(toolButtons_[i])
            ->setChecked(i == static_cast<int>(t));
    updateSettingsVisibility();
}

void MainWindow::updateSettingsVisibility() {
    bool isFill = (activeTool_ == Tool::Fill);
    brushPanel_->setVisible(!isFill);
    fillPanel_->setVisible(isFill);
    if (!isFill || !algoCombo_) return;

    FillAlgorithm algo = static_cast<FillAlgorithm>(algoCombo_->currentIndex());
    bool scribble = isScribbleAlgorithm(algo);

    // Region-grow params only for the click-seed algorithms.
    intensityPanel_->setVisible(!scribble);
    edgePanel_->setVisible(!scribble && algo != FillAlgorithm::Standard);

    // Scribble params: β for GrowCut/RandomWalker (grabCut uses its own model).
    betaPanel_->setVisible(scribble && algo != FillAlgorithm::GraphCut);
    graphCutNote_->setVisible(algo == FillAlgorithm::GraphCut);
    seedPanel_->setVisible(scribble);
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
    canvas_->refresh();
    updateUndoState();
    updateStatus();
}

void MainWindow::onExport() {
    if (!doc_->hasImage()) {
        QMessageBox::information(this, "OrthoSeg", "Load an image first.");
        return;
    }
    QString path = QFileDialog::getSaveFileName(
        this, "Export Mask", "bone_segmentation_mask.png", "PNG (*.png)");
    if (path.isEmpty()) return;
    if (!doc_->exportMask(path.toStdString()))
        QMessageBox::warning(this, "OrthoSeg", "Failed to export mask.");
}

void MainWindow::onClear() {
    if (!doc_->hasImage()) return;
    doc_->pushHistory();
    doc_->clearMask();
    doc_->clearSeeds();
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

    doc_->pushHistory();
    bool ok = doc_->runSeedSegmentation(algo, activeLabel_, currentBeta());
    if (!ok)
        QMessageBox::warning(this, "OrthoSeg", "Segmentation could not run.");
    canvas_->update();
    updateUndoState();
}

void MainWindow::onUndo() {
    doc_->undo();
    canvas_->update();
    updateUndoState();
}

} // namespace orthoseg
