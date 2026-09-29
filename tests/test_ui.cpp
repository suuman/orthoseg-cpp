#include "CanvasWidget.h"
#include "MainWindow.h"
#include "FileBrowser.h"
#include "ImageFilePreview.h"
#include <QToolButton>
#include <QElapsedTimer>
#include <QThread>
#include <QKeyEvent>
#include <QApplication>
#include <QEvent>
#include <QFileDialog>
#include <QFileInfo>
#include <QDir>
#include <QLineEdit>
#include <QSlider>
#include <QScrollArea>
#include <QTemporaryDir>
#include <QTimer>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QComboBox>
#include <QPushButton>
#include <QCheckBox>
#include <QMessageBox>
#include <opencv2/imgcodecs.hpp>
#include <cstdio>

using namespace orthoseg;

static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { \
    std::printf("FAIL: %s\n", msg); ++failures; } \
    else std::printf("ok  : %s\n", msg); } while (0)

class PaintObserver : public QObject {
public:
    int paints = 0;
    bool eventFilter(QObject*, QEvent* event) override {
        if (event->type() == QEvent::Paint) ++paints;
        return false;
    }
};

int main(int argc, char** argv) {
    QTemporaryDir settingsDirectory;
    qputenv("XDG_CONFIG_HOME", settingsDirectory.path().toUtf8());
    QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
    QApplication app(argc, argv);
    MainWindow window;
    window.show();
    QApplication::processEvents();

    QSlider* brush = nullptr;
    for (auto* slider : window.findChildren<QSlider*>())
        if (slider->minimum() == 1 && slider->maximum() == 100) brush = slider;
    CHECK(brush != nullptr, "brush size control exists");
    if (!brush) return 1;
    window.showFillAlgorithm(3);
    QApplication::processEvents();
    CHECK(brush->isVisible(), "brush size can be adjusted while drawing seeds");
    window.showFillAlgorithm(0);
    QApplication::processEvents();
    CHECK(!brush->isVisible(), "click fills hide the unused brush size control");

    Document doc;
    CanvasWidget canvas(&doc);
    PaintObserver observer;
    canvas.installEventFilter(&observer);
    canvas.show();
    canvas.setActiveTool(Tool::Fill);
    canvas.setFillAlgorithm(FillAlgorithm::GrowCut);
    QApplication::processEvents();
    observer.paints = 0;
    canvas.setActiveTool(Tool::Brush);
    QApplication::processEvents();
    CHECK(observer.paints > 0, "changing away from seed mode repaints the overlay");
    canvas.setActiveTool(Tool::Fill);
    QApplication::processEvents();
    observer.paints = 0;
    canvas.setFillAlgorithm(FillAlgorithm::Standard);
    QApplication::processEvents();
    CHECK(observer.paints > 0, "changing fill algorithm repaints the overlay");

    // Exercise the actual upload/export slots and save dialog. A filename
    // without a suffix must become a PNG before the overwrite check and write.
    QTemporaryDir tmp;
    CHECK(tmp.isValid(), "create temporary directory for file dialogs");
    if (!tmp.isValid()) return 1;
    const QString input = tmp.filePath("source.png");
    cv::imwrite(input.toStdString(), cv::Mat(20, 20, CV_8UC3, cv::Scalar(80, 80, 80)));
    auto chooseFile = [&window](const QString& path) {
        QTimer::singleShot(0, &window, [path] {
            auto* dialog = qobject_cast<QFileDialog*>(QApplication::activeModalWidget());
            CHECK(dialog != nullptr, "file dialog opens");
            if (dialog) {
                dialog->selectFile(path);
                QMetaObject::invokeMethod(dialog, "accept", Qt::DirectConnection);
            } else if (auto* modal = QApplication::activeModalWidget()) {
                modal->close();
            }
        });
    };
    chooseFile(input);
    CHECK(QMetaObject::invokeMethod(&window, "onUpload", Qt::DirectConnection),
          "upload slot is invoked");
    chooseFile(tmp.filePath("annotation"));
    CHECK(QMetaObject::invokeMethod(&window, "onExport", Qt::DirectConnection),
          "export slot is invoked");
    CHECK(QFileInfo::exists(tmp.filePath("annotation.png")),
          "export adds the PNG suffix when the user omits an extension");
    auto* isolated = window.findChild<QCheckBox*>("isolatedLabelView");
    auto* drawFillTool = window.findChild<QPushButton*>("drawFillToolBtn");
    auto* lassoTool = window.findChild<QPushButton*>("lassoToolBtn");
    auto* importButton = window.findChild<QPushButton*>("importMaskButton");
    CHECK(isolated && drawFillTool && lassoTool && importButton,
          "isolated view, Draw & Fill, Lasso, and Import Mask controls are available");
    auto* sidebar = window.findChild<QScrollArea*>();
    CHECK(sidebar && drawFillTool->mapTo(sidebar->viewport(),
              QPoint(drawFillTool->width(), 0)).x() <= sidebar->viewport()->width() &&
          lassoTool->mapTo(sidebar->viewport(),
              QPoint(lassoTool->width(), 0)).x() <= sidebar->viewport()->width(),
          "new tools fit inside the sidebar viewport");
    const QString overlapPath = tmp.filePath("ui-overlap.png");
    cv::Mat overlapFile(20, 20, CV_8UC3, cv::Scalar(0, 0, 0));
    overlapFile.at<cv::Vec3b>(10, 10) = cv::Vec3b(3, 2, 1);
    cv::imwrite(overlapPath.toStdString(), overlapFile);
    chooseFile(overlapPath);
    CHECK(QMetaObject::invokeMethod(&window, "onImportMask", Qt::DirectConnection) &&
          window.document()->maskChannels().at<cv::Vec3b>(10, 10) == cv::Vec3b(3, 2, 1),
          "Import Mask action restores overlapping RGB label channels");
    isolated->click();
    CHECK(window.canvas()->isolatedView(), "isolated view checkbox changes canvas mode");
    isolated->click();
    drawFillTool->click();
    CHECK(window.findChild<QPushButton*>("drawOutlineButton")->isVisible() &&
          window.findChild<QPushButton*>("fillClosedAreaButton")->isVisible() &&
          window.findChild<QPushButton*>("fillOutlineButton")->isVisible(),
          "Draw & Fill exposes outline and enclosed-hole controls");
    lassoTool->click();
    CHECK(window.findChild<QPushButton*>("fillOutlineButton")->isVisible(),
          "Lasso exposes a fill-outline action");

    auto* edgeToggle = window.findChild<QCheckBox*>("constrainToEdges");
    auto* edgeSlider = window.findChild<QSlider*>("edgeThresholdSlider");
    CHECK(edgeToggle && edgeToggle->isVisible() && edgeSlider && !edgeSlider->isVisible(),
          "lasso edge stopping is optional and initially off");
    edgeToggle->click();
    CHECK(edgeSlider->isVisible(), "edge threshold appears when edge stopping is enabled");
    edgeToggle->click();
    auto* toolbarBatchButton = window.findChild<QPushButton*>("batchModeButton");
    auto* managementButton = window.findChild<QPushButton*>("modelManagementButton");
    CHECK(toolbarBatchButton && managementButton && toolbarBatchButton->parentWidget() == managementButton->parentWidget(),
          "model management lives in top toolbar alongside batch mode");
    CHECK(managementButton->isEnabled() || managementButton->isHidden(),
          "management button is hidden when access is unavailable");
    auto* panButton = window.findChild<QPushButton*>("panImageButton");
    CHECK(panButton && panButton->isCheckable(), "drag pan control remains available");
    CHECK(!window.findChild<QWidget*>("canvasNavigation"), "image movement and centering row removed");

    // Source-coordinate box gestures must follow the same transform as rendering.
    CHECK(doc.loadImage(input.toStdString()), "load coordinate test image");
    canvas.refresh();
    canvas.resize(640, 480);
    canvas.setActiveTool(Tool::AIFill);
    auto mouse = [&canvas](QEvent::Type type, QPointF position, Qt::MouseButton button,
                           Qt::MouseButtons buttons,
                           Qt::KeyboardModifiers modifiers = Qt::NoModifier) {
        QMouseEvent event(type, position, position, button, buttons, modifiers);
        QApplication::sendEvent(&canvas, &event);
    };
    auto wheel = [&canvas](QPointF position, int angle) {
        QWheelEvent event(position, canvas.mapToGlobal(position.toPoint()),
                          QPoint(), QPoint(0, angle), Qt::NoButton,
                          Qt::NoModifier, Qt::NoScrollPhase, false);
        QApplication::sendEvent(&canvas, &event);
    };
    auto sourcePoint = [&canvas](double x, double y) {
        const QRectF rect = canvas.imageRect();
        return QPointF(rect.left() + (x + 0.5) * rect.width() / 20,
                       rect.top() + (y + 0.5) * rect.height() / 20);
    };
    for (int i = 0; i < 4; ++i) {
        if (i) canvas.zoomIn();
        mouse(QEvent::MouseButtonPress, {200, 200}, Qt::MiddleButton, Qt::MiddleButton);
        mouse(QEvent::MouseMove, {213, 193}, Qt::NoButton, Qt::MiddleButton);
        mouse(QEvent::MouseButtonRelease, {213, 193}, Qt::MiddleButton, Qt::NoButton);
        CHECK(canvas.widgetToImage(sourcePoint(6, 8)) == QPoint(6, 8),
              "viewer-to-source mapping follows zoom and middle-button pan");
        mouse(QEvent::MouseButtonPress, sourcePoint(4, 5), Qt::LeftButton, Qt::LeftButton);
        mouse(QEvent::MouseMove, sourcePoint(13, 16), Qt::NoButton, Qt::LeftButton);
        mouse(QEvent::MouseButtonRelease, sourcePoint(13, 16), Qt::LeftButton, Qt::NoButton);
        CHECK(doc.aiFill().box && doc.aiFill().box->x0 == 4 && doc.aiFill().box->y0 == 5 &&
              doc.aiFill().box->x1 == 13 && doc.aiFill().box->y1 == 16,
              "box gesture stores original pixel coordinates");
        CHECK(doc.aiFill().femurBox && doc.aiFill().femurBox->x0 == 4,
              "femur box is stored when Femur is active");
    }
    // Test dual bounding box: draw Tibia box while keeping Femur box
    canvas.setActiveLabel(Label::Tibia);
    mouse(QEvent::MouseButtonPress, sourcePoint(2, 3), Qt::LeftButton, Qt::LeftButton);
    mouse(QEvent::MouseMove, sourcePoint(8, 9), Qt::NoButton, Qt::LeftButton);
    mouse(QEvent::MouseButtonRelease, sourcePoint(8, 9), Qt::LeftButton, Qt::NoButton);
    CHECK(doc.aiFill().tibiaBox && doc.aiFill().tibiaBox->x0 == 2 &&
          doc.aiFill().femurBox && doc.aiFill().femurBox->x0 == 4,
          "both femur and tibia boxes persist concurrently");
    canvas.setActiveLabel(Label::Femur);
    doc.aiFill().showSecondaryBoxes = true;
    doc.aiFill().activeBoxNumber = 2;
    mouse(QEvent::MouseButtonPress, sourcePoint(10, 11), Qt::LeftButton, Qt::LeftButton);
    mouse(QEvent::MouseMove, sourcePoint(17, 18), Qt::NoButton, Qt::LeftButton);
    mouse(QEvent::MouseButtonRelease, sourcePoint(17, 18), Qt::LeftButton, Qt::NoButton);
    CHECK(doc.aiFill().femurBox2 && doc.aiFill().femurBox2->x0 == 10 &&
          doc.aiFill().femurBox && doc.aiFill().femurBox->x0 == 4,
          "next bounding box preserves first Femur box for bilateral cases");
    doc.aiFill().showSecondaryBoxes = false;
    doc.aiFill().activeBoxNumber = 1;

    CHECK(cv::countNonZero(doc.mask()) == 0, "AI box gestures preserve editable annotations");
    canvas.zoomReset();
    CHECK(canvas.widgetToImage(sourcePoint(1, 2)) == QPoint(1, 2), "reset restores fit mapping");

    // A point near either vertical end should stay under the pointer while
    // zooming, then be movable to the viewport center without changing tools.
    const QPointF center(canvas.width() / 2.0, canvas.height() / 2.0);
    const QPointF lower = sourcePoint(10, 17);
    wheel(lower, 120);
    CHECK(canvas.zoom() > 1.0f && canvas.widgetToImage(lower) == QPoint(10, 17),
          "wheel zoom keeps the lower image region under the pointer");
    mouse(QEvent::MouseButtonPress, lower, Qt::RightButton, Qt::RightButton);
    mouse(QEvent::MouseMove, center, Qt::NoButton, Qt::RightButton);
    mouse(QEvent::MouseButtonRelease, center, Qt::RightButton, Qt::NoButton);
    CHECK(canvas.widgetToImage(center) == QPoint(10, 17),
          "right-drag brings the lower image region to the viewport center");
    wheel(center, 120);
    CHECK(canvas.widgetToImage(center) == QPoint(10, 17),
          "zooming in after panning keeps the centered region fixed");
    wheel(center, -120);
    CHECK(canvas.widgetToImage(center) == QPoint(10, 17),
          "zooming out after panning keeps the centered region fixed");

    canvas.zoomReset();
    const QPointF upper = sourcePoint(10, 2);
    wheel(upper, 120);
    CHECK(canvas.widgetToImage(upper) == QPoint(10, 2),
          "wheel zoom keeps the upper image region under the pointer");
    const Box boxBeforePan = *doc.aiFill().box;
    mouse(QEvent::MouseButtonPress, upper, Qt::LeftButton, Qt::LeftButton,
          Qt::ControlModifier);
    mouse(QEvent::MouseMove, center, Qt::NoButton, Qt::LeftButton,
          Qt::ControlModifier);
    mouse(QEvent::MouseButtonRelease, center, Qt::LeftButton, Qt::NoButton,
          Qt::ControlModifier);
    CHECK(canvas.widgetToImage(center) == QPoint(10, 2),
          "Ctrl+left-drag brings the upper image region to the viewport center");
    CHECK(doc.aiFill().box->x0 == boxBeforePan.x0 &&
          doc.aiFill().box->y0 == boxBeforePan.y0,
          "temporary pan does not edit the AI bounding box");
    canvas.zoomReset();

    doc.aiFill().promptType = AIFillPromptType::PaintedMask;
    canvas.setBrushSize(2);
    mouse(QEvent::MouseButtonPress, sourcePoint(5, 6), Qt::LeftButton, Qt::LeftButton);
    mouse(QEvent::MouseMove, sourcePoint(10, 6), Qt::NoButton, Qt::LeftButton);
    mouse(QEvent::MouseButtonRelease, sourcePoint(10, 6), Qt::LeftButton, Qt::NoButton);
    CHECK(doc.aiFill().promptMask.size() == doc.sourceColor().size() &&
          doc.aiFill().promptMask.at<uchar>(6, 8) == 255,
          "paint prompt is binary and stored at source resolution");
    canvas.setActiveLabel(Label::Femur);
    canvas.update();
    QApplication::processEvents();
    const auto femurImg = canvas.grab().toImage();
    QPoint ptSample = sourcePoint(8, 6).toPoint();
    QRgb femurPix = femurImg.pixel(ptSample);
    CHECK(qRed(femurPix) > qGreen(femurPix) && qRed(femurPix) > qBlue(femurPix),
          "Femur painted prompt renders with normal fill Red color");
    canvas.setActiveLabel(Label::Tibia);
    canvas.update();
    QApplication::processEvents();
    const auto tibiaImg = canvas.grab().toImage();
    QRgb tibiaPix = tibiaImg.pixel(ptSample);
    CHECK(qGreen(tibiaPix) > qRed(tibiaPix) && qGreen(tibiaPix) > qBlue(tibiaPix),
          "Tibia painted prompt renders with normal fill Green color");
    canvas.setActiveLabel(Label::Femur);
    CHECK(cv::countNonZero(doc.mask()) == 0 && !doc.hasSeeds(),
          "prompt painting preserves annotation and seed layers");
    canvas.setAIPromptErase(true);
    mouse(QEvent::MouseButtonPress, sourcePoint(8, 6), Qt::LeftButton, Qt::LeftButton);
    mouse(QEvent::MouseButtonRelease, sourcePoint(8, 6), Qt::LeftButton, Qt::NoButton);
    CHECK(doc.aiFill().promptMask.at<uchar>(6, 8) == 0, "prompt eraser removes foreground");

    // Synthetic preview tests exercise rendering and editing, not inference.
    doc.aiFill().resultMask = cv::Mat::zeros(20, 20, CV_8UC1);
    doc.aiFill().resultMask.at<uchar>(12, 12) = 1;
    canvas.setActiveTool(Tool::Brush); // Hide prompt overlay for this comparison.
    const auto visible = canvas.grab().toImage();
    doc.aiFill().showResult = false;
    const auto hidden = canvas.grab().toImage();
    CHECK(visible != hidden, "AI result show/hide changes the rendered overlay");
    const auto sourceBefore = doc.sourceColor().clone();
    doc.applyAIResult();
    CHECK(doc.mask().at<uchar>(12, 12) == 1 && doc.aiFill().resultMask.empty(),
          "apply transfers preview to the existing editable mask");
    doc.undo();
    CHECK(doc.mask().at<uchar>(12, 12) == 0 &&
          cv::norm(sourceBefore, doc.sourceColor(), cv::NORM_INF) == 0,
          "undo restores annotations and source pixels remain unchanged");

    canvas.setActiveTool(Tool::Brush);
    canvas.setActiveLabel(Label::Femur);
    mouse(QEvent::MouseButtonPress, sourcePoint(10, 10), Qt::LeftButton, Qt::LeftButton);
    mouse(QEvent::MouseButtonRelease, sourcePoint(10, 10), Qt::LeftButton, Qt::NoButton);
    canvas.setActiveLabel(Label::Tibia);
    mouse(QEvent::MouseButtonPress, sourcePoint(10, 10), Qt::LeftButton, Qt::LeftButton);
    mouse(QEvent::MouseButtonRelease, sourcePoint(10, 10), Qt::LeftButton, Qt::NoButton);
    CHECK(doc.maskChannels().at<cv::Vec3b>(10, 10) == cv::Vec3b(0, 2, 1),
          "brush gestures add independent overlapping bone channels");
    const auto mixedView = canvas.grab().toImage();
    canvas.setIsolatedView(true);
    const auto tibiaOnly = canvas.grab().toImage();
    canvas.setActiveLabel(Label::Femur);
    const auto femurOnly = canvas.grab().toImage();
    CHECK(mixedView != tibiaOnly && tibiaOnly != femurOnly &&
          qRed(mixedView.pixel(sourcePoint(10, 10).toPoint())) >
          qRed(tibiaOnly.pixel(sourcePoint(10, 10).toPoint())) &&
          qGreen(mixedView.pixel(sourcePoint(10, 10).toPoint())) >
          qGreen(femurOnly.pixel(sourcePoint(10, 10).toPoint())),
          "overlap renders mixed color and isolated view renders only the active label");
    canvas.setIsolatedView(false);
    canvas.setActiveLabel(Label::Tibia);
    canvas.setActiveTool(Tool::Eraser);
    mouse(QEvent::MouseButtonPress, sourcePoint(10, 10), Qt::LeftButton, Qt::LeftButton);
    mouse(QEvent::MouseButtonRelease, sourcePoint(10, 10), Qt::LeftButton, Qt::NoButton);
    CHECK(doc.maskChannels().at<cv::Vec3b>(10, 10) == cv::Vec3b(0, 0, 1),
          "eraser gesture removes only the active label from an overlap");
    canvas.setActiveTool(Tool::Brush);
    doc.aiFill().showResult = true;
    doc.aiFill().resultMask = cv::Mat::zeros(20, 20, CV_8UC1);
    doc.aiFill().resultMask.at<uchar>(10, 10) = 2;
    const auto additivePreview = canvas.grab().toImage();
    doc.applyAIResult();
    CHECK(canvas.grab().toImage() == additivePreview &&
          doc.maskChannels().at<cv::Vec3b>(10, 10) == cv::Vec3b(0, 2, 1),
          "AI foreground preview shows the same mixed overlap color as Apply");
    canvas.setActiveTool(Tool::DrawFill);
    canvas.setActiveLabel(Label::Femur);
    mouse(QEvent::MouseButtonPress, sourcePoint(3, 3), Qt::LeftButton, Qt::LeftButton);
    for (auto point : {QPoint(8, 3), QPoint(8, 8), QPoint(3, 8), QPoint(3, 3)})
        mouse(QEvent::MouseMove, sourcePoint(point.x(), point.y()), Qt::NoButton, Qt::LeftButton);
    mouse(QEvent::MouseButtonRelease, sourcePoint(3, 3), Qt::LeftButton, Qt::NoButton);
    canvas.fillCurrentOutline();
    CHECK(doc.labelMask(Label::Femur).at<uchar>(5, 5) == 255,
          "Draw & Fill traces and fills a closed outline");
    canvas.setActiveTool(Tool::Lasso);
    canvas.setActiveLabel(Label::Fibula);
    canvas.setAutoFillOutline(true);
    mouse(QEvent::MouseButtonPress, sourcePoint(5, 5), Qt::LeftButton, Qt::LeftButton);
    for (auto point : {QPoint(9, 5), QPoint(9, 9), QPoint(5, 9)})
        mouse(QEvent::MouseMove, sourcePoint(point.x(), point.y()), Qt::NoButton, Qt::LeftButton);
    mouse(QEvent::MouseButtonRelease, sourcePoint(5, 5), Qt::LeftButton, Qt::NoButton);
    CHECK(doc.maskChannels().at<cv::Vec3b>(6, 6) == cv::Vec3b(3, 0, 1),
          "Lasso auto-fills a polygon while retaining an overlapping Femur label");

    // Exercise real mask file dialogs, including a failed import that preserves the prompt.
    auto* combo = window.findChild<QComboBox*>("aiPromptType");
    CHECK(combo && combo->count() == 4, "all four AI prompt modes are exposed");
    for (auto* button : window.findChildren<QPushButton*>())
        if (button->text().endsWith("AI Fill") && button->isCheckable()) button->click();
    const auto promptPath = tmp.filePath("prompt.tiff");
    cv::Mat fileMask = cv::Mat::zeros(20, 20, CV_16UC1);
    fileMask.at<ushort>(4, 6) = 1;
    cv::imwrite(promptPath.toStdString(), fileMask);
    chooseFile(promptPath);
    CHECK(QMetaObject::invokeMethod(&window, "onLoadPromptMask", Qt::DirectConnection),
          "load prompt mask slot is invoked");
    CHECK(combo->currentIndex() == 2, "loading a prompt selects Load Mask mode");
    QApplication::processEvents(); // Settle the scroll panel after changing prompt controls.
    auto* windowCanvas = window.findChild<CanvasWidget*>();
    const auto loadedPreview = windowCanvas->grab().toImage();
    if (!qEnvironmentVariableIsEmpty("ORTHOSEG_UI_CAPTURE"))
        window.grab().save(qEnvironmentVariable("ORTHOSEG_UI_CAPTURE"));
    cv::imwrite(tmp.filePath("wrong.png").toStdString(), cv::Mat(5, 5, CV_8UC1, cv::Scalar(255)));
    bool mismatchWarning = false;
    QTimer warningCloser;
    warningCloser.setInterval(5);
    QObject::connect(&warningCloser, &QTimer::timeout, &window, [&] {
        if (auto* warning = qobject_cast<QMessageBox*>(QApplication::activeModalWidget())) {
            mismatchWarning = warning->text().contains("dimensions must match");
            warning->accept();
        }
    });
    warningCloser.start();
    chooseFile(tmp.filePath("wrong.png"));
    QMetaObject::invokeMethod(&window, "onLoadPromptMask", Qt::DirectConnection);
    warningCloser.stop();
    CHECK(mismatchWarning && windowCanvas->grab().toImage() == loadedPreview,
          "mismatched loaded mask reports an error and preserves the existing prompt");
    for (auto* button : window.findChildren<QPushButton*>())
        if (button->text() == "Clear Prompt") button->click();
    CHECK(windowCanvas->grab().toImage() != loadedPreview,
          "loaded 16-bit prompt is visible and Clear Prompt removes it");

    // 1. Top bar Model Directory button exists
    auto* modelBtn = window.findChild<QPushButton*>("modelDirBtn");
    CHECK(modelBtn != nullptr, "MedSAM2 Models button exists in top bar");

    // 2. Opacity slider controls prompt mask rendering opacity
    canvas.setActiveTool(Tool::AIFill);
    doc.aiFill().promptType = AIFillPromptType::PaintedMask;
    doc.aiFill().showPrompt = true;
    doc.aiFill().promptMask = cv::Mat::zeros(doc.sourceColor().size(), CV_8UC1);
    doc.aiFill().promptMask.at<uchar>(6, 8) = 255;
    canvas.setMaskOpacity(0.2f);
    canvas.update();
    QApplication::processEvents();
    auto imgLowAlpha = canvas.grab().toImage();
    canvas.setMaskOpacity(0.9f);
    canvas.update();
    QApplication::processEvents();
    auto imgHighAlpha = canvas.grab().toImage();
    CHECK(imgLowAlpha.pixel(sourcePoint(8, 6).toPoint()) !=
          imgHighAlpha.pixel(sourcePoint(8, 6).toPoint()),
          "paint mask rendering respects mask opacity slider");

    // 3. Prompt invisibility after segmentation
    doc.aiFill().showPrompt = false;
    canvas.update();
    QApplication::processEvents();
    auto imgHidden = canvas.grab().toImage();
    doc.aiFill().showPrompt = true;
    canvas.update();
    QApplication::processEvents();
    auto imgVisible = canvas.grab().toImage();
    CHECK(imgHidden != imgVisible, "showPrompt false hides the prompt overlay");

    // 4. Use AI result as next prompt
    window.document()->aiFill().resultMask = cv::Mat::zeros(window.document()->sourceColor().size(), CV_8UC1);
    window.document()->aiFill().resultMask.at<uchar>(5, 5) = 1; // Femur
    window.document()->aiFill().resultMask.at<uchar>(10, 10) = 2; // Tibia
    QPushButton* useResultBtn = nullptr;
    for (auto* button : window.findChildren<QPushButton*>()) {
        if (button->text() == "Use AI Result as Next Prompt") {
            useResultBtn = button;
            break;
        }
    }
    CHECK(useResultBtn != nullptr, "Use AI Result as Next Prompt button exists");
    if (useResultBtn) {
        useResultBtn->click();
        QApplication::processEvents();
        CHECK(cv::countNonZero(window.document()->aiFill().promptMask) == 1,
              "Use AI Result selects the active bone without merging the other channel");
        CHECK(combo->currentIndex() == static_cast<int>(AIFillPromptType::PaintedMask),
              "Use AI Result button switches prompt type to PaintedMask");
        CHECK(window.document()->aiFill().showPrompt, "Use AI Result enables showPrompt");
    }

    // 5. Normal fill mask automatically converted to paint prompt format
    window.document()->paintLine(cv::Point(2, 2), cv::Point(4, 4), Label::Femur, 2);
    CHECK(cv::countNonZero(window.document()->mask()) > 0, "normal fill mask has annotations");
    combo->setCurrentIndex(static_cast<int>(AIFillPromptType::NormalFillMask));
    QApplication::processEvents();
    CHECK(combo->currentIndex() == static_cast<int>(AIFillPromptType::PaintedMask),
          "selecting Normal Fill Mask automatically converts to paint prompt format");
    CHECK(cv::countNonZero(window.document()->aiFill().promptMask) > 0,
          "prompt mask received normal mask annotations");

    // 6. Configure MedSAM2 Models button removed from left pane
    QPushButton* leftPaneModelBtn = nullptr;
    for (auto* btn : window.findChildren<QPushButton*>()) {
        if (btn->text().contains("Configure MedSAM2")) leftPaneModelBtn = btn;
    }
    CHECK(leftPaneModelBtn == nullptr, "Configure MedSAM2 Models button removed from left pane");

    // 7. CLAHE top bar controls and parameters sub-window
    auto* claheCb = window.findChild<QCheckBox*>("claheCheck");
    CHECK(claheCb != nullptr, "CLAHE checkbox exists on top bar");
    auto* claheSettings = window.findChild<QPushButton*>("claheSettingsBtn");
    CHECK(claheSettings != nullptr, "CLAHE settings button exists on top bar");

    // Test CLAHE display enhancement vs raw source image (processing uses original)
    const QString claheTestPath = tmp.filePath("clahe_test.png");
    cv::Mat testMat(20, 20, CV_8UC3);
    for (int y = 0; y < 20; ++y) {
        for (int x = 0; x < 20; ++x) {
            uchar val = static_cast<uchar>((x * 12 + y * 8) % 256);
            testMat.at<cv::Vec3b>(y, x) = cv::Vec3b(val, val, val);
        }
    }
    cv::imwrite(claheTestPath.toStdString(), testMat);
    Document claheDoc;
    CHECK(claheDoc.loadImage(claheTestPath.toStdString()), "load test image for CLAHE");
    CanvasWidget claheCanvas(&claheDoc);
    claheCanvas.refresh();
    const QImage rawDisplay = claheCanvas.displayImage();
    const cv::Mat rawSourceColor = claheDoc.sourceColor().clone();

    claheCanvas.setClaheEnabled(true);
    const QImage claheDisplay = claheCanvas.displayImage();
    CHECK(rawDisplay != claheDisplay, "CLAHE modifies displayed image in UI for better visibility");
    CHECK(cv::norm(rawSourceColor, claheDoc.sourceColor(), cv::NORM_INF) == 0,
          "source image in Document is strictly unchanged for processing");

    claheCanvas.setClaheEnabled(false);
    CHECK(claheCanvas.displayImage() == rawDisplay,
          "disabling CLAHE restores original image display");

    // Test clicking CLAHE checkbox opens parameters dialog and sets enabled
    claheCb->click(); // Toggles to checked, opening dialog
    QApplication::processEvents();
    auto* claheDlg = window.claheDialog();
    if (!claheDlg) claheDlg = window.findChild<QDialog*>("claheDialog");
    CHECK(claheDlg != nullptr && claheDlg->isVisible(), "clicking CLAHE checkbox opens parameters dialog");
    CHECK(claheDlg && !claheDlg->isModal(), "CLAHE dialog is modeless so canvas can be viewed and interacted with");
    if (claheDlg) {
        // Verify CLAHE dialog can be dragged or moved to side
        const QPoint origPos = claheDlg->pos();
        const QPoint targetSidePos = origPos + QPoint(80, 40);
        claheDlg->move(targetSidePos);
        CHECK(claheDlg->pos() == targetSidePos, "CLAHE sub-window can be dragged or moved to side");
        if (auto* reset = claheDlg->findChild<QPushButton*>("claheResetBtn")) reset->click();
    }
    CHECK(window.canvas()->claheEnabled(), "CLAHE is enabled after checking top bar checkbox");
    CHECK(window.canvas()->claheClipLimit() == 2.0 && window.canvas()->claheGridSize() == 8,
          "reset defaults in CLAHE dialog set default clipLimit and gridSize");

    claheCb->click(); // Unchecking disables CLAHE
    QApplication::processEvents();
    CHECK(!window.canvas()->claheEnabled(), "unchecking CLAHE disables display enhancement");
    CHECK(claheDlg && !claheDlg->isVisible(), "unchecking CLAHE hides the parameters dialog");

    // 8. AI Fill box is styled in prominent green
    auto* aiToolBtn = window.findChild<QPushButton*>("aiFillToolBtn");
    CHECK(aiToolBtn != nullptr, "AI Fill tool button exists in toolbox");
    CHECK(aiToolBtn && aiToolBtn->styleSheet().contains("#22c55e") && aiToolBtn->styleSheet().contains("#14532d"),
          "AI Fill toolbox button is filled with prominent green");
    auto* runAIFillBtn = window.findChild<QPushButton*>("runAIFill");
    CHECK(runAIFillBtn != nullptr, "Run AI Fill button exists");
    CHECK(runAIFillBtn && runAIFillBtn->styleSheet().contains("#22c55e"),
          "Run AI Fill action button is filled in prominent green");

    // 9. When AI mask is edited, use that mask for next prompt
    window.document()->aiFill().resultMask = cv::Mat::zeros(window.document()->sourceColor().size(), CV_8UC1);
    window.document()->aiFill().resultMask.at<uchar>(8, 8) = 1; // Femur result
    window.canvas()->setActiveTool(Tool::Brush);
    window.document()->paintLine(cv::Point(8, 8), cv::Point(10, 10), Label::Femur, 2);
    emit window.canvas()->maskChanged();
    QApplication::processEvents();
    CHECK(!window.document()->aiFill().promptMask.empty() &&
          cv::countNonZero(window.document()->aiFill().promptMask) > 0,
          "editing AI mask automatically updates promptMask for next AI prompt");
    CHECK(window.document()->aiFill().promptType == AIFillPromptType::PaintedMask,
          "editing AI mask switches prompt mode to PaintedMask for next pass");

    // Batch mode pairs nnUNet-named images with editable label masks and skips
    // any case already saved at the destination.
    QTimer batchSaveOnly;
    QObject::connect(&batchSaveOnly, &QTimer::timeout, &window, [] {
        auto* box = qobject_cast<QMessageBox*>(QApplication::activeModalWidget());
        if (box && box->windowTitle() == "Add to AI Training")
            for (auto* button : box->buttons()) if (button->text() == "Save Only") button->click();
    });
    batchSaveOnly.start(5);
    QTemporaryDir batchTemp;
    CHECK(batchTemp.isValid(), "create batch mode folders");
    QDir batchRoot(batchTemp.path());
    batchRoot.mkpath("images");
    batchRoot.mkpath("labels");
    batchRoot.mkpath("saved");
    const QString imageDir = batchRoot.filePath("images");
    const QString labelDir = batchRoot.filePath("labels");
    const QString savedDir = batchRoot.filePath("saved");
    for (const QString& stem : {"case_a", "case_b", "case_c"}) {
        cv::Mat xray(16, 16, CV_8UC3, cv::Scalar(80, 80, 80));
        cv::imwrite(QDir(imageDir).filePath(stem + "_0000.png").toStdString(), xray);
        cv::Mat label(16, 16, CV_8UC3, cv::Scalar(0, 0, 0));
        label.at<cv::Vec3b>(4, 4) = stem == "case_b" ? cv::Vec3b(0, 2, 1) : cv::Vec3b(0, 0, 1);
        cv::imwrite(QDir(labelDir).filePath(stem + ".png").toStdString(), label);
    }
    cv::imwrite(QDir(labelDir).filePath("unpaired.png").toStdString(),
                cv::Mat(16, 16, CV_8UC1, cv::Scalar(0)));
    const QString savedA = QDir(savedDir).filePath("case_a.png");
    cv::imwrite(savedA.toStdString(), cv::Mat(16, 16, CV_8UC3, cv::Scalar(3, 0, 0)));
    const auto savedABefore = cv::imread(savedA.toStdString(), cv::IMREAD_UNCHANGED);
    MainWindow batchWindow;
    batchWindow.show();
    QApplication::processEvents();
    auto* batchButton = batchWindow.findChild<QPushButton*>("batchModeButton");
    CHECK(batchButton && batchButton->isVisible(), "Batch Mode button is visible in the top bar");
    auto startBatch = [&] {
        bool configured = false;
        QTimer::singleShot(0, &batchWindow, [&] {
            auto* dlg = qobject_cast<QDialog*>(QApplication::activeModalWidget());
            auto* images = dlg ? dlg->findChild<QLineEdit*>("batchImagesDir") : nullptr;
            auto* labels = dlg ? dlg->findChild<QLineEdit*>("batchLabelsDir") : nullptr;
            auto* saved = dlg ? dlg->findChild<QLineEdit*>("batchOutputDir") : nullptr;
            auto* start = dlg ? dlg->findChild<QPushButton*>("batchStartButton") : nullptr;
            configured = images && labels && saved && start;
            if (!configured) { if (dlg) dlg->reject(); return; }
            images->setText(imageDir);
            labels->setText(labelDir);
            saved->setText(savedDir);
            start->click();
        });
        batchButton->click();
        CHECK(configured, "batch setup opens with three folder inputs");
    };
    startBatch();
    if (!batchWindow.document()->hasImage()) return 1;
    CHECK(batchButton->text() == "Batch 1/2" &&
          QString::fromStdString(batchWindow.document()->sourcePath()).endsWith("case_b_0000.png") &&
          batchWindow.document()->maskChannels().at<cv::Vec3b>(4, 4) == cv::Vec3b(0, 2, 1) &&
          !batchWindow.document()->canUndo(),
          "first unsaved pair loads automatically with its exact overlap mask");
    const QString savedB = QDir(savedDir).filePath("case_b.png");
    batchWindow.document()->aiFill().resultMask = cv::Mat::zeros(16, 16, CV_8UC1);
    batchWindow.document()->aiFill().resultMask.at<uchar>(4, 4) = 1;
    QTimer::singleShot(0, &batchWindow, [] {
        if (auto* msg = qobject_cast<QMessageBox*>(QApplication::activeModalWidget())) msg->accept();
    });
    QMetaObject::invokeMethod(&batchWindow, "onExport", Qt::DirectConnection);
    CHECK(!QFileInfo::exists(savedB) && batchButton->text() == "Batch 1/2",
          "batch save waits until an AI preview is applied or cleared");
    batchWindow.document()->aiFill().resultMask.release();
    batchWindow.document()->paintLine({4, 4}, {4, 4}, Label::Fibula, 1);
    QMetaObject::invokeMethod(&batchWindow, "onExport", Qt::DirectConnection);
    const auto savedBMask = cv::imread(savedB.toStdString(), cv::IMREAD_UNCHANGED);
    CHECK(savedBMask.type() == CV_8UC3 && savedBMask.at<cv::Vec3b>(4, 4) == cv::Vec3b(3, 2, 1) &&
          batchButton->text() == "Batch 2/2" &&
          QString::fromStdString(batchWindow.document()->sourcePath()).endsWith("case_c_0000.png"),
          "Save Mask & Next writes exact channels as case_b.png and loads case_c");
    const QString manualImage = batchRoot.filePath("manual.png");
    cv::imwrite(manualImage.toStdString(), cv::Mat(16, 16, CV_8UC3, cv::Scalar(40, 40, 40)));
    QTimer::singleShot(0, &batchWindow, [manualImage] {
        if (auto* dlg = qobject_cast<QFileDialog*>(QApplication::activeModalWidget())) {
            dlg->selectFile(manualImage);
            QMetaObject::invokeMethod(dlg, "accept", Qt::DirectConnection);
        }
    });
    QMetaObject::invokeMethod(&batchWindow, "onUpload", Qt::DirectConnection);
    CHECK(batchButton->text() == "Batch Mode" &&
          QString::fromStdString(batchWindow.document()->sourcePath()) == manualImage &&
          !QFileInfo::exists(QDir(savedDir).filePath("case_c.png")),
          "manual X-ray import ends batch mode without saving the unfinished case");
    startBatch();
    if (!batchWindow.document()->hasImage()) return 1;
    CHECK(batchButton->text() == "Batch 1/1" &&
          QString::fromStdString(batchWindow.document()->sourcePath()).endsWith("case_c_0000.png"),
          "resuming rescans the destination and loads only the remaining pair");
    QMetaObject::invokeMethod(&batchWindow, "onExport", Qt::DirectConnection);
    CHECK(QFileInfo::exists(QDir(savedDir).filePath("case_c.png")) &&
          batchButton->text() == "Batch Mode" &&
          cv::norm(savedABefore, cv::imread(savedA.toStdString(), cv::IMREAD_UNCHANGED), cv::NORM_INF) == 0,
          "last save completes the batch and leaves previously saved masks untouched");

    // Real mouse gestures exercise all four edge-aware tool paths.
    const QString edgeInput = tmp.filePath("edge-tools.png");
    cv::Mat edgeSource(20, 20, CV_8UC3, cv::Scalar(30, 30, 30));
    edgeSource(cv::Rect(3, 3, 10, 14)).setTo(cv::Scalar(210, 210, 210));
    cv::imwrite(edgeInput.toStdString(), edgeSource);
    CHECK(doc.loadImage(edgeInput.toStdString()), "load image for edge tool gestures");
    canvas.refresh();
    canvas.zoomReset();
    canvas.setActiveLabel(Label::Femur);
    canvas.setBrushSize(1);
    canvas.setEdgePenaltyThreshold(30);
    canvas.setAutoFillOutline(true);
    for (const auto tool : {Tool::Brush, Tool::Eraser, Tool::DrawFill, Tool::Lasso}) {
        for (const bool constrained : {false, true}) {
            doc.clearMask();
            canvas.setActiveTool(tool);
            canvas.setEdgeConstrained(constrained);
            if (tool == Tool::Eraser)
                doc.paintLine({6, 7}, {17, 7}, Label::Femur, 1);
            mouse(QEvent::MouseButtonPress, sourcePoint(6, 7), Qt::LeftButton, Qt::LeftButton);
            mouse(QEvent::MouseMove, sourcePoint(17, 7), Qt::NoButton, Qt::LeftButton);
            if (tool == Tool::DrawFill || tool == Tool::Lasso) {
                mouse(QEvent::MouseMove, sourcePoint(17, 12), Qt::NoButton, Qt::LeftButton);
                mouse(QEvent::MouseMove, sourcePoint(6, 12), Qt::NoButton, Qt::LeftButton);
            }
            mouse(QEvent::MouseButtonRelease, sourcePoint(6, 7), Qt::LeftButton, Qt::NoButton);
            const bool outsidePainted = doc.mask().at<uchar>(7, 17) != 0;
            CHECK(outsidePainted == (tool == Tool::Eraser ? constrained : tool == Tool::DrawFill || !constrained),
                  "tool gesture respects edge opt-in and freehand opt-out");
        }
    }
    canvas.setPanMode(true);
    const auto beforePan = canvas.imageRect();
    const auto beforeMask = doc.maskChannels().clone();
    mouse(QEvent::MouseButtonPress, {200, 200}, Qt::LeftButton, Qt::LeftButton);
    mouse(QEvent::MouseMove, {220, 140}, Qt::NoButton, Qt::LeftButton);
    mouse(QEvent::MouseButtonRelease, {220, 140}, Qt::LeftButton, Qt::NoButton);
    CHECK(std::abs(canvas.imageRect().top() - beforePan.top() + 60) < 0.01 &&
          cv::norm(beforeMask, doc.maskChannels(), cv::NORM_INF) == 0,
          "pan mode left drag moves image vertically without painting");
    CHECK(canvas.cursor().shape() == Qt::OpenHandCursor, "drag remains active after release");
    const auto secondPan = canvas.imageRect();
    const auto secondStart = sourcePoint(6, 7);
    mouse(QEvent::MouseButtonPress, secondStart, Qt::LeftButton, Qt::LeftButton);
    mouse(QEvent::MouseMove, secondStart + QPointF(15, 25), Qt::NoButton, Qt::LeftButton);
    mouse(QEvent::MouseButtonRelease, secondStart + QPointF(15, 25), Qt::LeftButton, Qt::NoButton);
    CHECK(std::abs(canvas.imageRect().top() - secondPan.top() - 25) < 0.01 &&
          cv::norm(beforeMask, doc.maskChannels(), cv::NORM_INF) == 0 &&
          canvas.cursor().shape() == Qt::OpenHandCursor,
          "second drag pans without applying the previous tool");
    canvas.setPanMode(false);
    canvas.setActiveTool(Tool::Brush);
    canvas.setEdgeConstrained(false);
    doc.clearMask();
    mouse(QEvent::MouseButtonPress, sourcePoint(6, 7), Qt::LeftButton, Qt::LeftButton);
    mouse(QEvent::MouseButtonRelease, sourcePoint(6, 7), Qt::LeftButton, Qt::NoButton);
    CHECK(doc.mask().at<uchar>(7, 6) == 1, "editing resumes after drag mode is explicitly disabled");

    // A closed drawn mask fills across source-image edges, preserving other bones.
    doc.clearMask();
    doc.paintLine({1, 1}, {18, 1}, Label::Femur, 1);
    doc.paintLine({18, 1}, {18, 18}, Label::Femur, 1);
    doc.paintLine({18, 18}, {1, 18}, Label::Femur, 1);
    doc.paintLine({1, 18}, {1, 1}, Label::Femur, 1);
    doc.paintLine({15, 10}, {15, 10}, Label::Tibia, 1);
    canvas.setActiveTool(Tool::DrawFill);
    canvas.setDrawFillHoles(true);
    canvas.setEdgeConstrained(true);
    mouse(QEvent::MouseButtonPress, sourcePoint(6, 7), Qt::LeftButton, Qt::LeftButton);
    mouse(QEvent::MouseButtonRelease, sourcePoint(6, 7), Qt::LeftButton, Qt::NoButton);
    CHECK(doc.maskChannels().at<cv::Vec3b>(10, 15) == cv::Vec3b(0, 2, 1) &&
          doc.mask().at<uchar>(0, 0) == 0,
          "fill closed area ignores source edges and respects drawn mask boundary and overlap");
    doc.undo();
    CHECK(doc.maskChannels().at<cv::Vec3b>(10, 15) == cv::Vec3b(0, 2, 0), "closed-area fill is undoable");
    CHECK(panButton->parentWidget() == toolbarBatchButton->parentWidget(), "drag button is in top bar");
    panButton->click();
    CHECK(panButton->isChecked(), "drag button activates persistent pan");
    panButton->click();
    CHECK(!panButton->isChecked() && window.canvas()->cursor().shape() == Qt::CrossCursor,
          "second button click restores editing without dragging");

    drawFillTool->click();
    panButton->click();
    CHECK(panButton->isChecked() && !drawFillTool->isChecked(),
          "drag mode suspends the selected editing tool");
    panButton->click();
    CHECK(!panButton->isChecked() && drawFillTool->isChecked(),
          "turning drag off restores the previous tool");
    panButton->click();
    lassoTool->click();
    CHECK(!panButton->isChecked() && lassoTool->isChecked() &&
          window.canvas()->cursor().shape() == Qt::CrossCursor,
          "selecting an editing tool disables drag and activates the selected tool");
    panButton->click();
    lassoTool->click();
    CHECK(!panButton->isChecked() && lassoTool->isChecked(),
          "reselecting the previous tool also exits drag mode");

    // Startup and image changes require an explicit tool selection.
    MainWindow idleWindow;
    idleWindow.show();
    const QString idleInput = tmp.filePath("case_0000.png");
    cv::imwrite(idleInput.toStdString(), cv::Mat(20, 20, CV_8UC3, cv::Scalar(80, 80, 80)));
    auto clickIdleCanvas = [&] {
        auto* c = idleWindow.canvas();
        QPointF point = c->imageRect().center();
        QMouseEvent press(QEvent::MouseButtonPress, point, point, Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
        QMouseEvent release(QEvent::MouseButtonRelease, point, point, Qt::LeftButton, Qt::NoButton, Qt::NoModifier);
        QApplication::sendEvent(c, &press);
        QApplication::sendEvent(c, &release);
    };
    chooseFile(idleInput);
    QMetaObject::invokeMethod(&idleWindow, "onUpload", Qt::DirectConnection);
    clickIdleCanvas();
    CHECK(cv::countNonZero(idleWindow.document()->mask()) == 0,
          "new image starts idle and clicking cannot paint");
    idleWindow.findChild<QPushButton*>("aiFillToolBtn")->click();
    auto* idlePrompt = idleWindow.findChild<QComboBox*>("aiPromptType");
    idlePrompt->setCurrentIndex(1); // Programmatic restoration must not arm painting.
    clickIdleCanvas();
    CHECK(idleWindow.document()->aiFill().promptMask.empty() ||
          cv::countNonZero(idleWindow.document()->aiFill().promptMask) == 0,
          "opening AI Fill or restoring a prompt setting does not start painting");
    QMetaObject::invokeMethod(idlePrompt, "activated", Qt::DirectConnection, Q_ARG(int, 1));
    clickIdleCanvas();
    CHECK(!idleWindow.document()->aiFill().promptMask.empty() &&
          cv::countNonZero(idleWindow.document()->aiFill().promptMask) > 0,
          "explicit Paint Mask selection enables prompt painting");
    QTimer::singleShot(0, &idleWindow, [&] {
        auto* dialog = qobject_cast<QFileDialog*>(QApplication::activeModalWidget());
        CHECK(dialog && dialog->directory().absolutePath() == tmp.path(), "image picker remembers last opened folder");
        if (dialog) dialog->reject();
    });
    QMetaObject::invokeMethod(&idleWindow, "onUpload", Qt::DirectConnection);
    const QString typedOutput = tmp.filePath("typed-export");
    QDir().mkpath(typedOutput);
    QTimer::singleShot(0, &idleWindow, [&] {
        auto* dialog = qobject_cast<QFileDialog*>(QApplication::activeModalWidget());
        CHECK(dialog && QFileInfo(dialog->selectedFiles().value(0)).fileName() == "case.png",
              "export removes nnUNet channel suffix from the mask name");
        auto* folder = dialog ? dialog->findChild<QLineEdit*>("exportDestinationFolder") : nullptr;
        CHECK(folder, "export has an editable destination-folder field");
        if (folder) folder->setText(typedOutput);
        if (dialog) QMetaObject::invokeMethod(dialog, "accept", Qt::DirectConnection);
    });
    QMetaObject::invokeMethod(&idleWindow, "onExport", Qt::DirectConnection);
    CHECK(QFileInfo::exists(QDir(typedOutput).filePath("case.png")), "mask saves in the typed folder with nnUNet filename");
    QTimer::singleShot(0, &idleWindow, [&] {
        auto* dialog = qobject_cast<QFileDialog*>(QApplication::activeModalWidget());
        CHECK(dialog && dialog->directory().absolutePath() == typedOutput, "export remembers the last save folder");
        if (dialog) dialog->reject();
    });
    QMetaObject::invokeMethod(&idleWindow, "onExport", Qt::DirectConnection);

    idleWindow.document()->paintAIPrompt({3, 3}, {3, 3}, false, 1);
    idleWindow.document()->aiFill().femurBox = Box{1, 1, 10, 10};
    for (auto* button : idleWindow.findChildren<QPushButton*>())
        if (button->text().trimmed() == "Tibia") button->click();
    CHECK(idleWindow.document()->aiFill().promptMask.empty(), "changing anatomy clears the previous bone's mask prompt");
    CHECK(idleWindow.document()->aiFill().femurBox.has_value(), "anatomy selection preserves bone-specific box prompts");
    idleWindow.document()->aiFill().resultMask = cv::Mat(idleWindow.document()->mask().size(), CV_8UC1, cv::Scalar(1));
    bool exportBlocked = false;
    QTimer::singleShot(0, &idleWindow, [&] {
        auto* modal = QApplication::activeModalWidget();
        auto* message = qobject_cast<QMessageBox*>(modal);
        exportBlocked = message && message->text().contains("Apply or clear");
        if (auto* dialog = qobject_cast<QDialog*>(modal)) dialog->reject();
    });
    QMetaObject::invokeMethod(&idleWindow, "onExport", Qt::DirectConnection);
    CHECK(exportBlocked && !idleWindow.document()->aiFill().resultMask.empty(),
          "ordinary export blocks unapplied AI previews before choosing a file");

    // Tool profiles are independent except for the shared Brush/Eraser pair.
    auto clickTool = [&](const QString& name) {
        for (auto* b : idleWindow.findChildren<QPushButton*>())
            if (b->text().contains(name)) { b->click(); return; }
    };
    auto* size = idleWindow.findChild<QSlider*>("brushSizeSlider");
    auto* threshold = idleWindow.findChild<QSlider*>("edgeThresholdSlider");
    auto* stopEdges = idleWindow.findChild<QCheckBox*>("constrainToEdges");
    clickTool("Brush"); size->setValue(37); threshold->setValue(67); stopEdges->setChecked(true);
    clickTool("Eraser");
    CHECK(size->value() == 37 && threshold->value() == 67 && stopEdges->isChecked(), "Brush and Eraser share size and edge settings");
    size->setValue(41);
    idleWindow.findChild<QPushButton*>("lassoToolBtn")->click();
    CHECK(threshold->value() == 30 && !stopEdges->isChecked(), "Lasso has independent defaults");
    threshold->setValue(92);
    idleWindow.showFillAlgorithm(0);
    CHECK(size->value() == 20 && threshold->value() == 30, "Fill does not inherit brush or lasso settings");
    size->setValue(13); threshold->setValue(24);
    clickTool("Brush");
    CHECK(size->value() == 41 && threshold->value() == 67, "Brush restores settings last used by Eraser");
    idleWindow.findChild<QPushButton*>("lassoToolBtn")->click();
    CHECK(threshold->value() == 92, "Lasso restores its own edge threshold");
    idleWindow.showFillAlgorithm(0);
    CHECK(size->value() == 13 && threshold->value() == 24, "Fill restores its own settings");
    auto* drawMode = idleWindow.findChild<QPushButton*>("drawOutlineButton");
    auto* fillMode = idleWindow.findChild<QPushButton*>("fillClosedAreaButton");
    idleWindow.findChild<QPushButton*>("drawFillToolBtn")->click(); fillMode->click();
    clickTool("Brush"); idleWindow.findChild<QPushButton*>("drawFillToolBtn")->click();
    CHECK(drawMode->isChecked() && !fillMode->isChecked(), "Draw & Fill reopens in Draw outline mode");

    idleWindow.document()->aiFill().resultMask.release();
    const auto beforeReopen = idleWindow.document()->maskChannels().clone();
    bool warned = false;
    QTimer reopenDialogs;
    QObject::connect(&reopenDialogs, &QTimer::timeout, &idleWindow, [&] {
        if (auto* d = qobject_cast<QFileDialog*>(QApplication::activeModalWidget())) {
            d->selectFile(idleInput);
            QMetaObject::invokeMethod(d, "accept", Qt::QueuedConnection);
        } else if (auto* box = qobject_cast<QMessageBox*>(QApplication::activeModalWidget())) {
            warned = box->windowTitle().contains("Already processed") && box->text().contains("case.png");
            box->button(QMessageBox::Cancel)->click();
        }
    });
    reopenDialogs.start(5);
    QMetaObject::invokeMethod(&idleWindow, "onUpload", Qt::DirectConnection);
    reopenDialogs.stop();
    CHECK(warned && cv::norm(beforeReopen, idleWindow.document()->maskChannels(), cv::NORM_INF) == 0,
          "reopening a saved case warns and Cancel preserves the current annotation");
    MainWindow freshSession;
    bool freshWarned = false;
    QTimer freshDialogs;
    QObject::connect(&freshDialogs, &QTimer::timeout, &freshSession, [&] {
        if (auto* d = qobject_cast<QFileDialog*>(QApplication::activeModalWidget())) {
            d->selectFile(idleInput); QMetaObject::invokeMethod(d, "accept", Qt::QueuedConnection);
        } else if (auto* box = qobject_cast<QMessageBox*>(QApplication::activeModalWidget())) {
            freshWarned = true; box->button(QMessageBox::Cancel)->click();
        }
    });
    freshDialogs.start(5); QMetaObject::invokeMethod(&freshSession, "onUpload", Qt::DirectConnection); freshDialogs.stop();
    CHECK(freshSession.document()->hasImage() && !freshWarned, "processed-file warnings are limited to the current session");
    QTimer::singleShot(0, &freshSession, [&] {
        auto* d = qobject_cast<QFileDialog*>(QApplication::activeModalWidget());
        CHECK(d && d->directory().absolutePath() == typedOutput, "last successful save folder survives a new window");
        if (d) d->reject();
    });
    QMetaObject::invokeMethod(&freshSession, "onExport", Qt::DirectConnection);

    // Highlighting a file previews it without opening/replacing the document.
    const auto sourceBeforePreview = freshSession.document()->sourcePath();
    QTimer::singleShot(0, &freshSession, [&] {
        auto* dialog = qobject_cast<QFileDialog*>(QApplication::activeModalWidget());
        CHECK(dialog && dialog->findChild<QWidget*>("xrayFilePreview"), "Upload X-ray includes preview pane");
        if (dialog) dialog->reject();
    });
    QMetaObject::invokeMethod(&freshSession, "onUpload", Qt::DirectConnection);
    CHECK(freshSession.document()->sourcePath() == sourceBeforePreview, "cancelling preview preserves open image");
    QFileDialog previewDialog(nullptr, "Preview regression", tmp.path());
    previewDialog.setFileMode(QFileDialog::ExistingFile);
    configureFileBrowser(previewDialog);
    new ImageFilePreview(previewDialog);
    previewDialog.show();
    auto* previewImage = previewDialog.findChild<QLabel*>("xrayPreviewImage");
    auto* previewDetails = previewDialog.findChild<QLabel*>("xrayPreviewDetails");
    auto* previewView = previewDialog.findChild<QAbstractItemView*>("treeView");
    auto* previewFiles = qobject_cast<QFileSystemModel*>(previewView->model());
    auto previewWait = [](auto condition) {
        QElapsedTimer elapsed; elapsed.start();
        while (!condition() && elapsed.elapsed() < 5000) { QApplication::processEvents(); QThread::msleep(1); }
        return condition();
    };
    CHECK(previewWait([&] { return previewFiles->index(input).isValid(); }), "preview file is available");
    previewView->setCurrentIndex(previewFiles->index(input));
    CHECK(previewWait([&] { return !previewImage->pixmap().isNull(); }), "highlighting PNG displays preview before Open");
    CHECK(previewDetails->text().contains(QFileInfo(input).fileName()) && previewDetails->text().contains("pixels"),
          "preview displays filename and dimensions");
    const auto tiffPreviewPath = tmp.filePath("preview-16bit.tiff");
    cv::imwrite(tiffPreviewPath.toStdString(), cv::Mat(80, 40, CV_16UC1, cv::Scalar(32768)));
    previewDialog.currentChanged(tiffPreviewPath);
    CHECK(previewWait([&] { return previewDetails->text().contains("40 × 80 pixels"); }), "16-bit TIFF can be previewed");
    const auto invalidPreviewPath = tmp.filePath("invalid-preview.png");
    QFile invalidPreview(invalidPreviewPath); CHECK(invalidPreview.open(QIODevice::WriteOnly), "create unreadable image fixture"); invalidPreview.write("invalid"); invalidPreview.close();
    previewDialog.currentChanged(invalidPreviewPath);
    CHECK(previewImage->pixmap().isNull(), "new selection immediately clears previous preview");
    CHECK(previewWait([&] { return previewImage->text() == "Preview unavailable"; }), "unreadable file shows unavailable state");
    previewDialog.currentChanged(input);
    previewDialog.directoryEntered(tmp.path());
    QApplication::processEvents();
    CHECK(previewImage->pixmap().isNull() && previewDetails->text().isEmpty(), "folder navigation clears pending preview");
    previewDialog.close();

    const QString childFolder = tmp.filePath("navigation-child");
    QDir().mkpath(childFolder);
    QFileDialog browser(nullptr, "Navigation regression", tmp.path());
    browser.setFileMode(QFileDialog::ExistingFile); configureFileBrowser(browser); browser.show();
    auto* fileView = browser.findChild<QAbstractItemView*>("treeView");
    QElapsedTimer loading; loading.start();
    while (fileView->model()->rowCount(fileView->rootIndex()) < 2 && loading.elapsed() < 2000) {
        QApplication::processEvents(); QThread::msleep(1);
    }
    fileView->setCurrentIndex(fileView->model()->index(0, 0, fileView->rootIndex()));
    QKeyEvent arrow(QEvent::KeyPress, Qt::Key_Down, Qt::NoModifier);
    QApplication::sendEvent(fileView, &arrow);
    CHECK(fileView->currentIndex().row() == 1, "file browser supports keyboard Down navigation");
    auto* fs = qobject_cast<QFileSystemModel*>(fileView->model());
    CHECK(browser.findChild<QToolButton*>("backButton")->isEnabled() && browser.findChild<QToolButton*>("forwardButton")->isEnabled(), "folder arrows enabled without navigation history");
    fileView->setCurrentIndex(fs->index(childFolder));
    browser.findChild<QToolButton*>("forwardButton")->click();
    CHECK(browser.directory().absolutePath() == childFolder, "file browser opens the selected folder");
    browser.findChild<QToolButton*>("backButton")->click();
    CHECK(browser.directory().absolutePath() == tmp.path(), "file browser returns to its parent folder");
    browser.close();
    QFileDialog saveBrowser(nullptr, "Save navigation regression", childFolder);
    saveBrowser.setAcceptMode(QFileDialog::AcceptSave);
    configureFileBrowser(saveBrowser); saveBrowser.show(); QApplication::processEvents();
    auto* saveUp = saveBrowser.findChild<QToolButton*>("backButton");
    CHECK(saveUp && saveUp->isEnabled(), "save dialog parent arrow is enabled without history");
    saveUp->click();
    CHECK(saveBrowser.directory().absolutePath() == tmp.path(), "save dialog arrow goes up one folder");
    CHECK(saveBrowser.findChild<QToolButton*>("forwardButton")->isEnabled(), "save dialog child arrow remains enabled");
    saveBrowser.close();

    std::printf("\n%d UI test failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
