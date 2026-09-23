#include "CanvasWidget.h"
#include "MainWindow.h"
#include <QApplication>
#include <QEvent>
#include <QFileDialog>
#include <QFileInfo>
#include <QSlider>
#include <QTemporaryDir>
#include <QTimer>
#include <QMouseEvent>
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
    QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
    QApplication app(argc, argv);
    MainWindow window;
    window.show();
    QApplication::processEvents();

    QSlider* brush = nullptr;
    for (auto* slider : window.findChildren<QSlider*>())
        if (slider->minimum() == 2 && slider->maximum() == 100) brush = slider;
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

    // Source-coordinate box gestures must follow the same transform as rendering.
    CHECK(doc.loadImage(input.toStdString()), "load coordinate test image");
    canvas.refresh();
    canvas.resize(640, 480);
    canvas.setActiveTool(Tool::AIFill);
    auto mouse = [&canvas](QEvent::Type type, QPointF position, Qt::MouseButton button,
                           Qt::MouseButtons buttons) {
        QMouseEvent event(type, position, position, button, buttons, Qt::NoModifier);
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
        CHECK(cv::countNonZero(window.document()->aiFill().promptMask) == 2,
              "Use AI Result button copies resultMask into promptMask");
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

    std::printf("\n%d UI test failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
