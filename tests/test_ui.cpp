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
    }
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
    CHECK(combo && combo->count() == 3, "all three AI prompt modes are exposed");
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

    std::printf("\n%d UI test failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
