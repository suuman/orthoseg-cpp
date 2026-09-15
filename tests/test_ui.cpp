#include "CanvasWidget.h"
#include "MainWindow.h"
#include <QApplication>
#include <QEvent>
#include <QFileDialog>
#include <QFileInfo>
#include <QSlider>
#include <QTemporaryDir>
#include <QTimer>
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

    std::printf("\n%d UI test failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
