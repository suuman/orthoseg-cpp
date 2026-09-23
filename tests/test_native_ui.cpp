#include "MainWindow.h"
#include <QApplication>
#include <QComboBox>
#include <QDialog>
#include <QElapsedTimer>
#include <QFileInfo>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QTemporaryDir>
#include <QThread>
#include <QTimer>
#include <opencv2/imgcodecs.hpp>
#include <cstdio>

using namespace orthoseg;

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    if (argc != 2) return 2;
    QTemporaryDir temp;
    if (!temp.isValid()) return 2;
    const auto sourcePath = temp.filePath("native-xray.png");
    cv::Mat source(512, 128, CV_8UC3, cv::Scalar(0, 0, 0));
    source(cv::Rect(20, 50, 85, 410)).setTo(cv::Scalar(120, 120, 120));
    if (!cv::imwrite(sourcePath.toStdString(), source)) return 2;
    qputenv("ORTHOSEG_NNUNET_MODEL", argv[1]);
    MainWindow window;
    window.show();
    if (!window.document()->loadImage(sourcePath.toStdString())) return 2;
    window.canvas()->refresh();
    auto* aiTool = window.findChild<QPushButton*>("aiFillToolBtn");
    auto* selector = window.findChild<QComboBox*>("aiModelSelector");
    auto* run = window.findChild<QPushButton*>("runAIFill");
    auto* modelsButton = window.findChild<QPushButton*>("modelDirBtn");
    auto* apply = window.findChild<QPushButton*>("applyAIResult");
    auto* clear = window.findChild<QPushButton*>("clearAIResult");
    if (!aiTool || !selector || !run || !modelsButton || !apply || !clear) return 2;
    aiTool->click();
    selector->setCurrentIndex(4);
    if (!QFileInfo(QString::fromLocal8Bit(argv[1])).isFile() || !modelsButton->text().contains("Local Models") ||
        !apply->isVisible() || !clear->isVisible()) return 2;
    bool localDialogValid = false;
    QTimer::singleShot(0, [&] {
        auto* dialog = qobject_cast<QDialog*>(QApplication::activeModalWidget());
        if (!dialog) return;
        auto* model = dialog->findChild<QLineEdit*>("nativeNnUnetModel");
        auto* device = dialog->findChild<QComboBox*>("nativeNnUnetDevice");
        localDialogValid = model && device && QFileInfo(model->text()).isFile() &&
                           device->currentData().toString() == "auto";
        dialog->accept();
    });
    modelsButton->click();
    if (!localDialogValid) return 2;
    int errors = 0;
    QTimer dialogs;
    QObject::connect(&dialogs, &QTimer::timeout, [&] {
        if (auto* box = qobject_cast<QMessageBox*>(QApplication::activeModalWidget())) {
            ++errors;
            box->accept();
        }
    });
    dialogs.start(10);
    run->click();
    QElapsedTimer clock;
    clock.start();
    while (!run->isEnabled() && clock.elapsed() < 90000) {
        QApplication::processEvents();
        QThread::msleep(5);
    }
    QApplication::processEvents();
    const auto& preview = window.document()->aiFill().resultMask;
    if (!run->isEnabled() || errors || preview.size() != source.size() ||
        preview.type() != CV_8UC1 || cv::countNonZero(preview > 2) ||
        cv::countNonZero(preview) == 0 || !window.document()->aiFill().resultReplacesAnatomy ||
        cv::countNonZero(window.document()->mask()) != 0) {
        std::fprintf(stderr, "Native AI Fill did not produce a valid preview (errors=%d)\n", errors);
        return 1;
    }
    apply->click();
    const auto& mask = window.document()->mask();
    if (!run->isEnabled() || errors || mask.size() != source.size() ||
        mask.type() != CV_8UC1 || cv::countNonZero(mask > 2) ||
        cv::countNonZero(mask) == 0 || !window.document()->canUndo()) {
        std::fprintf(stderr, "Native AI Fill did not apply a valid, undoable mask (errors=%d)\n", errors);
        return 1;
    }
    window.document()->undo();
    if (cv::countNonZero(window.document()->mask()) != 0) return 1;
    std::printf("Native AI Fill applied source-size mask and undo without MONAI server\n");
    return 0;
}
