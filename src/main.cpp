#include "MainWindow.h"
#include <QApplication>
#include <QTimer>
#include <QPixmap>

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    app.setApplicationName("OrthoSeg");

    // Global dark styling for sliders and scrollbars to match the reference.
    app.setStyleSheet(R"(
        QSlider::groove:horizontal {
            height: 6px; background: #1e293b; border-radius: 3px;
        }
        QSlider::handle:horizontal {
            width: 14px; margin: -5px 0; border-radius: 7px; background: #38bdf8;
        }
        QSlider::sub-page:horizontal { background: #38bdf8; border-radius: 3px; }
        QToolTip { background: #1e293b; color: #e2e8f0; border: 1px solid #334155; }
    )");

    orthoseg::MainWindow win;
    win.show();

    // Headless UI check: --fill-algo <idx> switches to Fill with that
    // algorithm; --screenshot <out.png> captures the window and exits.
    QStringList args = app.arguments();
    int ai = args.indexOf("--fill-algo");
    if (ai >= 0 && ai + 1 < args.size())
        win.showFillAlgorithm(args[ai + 1].toInt());
    int si = args.indexOf("--screenshot");
    if (si >= 0 && si + 1 < args.size()) {
        QString out = args[si + 1];
        QTimer::singleShot(400, &app, [&win, out, &app] {
            win.grab().save(out);
            app.quit();
        });
    }
    return app.exec();
}
