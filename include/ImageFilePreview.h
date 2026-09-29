#pragma once
#include <QFileDialog>
#include <QFileInfo>
#include <QGridLayout>
#include <QImage>
#include <QLabel>
#include <QPixmap>
#include <QThread>
#include <QTimer>
#include <QVBoxLayout>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <memory>

namespace orthoseg {
// Preview decoding stays off the GUI thread and never changes the open document.
class ImageFilePreview : public QWidget {
public:
    explicit ImageFilePreview(QFileDialog& dialog) : QWidget(&dialog) {
        setObjectName("xrayFilePreview");
        setFixedWidth(300);
        auto* layout = new QVBoxLayout(this);
        image_ = new QLabel("Highlight an X-ray to preview", this);
        image_->setObjectName("xrayPreviewImage");
        image_->setAlignment(Qt::AlignCenter);
        image_->setWordWrap(true);
        image_->setFixedSize(280, 340);
        image_->setStyleSheet("background:#111827;color:#cbd5e1;border:1px solid #475569;");
        details_ = new QLabel(this);
        details_->setObjectName("xrayPreviewDetails");
        details_->setTextFormat(Qt::PlainText);
        details_->setWordWrap(true);
        layout->addWidget(image_);
        layout->addWidget(details_);
        layout->addStretch();
        timer_.setSingleShot(true);
        timer_.setInterval(150);
        connect(&timer_, &QTimer::timeout, this, [this] { decode(); });
        connect(&dialog, &QFileDialog::currentChanged, this, [this](const QString& path) { select(path); });
        connect(&dialog, &QFileDialog::directoryEntered, this, [this] { select({}); });
        if (auto* grid = qobject_cast<QGridLayout*>(dialog.layout()))
            grid->addWidget(this, 0, grid->columnCount(), grid->rowCount(), 1);
        dialog.resize(1000, 600);
    }
private:
    struct Result { QImage image; int width = 0; int height = 0; };
    QLabel* image_;
    QLabel* details_;
    QTimer timer_;
    QString path_;
    unsigned revision_ = 0;
    bool decoding_ = false;

    void select(const QString& path) {
        ++revision_;
        timer_.stop();
        path_ = QFileInfo(path).isFile() ? path : QString();
        image_->clear();
        details_->clear();
        if (path_.isEmpty()) {
            image_->setText("Highlight an X-ray to preview");
            return;
        }
        details_->setText(QFileInfo(path_).fileName());
        image_->setText("Loading preview…");
        timer_.start(150);
    }
    void decode() {
        if (decoding_ || path_.isEmpty()) return;
        decoding_ = true;
        const auto path = path_;
        const auto revision = revision_;
        auto result = std::make_shared<Result>();
        auto* thread = QThread::create([path, result] {
            try {
                // Match the editor's decoder, including 16-bit TIFF display conversion.
                const auto original = cv::imread(path.toStdString(), cv::IMREAD_COLOR);
                if (original.empty()) return;
                result->width = original.cols;
                result->height = original.rows;
                const auto size = QSize(original.cols, original.rows).scaled(280, 340, Qt::KeepAspectRatio);
                cv::Mat thumbnail;
                cv::resize(original, thumbnail, cv::Size(size.width(), size.height()), 0, 0, cv::INTER_AREA);
                cv::cvtColor(thumbnail, thumbnail, cv::COLOR_BGR2RGB);
                result->image = QImage(thumbnail.data, thumbnail.cols, thumbnail.rows,
                    static_cast<int>(thumbnail.step), QImage::Format_RGB888).copy();
            } catch (const std::exception&) {
                result->image = {};
            }
        });
        connect(thread, &QThread::finished, thread, &QObject::deleteLater);
        connect(thread, &QThread::finished, this, [this, revision, path, result] {
            decoding_ = false;
            if (revision != revision_) {
                if (!path_.isEmpty()) timer_.start(0);
                return;
            }
            if (result->image.isNull()) {
                image_->setText("Preview unavailable");
            } else {
                image_->setPixmap(QPixmap::fromImage(result->image));
                details_->setText(QString("%1\n%2 × %3 pixels")
                    .arg(QFileInfo(path).fileName()).arg(result->width).arg(result->height));
            }
        });
        thread->start();
    }
};
} // namespace orthoseg
