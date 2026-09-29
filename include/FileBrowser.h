#pragma once
#include <QAbstractItemView>
#include <QFileDialog>
#include <QFileInfo>
#include <QFileSystemModel>
#include <QLayout>
#include <QToolButton>
#include <QTimer>

namespace orthoseg {
// Replace history arrows with explicit parent/selected-child folder navigation.
inline void configureFileBrowser(QFileDialog& dialog) {
    dialog.setOption(QFileDialog::DontUseNativeDialog);
    dialog.setViewMode(QFileDialog::Detail);
    auto view = [&dialog]() -> QAbstractItemView* {
        return dialog.findChild<QAbstractItemView*>("treeView");
    };
    auto replace = [&dialog](const char* name, const QString& tooltip, auto action) {
        auto* history = dialog.findChild<QToolButton*>(name);
        if (!history) return;
        auto* button = new QToolButton(history->parentWidget());
        button->setIcon(history->icon());
        button->setToolTip(tooltip);
        button->setAccessibleName(tooltip);
        history->setObjectName(QString(name) + "History");
        button->setObjectName(name);
        if (auto* item = dialog.layout()->replaceWidget(history, button)) delete item;
        history->hide();
        button->setEnabled(true);
        QObject::connect(button, &QToolButton::clicked, &dialog, action);
    };
    replace("backButton", "Up one folder", [&dialog] {
        auto directory = dialog.directory();
        if (directory.cdUp()) {
            dialog.setDirectory(directory);
            dialog.directoryEntered(directory.absolutePath());
        }
    });
    replace("forwardButton", "Open selected folder (one level down)", [&dialog, view] {
        auto* list = view();
        auto* files = list ? qobject_cast<QFileSystemModel*>(list->model()) : nullptr;
        const auto index = list ? list->currentIndex() : QModelIndex();
        const auto path = files && index.isValid() ? files->filePath(index) : QString();
        if (QFileInfo(path).isDir() && QFileInfo(path).absolutePath() == dialog.directory().absolutePath()) {
            dialog.setDirectory(path);
            dialog.directoryEntered(dialog.directory().absolutePath());
        } else if (list) {
            list->setFocus();
        }
    });
    QTimer::singleShot(0, &dialog, [view] { if (auto* list = view()) list->setFocus(); });
}
inline QString browseFile(QWidget* parent, const QString& title, const QString& directory, const QString& filter) {
    QFileDialog dialog(parent, title, directory, filter);
    dialog.setFileMode(QFileDialog::ExistingFile);
    configureFileBrowser(dialog);
    return dialog.exec() == QDialog::Accepted ? dialog.selectedFiles().value(0) : QString();
}
}
