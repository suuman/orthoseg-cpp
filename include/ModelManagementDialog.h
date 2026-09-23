#pragma once
#include "MonaiClient.h"
#include <QDialog>
#include <QTimer>

class QLabel;
class QPushButton;
class QPlainTextEdit;
class QTableWidget;

namespace orthoseg {
class ModelManagementDialog : public QDialog {
public:
    explicit ModelManagementDialog(QWidget* parent = nullptr, QUrl base = MonaiClient::configuredUrl());
    void refresh();
    void setBackendUrl(QUrl base);
protected:
    void showEvent(QShowEvent* event) override;
    void hideEvent(QHideEvent* event) override;
private:
    void render(const QJsonObject& value);
    void updateActions();
    void startTraining();
    void promoteCandidate();
    MonaiClient client_;
    QTimer poll_;
    QJsonObject snapshot_;
    bool busy_ = false;
    bool online_ = false;
    unsigned backendRevision_ = 0;
    QLabel* backend_;
    QLabel* production_;
    QLabel* dataset_;
    QLabel* candidate_;
    QLabel* job_;
    QLabel* policy_;
    QTableWidget* metrics_;
    QPlainTextEdit* log_;
    QPushButton* train_;
    QPushButton* promote_;
    QPushButton* refresh_;
};
} // namespace orthoseg
