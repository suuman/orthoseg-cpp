#pragma once
#include "MonaiClient.h"
#include <QDialog>
#include <QTimer>

class QLabel;
class QComboBox;
class QPushButton;
class QPlainTextEdit;
class QTableWidget;
class QSpinBox;

namespace orthoseg {
class ModelManagementDialog : public QDialog {
public:
    explicit ModelManagementDialog(QWidget* parent = nullptr, QUrl base = MonaiClient::configuredUrl());
    void refresh();
    void setBackendUrl(QUrl base);
    void selectModel(const QString& model);
protected:
    void showEvent(QShowEvent* event) override;
    void hideEvent(QHideEvent* event) override;
private:
    void refreshCases();
    void exportCases();
    void exportCasesPage(const QString& path, const QString& model, unsigned revision, int offset, int total, QByteArray csv);
    bool exportingCases_ = false;
    QPushButton* exportCases_;
    int caseOffset_ = 0;
    int caseTotal_ = 0;
    unsigned caseRevision_ = 0;
    bool casesBusy_ = false;
    QTableWidget* cases_;
    QLabel* casePage_;
    QPushButton* previousCases_;
    QPushButton* nextCases_;
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
    QComboBox* model_;
    QSpinBox* reminderCases_;
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
