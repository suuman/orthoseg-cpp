#include "ModelManagementDialog.h"
#include <QLabel>
#include <QPushButton>
#include <QPlainTextEdit>
#include <QTableWidget>
#include <QHeaderView>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QMessageBox>
#include <QJsonArray>
#include <QShowEvent>
#include <QHideEvent>
#include <utility>

namespace orthoseg {
namespace {
QString text(const QJsonValue& value) {
    if (value.isString()) return value.toString().isEmpty() ? "N/A" : value.toString();
    if (value.isDouble()) return QString::number(value.toDouble());
    return "N/A";
}
bool active(const QJsonObject& value) {
    return value["training_active"].toBool() ||
        QStringList{"queued", "preparing", "training", "validating"}.contains(value["job"].toObject()["status"].toString());
}
QString metricText(const QJsonValue& value) { return value.isDouble() ? QString::number(value.toDouble(), 'f', 4) : "N/A"; }
}
ModelManagementDialog::ModelManagementDialog(QWidget* parent, QUrl base) : QDialog(parent), client_(this, std::move(base)) {
    setWindowTitle("AI Model Management");
    setObjectName("modelManagementDialog");
    setModal(false);
    resize(700, 780);
    setStyleSheet("QDialog{background:#0f172a;color:#e2e8f0;} QLabel{color:#e2e8f0;} "
                  "QPushButton{padding:8px;background:#1e293b;color:#e2e8f0;border:1px solid #475569;border-radius:6px;} "
                  "QPushButton:disabled{color:#64748b;} QTableWidget,QPlainTextEdit{background:#1e293b;color:#e2e8f0;}");
    auto* layout = new QVBoxLayout(this);
    auto label = [&](const char* name) {
        auto* value = new QLabel("N/A", this);
        value->setObjectName(name);
        value->setTextFormat(Qt::PlainText);
        value->setWordWrap(true);
        value->setTextInteractionFlags(Qt::TextSelectableByMouse);
        layout->addWidget(value);
        return value;
    };
    backend_ = label("managementBackend");
    production_ = label("managementProduction");
    dataset_ = label("managementDataset");
    candidate_ = label("managementCandidate");
    metrics_ = new QTableWidget(3, 3, this);
    metrics_->setObjectName("managementMetrics");
    metrics_->setHorizontalHeaderLabels({"Validation metric", "Production", "Candidate"});
    metrics_->verticalHeader()->hide();
    metrics_->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    metrics_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    metrics_->setMaximumHeight(135);
    layout->addWidget(metrics_);
    policy_ = label("managementPolicy");
    job_ = label("managementJob");
    log_ = new QPlainTextEdit(this);
    log_->setObjectName("managementLog");
    log_->setReadOnly(true);
    log_->setMaximumBlockCount(80);
    log_->setPlaceholderText("Recent training log");
    layout->addWidget(log_, 1);
    auto* actions = new QHBoxLayout;
    train_ = new QPushButton("Fine-tune New Candidate", this);
    train_->setObjectName("managementTrain");
    promote_ = new QPushButton("Promote Candidate", this);
    promote_->setObjectName("managementPromote");
    refresh_ = new QPushButton("Refresh", this);
    refresh_->setObjectName("managementRefresh");
    actions->addWidget(train_); actions->addWidget(promote_); actions->addWidget(refresh_);
    layout->addLayout(actions);
    connect(train_, &QPushButton::clicked, this, &ModelManagementDialog::startTraining);
    connect(promote_, &QPushButton::clicked, this, &ModelManagementDialog::promoteCandidate);
    connect(refresh_, &QPushButton::clicked, this, &ModelManagementDialog::refresh);
    poll_.setInterval(5000);
    connect(&poll_, &QTimer::timeout, this, &ModelManagementDialog::refresh);
    updateActions();
}
void ModelManagementDialog::showEvent(QShowEvent* event) { QDialog::showEvent(event); refresh(); }
void ModelManagementDialog::hideEvent(QHideEvent* event) { poll_.stop(); QDialog::hideEvent(event); }
void ModelManagementDialog::setBackendUrl(QUrl base) {
    if (client_.baseUrl() == base) return;
    client_.setBaseUrl(std::move(base));
    ++backendRevision_;
    busy_ = false;
    online_ = false;
    snapshot_ = {};
    poll_.stop();
    render(snapshot_);
    updateActions();
    if (isVisible()) refresh();
}
void ModelManagementDialog::updateActions() {
    const auto candidate = snapshot_["candidate"].toObject();
    train_->setEnabled(online_ && !busy_ && !active(snapshot_) && snapshot_["can_start"].toBool());
    promote_->setEnabled(online_ && !busy_ && !active(snapshot_) && candidate["valid"].toBool() &&
                         !candidate["version"].toString().isEmpty() &&
                         candidate["version"] != snapshot_["production"].toObject()["version"]);
    refresh_->setEnabled(!busy_);
}
void ModelManagementDialog::refresh() {
    if (busy_) return;
    busy_ = true;
    const auto revision = backendRevision_;
    poll_.stop();
    updateActions();
    client_.managementStatus([this, revision](const QJsonObject& value, const QString& error) {
        if (revision != backendRevision_) return;
        busy_ = false;
        online_ = error.isEmpty();
        if (!online_) {
            snapshot_ = {};
            render(snapshot_);
            backend_->setText("Backend unavailable or management disabled.\n" + error);
        } else {
            snapshot_ = value;
            render(value);
            if (active(value) && isVisible()) poll_.start();
        }
        updateActions();
    });
}
void ModelManagementDialog::render(const QJsonObject& value) {
    const auto backend = value["backend"].toObject();
    const auto production = value["production"].toObject();
    const auto dataset = value["dataset"].toObject();
    const auto candidate = value["candidate"].toObject();
    const auto job = value["job"].toObject();
    const auto policy = value["training_policy"].toObject();
    backend_->setText(QString("Backend: %1   Device: %2   GPU: %3\nURL: %4\nModel loaded: %5")
        .arg(text(backend["status"]), text(backend["device"]), text(backend["gpu"]),
             client_.baseUrl().toString(), backend.contains("model_loaded") ? (backend["model_loaded"].toBool() ? "Yes" : "No") : "N/A"));
    production_->setText(QString("Production on disk: %1\nLoaded for inference: %2%3\nArchitecture: %4   Cases: %5\nCreated: %6   Parent: %7")
        .arg(text(production["version"]), text(value["loaded_model_version"]),
             value["restart_required"].toBool() ? " — restart backend required" : "", text(production["architecture"]),
             text(production["training_case_count"]), text(production["created_at"]), text(production["parent_model_version"])));
    dataset_->setText(QString("Approved cases: %1   New/revised since training: %2\nValidation pairs: %3%4")
        .arg(text(dataset["total_approved_cases"]), text(dataset["new_cases_since_last_training"]), text(dataset["validation_case_count"]),
             dataset.contains("validation_pairs_match") && !dataset["validation_pairs_match"].toBool() ? " — unmatched files; repair dataset before training" : ""));
    candidate_->setText(QString("Latest completed candidate: %1   Status: %2\nParent: %3")
        .arg(text(candidate["version"]), text(candidate["status"]), text(candidate["parent_model_version"])));
    const QStringList keys{"mean_foreground_dice", "femur_dice", "tibia_dice"};
    const QStringList names{"Mean foreground Dice", "Femur Dice", "Tibia Dice"};
    for (int row = 0; row < keys.size(); ++row) {
        metrics_->setItem(row, 0, new QTableWidgetItem(names[row]));
        metrics_->setItem(row, 1, new QTableWidgetItem(metricText(production["validation_metrics"].toObject()[keys[row]])));
        metrics_->setItem(row, 2, new QTableWidgetItem(metricText(candidate["validation_metrics"].toObject()[keys[row]])));
    }
    policy_->setText(QString("Use backend defaults: %1 epochs. Administrative jobs use %2 (%3 threads).\nProduction is not replaced automatically; inference stays available.")
        .arg(text(policy["epochs"]), text(policy["device"]), text(policy["cpu_threads"])));
    QString status = job.isEmpty() ? (value["training_active"].toBool() ? "Training active (local script)" : "Idle") : text(job["status"]);
    job_->setText(QString("Training status: %1\nCandidate: %2   Epoch: %3 / %4   Best Dice: %5\nStarted: %6%7")
        .arg(status, text(job["candidate_version"]), text(job["epoch"]), text(job["epochs"]), metricText(job["best_validation_dice"]),
             text(job["started_at"]), job["error"].isString() ? "\nFine-tuning failed: " + job["error"].toString() : ""));
    QStringList lines;
    for (const auto& line : job["recent_log"].toArray()) lines.append(line.toString());
    log_->setPlainText(lines.join('\n'));
}
void ModelManagementDialog::startTraining() {
    if (!train_->isEnabled()) return;
    busy_ = true; poll_.stop(); updateActions();
    const auto dataset = snapshot_["dataset"].toObject();
    const auto production = snapshot_["production"].toObject();
    const auto answer = QMessageBox::question(this, "Start Fine-tuning",
        QString("Start fine-tuning a new candidate model?\n\nProduction: %1\nApproved cases: %2\nNew/revised cases: %3\nValidation pairs: %4\n\nUse backend defaults. The current production model will NOT be replaced automatically.")
            .arg(text(production["version"]), text(dataset["total_approved_cases"]), text(dataset["new_cases_since_last_training"]), text(dataset["validation_case_count"])),
        QMessageBox::Yes | QMessageBox::Cancel, QMessageBox::Cancel);
    if (answer != QMessageBox::Yes) { busy_ = false; updateActions(); return; }
    const auto revision = backendRevision_;
    client_.startTraining([this, revision](const QJsonObject& job, const QString& error) {
        if (revision != backendRevision_) return;
        busy_ = false;
        if (!error.isEmpty()) QMessageBox::warning(this, "Fine-tuning failed", error);
        else { snapshot_["job"] = job; snapshot_["training_active"] = true; render(snapshot_); }
        refresh();
    });
}
void ModelManagementDialog::promoteCandidate() {
    if (!promote_->isEnabled()) return;
    busy_ = true; poll_.stop(); updateActions();
    const auto version = snapshot_["candidate"].toObject()["version"].toString();
    const auto answer = QMessageBox::question(this, "Promote Candidate",
        QString("Promote candidate model?\n\nCurrent production: %1\nNew candidate: %2\n\nThe previous production release will be retained in the archive. Restart the backend to load the new model.")
            .arg(text(snapshot_["production"].toObject()["version"]), version),
        QMessageBox::Yes | QMessageBox::Cancel, QMessageBox::Cancel);
    if (answer != QMessageBox::Yes) { busy_ = false; updateActions(); return; }
    const auto revision = backendRevision_;
    client_.promoteCandidate(version, [this, revision](const QJsonObject&, const QString& error) {
        if (revision != backendRevision_) return;
        busy_ = false;
        if (!error.isEmpty()) QMessageBox::warning(this, "Promotion failed", error);
        else QMessageBox::information(this, "Candidate promoted", "Candidate promoted successfully. Restart the MONAI backend to load the new model.");
        refresh();
    });
}
} // namespace orthoseg
