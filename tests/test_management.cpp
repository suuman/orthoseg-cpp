#include "MainWindow.h"
#include "ModelManagementDialog.h"
#include <QApplication>
#include <QAction>
#include <QLabel>
#include <QComboBox>
#include <QPushButton>
#include <QMessageBox>
#include <QJsonDocument>
#include <QJsonArray>
#include <QTableWidget>
#include <QTcpServer>
#include <QTcpSocket>
#include <QElapsedTimer>
#include <QThread>
#include <QTimer>
#include <QSpinBox>
#include <QSettings>
#include <QTemporaryDir>
#include <QFileDialog>
#include <QFile>
#include <memory>
#include <cstdio>
using namespace orthoseg;
static int failures=0;
#define CHECK(c,m) do { if(!(c)){ std::printf("FAIL: %s\n",m); ++failures; }else std::printf("ok  : %s\n",m); }while(0)
static bool waitFor(std::function<bool()> fn) {
    QElapsedTimer elapsed;elapsed.start();
    while(!fn() && elapsed.elapsed()<5000){ QApplication::processEvents();QThread::msleep(1); }
    return fn();
}
struct Backend : QTcpServer {
    QJsonObject state;
    int starts=0,promotions=0,reads=0;
    bool failPromotion=false;
    QByteArray casesPath;
    QByteArray promotionPath, selectedModel;
    Backend() {
        const QJsonObject metrics{{"mean_foreground_dice",.92},{"femur_dice",.94},{"tibia_dice",.90}};
        state = {{"backend",QJsonObject{{"status","ok"},{"device","cpu"},{"model_loaded",true}}},
                 {"production",QJsonObject{{"version","production001"},{"architecture","unet"},{"training_case_count",12},{"validation_metrics",metrics}}},
                 {"loaded_model_version","production001"},{"restart_required",false},
                 {"dataset",QJsonObject{{"total_approved_cases",15},{"new_cases_since_last_training",3},{"validation_case_count",4},{"validation_pairs_match",true}}},
                 {"candidate",QJsonObject{{"version","candidate002"},{"valid",true},{"status","completed"},{"parent_model_version","production001"},{"validation_metrics",metrics}}},
                 {"can_start",true},{"training_active",false},
                 {"training_policy",QJsonObject{{"device","cpu"},{"cpu_threads",2},{"epochs",50}}}};
        if(!listen(QHostAddress::LocalHost)) throw std::runtime_error("Cannot bind local management test server");
        connect(this,&QTcpServer::newConnection,this,[this]{
            while(hasPendingConnections()){
                auto* socket=nextPendingConnection();auto buffer=std::make_shared<QByteArray>();
                connect(socket,&QTcpSocket::disconnected,socket,&QObject::deleteLater);
                connect(socket,&QTcpSocket::readyRead,this,[this,socket,buffer]{
                    buffer->append(socket->readAll());int split=buffer->indexOf("\r\n\r\n");
                    if(split<0 || socket->property("done").toBool())return;
                    auto header=buffer->left(split);int length=0;
                    for(auto line:header.split('\n'))if(line.toLower().startsWith("content-length:"))length=line.mid(15).trimmed().toInt();
                    if(buffer->size()<split+4+length)return;
                    socket->setProperty("done",true);
                    auto method=header.split(' ').value(0);auto path=header.split(' ').value(1);
                    int code=200;QJsonObject result;
                    if(method=="GET" && path.startsWith("/management/status")){ ++reads;result=state; selectedModel = path; }
                    else if(method=="GET" && path.startsWith("/management/cases")) {
                        casesPath = path;
                        const bool second = path.contains("offset=50");
                        QJsonArray items;
                        for (int i=0; i<(second ? 1 : 50); ++i)
                            items.append(QJsonObject{{"case_id", QString("case%1").arg(i)},
                                {"filename", second ? "last.png" : "knee.png"},
                                {"labels", QJsonArray{"Femur", "Tibia"}}, {"overlap_pixels", 12},
                                {"status", path.contains("model=medsam2") ? "pending" : "incorporated"}});
                        result = {{"items",items},{"total",51},{"offset",second ? 50 : 0},{"limit",50}};
                    }
                    else if(method=="POST" && path=="/training/start"){
                        ++starts;code=202;result={{"id",QString(32,'a')},{"status","queued"}};
                        state["job"]=result;state["training_active"]=true;state["can_start"]=false;
                    } else if(method=="POST" && path.endsWith("/promote")){
                        ++promotions;promotionPath=path;
                        if(failPromotion){code=500;result={{"detail","Promotion refused"}};}
                        else{
                            auto production=state["production"].toObject();production["version"]="candidate002";state["production"]=production;
                            state["restart_required"]=true;result={{"status","promoted"},{"restart_required",true}};
                        }
                    }else{code=404;result={{"detail","Unexpected operation"}};}
                    auto body=QJsonDocument(result).toJson(QJsonDocument::Compact);
                    socket->write("HTTP/1.1 "+QByteArray::number(code)+" Result\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: "+QByteArray::number(body.size())+"\r\n\r\n"+body);
                    socket->disconnectFromHost();
                });
            }
        });
    }
};
int main(int argc,char** argv){
    std::setbuf(stdout,nullptr);
    QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
    QTemporaryDir settingsDirectory;
    qputenv("XDG_CONFIG_HOME", settingsDirectory.path().toUtf8());
    QApplication app(argc,argv);
    if (app.arguments().contains("--live") || app.arguments().contains("--live-medsam")) {
        qputenv("ORTHOSEG_ENABLE_MODEL_MANAGEMENT", "1");
        MainWindow window; window.show();
        auto* action=window.findChild<QAction*>("modelManagementAction");
        auto* managementButton=window.findChild<QPushButton*>("modelManagementButton");
        CHECK(managementButton && waitFor([&]{return managementButton->isEnabled();}), "live management permission available");
        if (!managementButton || !managementButton->isEnabled()) return 1;
        action->trigger();
        auto* panel=dynamic_cast<ModelManagementDialog*>(window.findChild<QDialog*>("modelManagementDialog"));
        if (app.arguments().contains("--live-medsam"))
            panel->findChild<QComboBox*>("managementModel")->setCurrentIndex(1);
        auto* train=panel->findChild<QPushButton*>("managementTrain");
        auto* promote=panel->findChild<QPushButton*>("managementPromote");
        auto* refresh=panel->findChild<QPushButton*>("managementRefresh");
        CHECK(waitFor([&]{return train->isEnabled();}), "live management status available");
        int errors=0;
        QTimer confirmations;
        QObject::connect(&confirmations,&QTimer::timeout,&app,[&]{
            if(auto* box=qobject_cast<QMessageBox*>(QApplication::activeModalWidget())) {
                if(box->property("handled").toBool())return;box->setProperty("handled",true);
                if(box->standardButtons().testFlag(QMessageBox::Yes))box->button(QMessageBox::Yes)->click();
                else { if(box->icon()==QMessageBox::Warning)++errors;box->accept(); }
            }
        }); confirmations.start(5);
        if(!train->isEnabled())return 1;
        train->click();
        QElapsedTimer elapsed;elapsed.start();
        while(!promote->isEnabled() && elapsed.elapsed()<600000 && !errors){
            QApplication::processEvents();QThread::msleep(10);
        }
        CHECK(promote->isEnabled() && !errors, "live manual training completes with candidate");
        if(!promote->isEnabled())return 1;
        const auto status=panel->findChild<QLabel*>("managementJob")->text();
        CHECK(status.contains("completed"), "live completed job shown");
        promote->click();
        CHECK(waitFor([&]{return refresh->isEnabled();}) && !errors, "live promotion succeeded");
        const auto production=panel->findChild<QLabel*>("managementProduction")->text();
        CHECK(production.contains("restart backend required"), "live disk/loaded version mismatch explained");
        panel->grab().save(app.arguments().value(2, "/tmp/orthoseg-management.png"));
        return failures?1:0;
    }
    qunsetenv("ORTHOSEG_ENABLE_MODEL_MANAGEMENT");
    qputenv("MONAI_BACKEND_URL", "http://127.0.0.1:9");
    MainWindow normal;
    CHECK(!normal.findChild<QAction*>("modelManagementAction"),"normal annotator has no management controls");
    auto* normalButton=normal.findChild<QPushButton*>("modelManagementButton");
    CHECK(normalButton && !normalButton->isEnabled() && normalButton->isHidden(),"management button starts hidden without backend privilege");
    Backend server;
    qputenv("MONAI_BACKEND_URL",QString("http://127.0.0.1:%1").arg(server.serverPort()).toUtf8());
    qputenv("ORTHOSEG_ENABLE_MODEL_MANAGEMENT","1");
    MainWindow admin;admin.show();
    auto* action=admin.findChild<QAction*>("modelManagementAction");
    CHECK(action,"admin Tools action exists");if(!action)return 1;
    auto* managementButton=admin.findChild<QPushButton*>("modelManagementButton");
    CHECK(managementButton && waitFor([&]{return managementButton->isEnabled();}),
          "management button enabled by backend permission");
    CHECK(managementButton->isVisible(), "authorized management button is visible in top toolbar");
    auto* toolbarBatch = admin.findChild<QPushButton*>("batchModeButton");
    CHECK(toolbarBatch && toolbarBatch->parentWidget() == managementButton->parentWidget() &&
          toolbarBatch->geometry().right() < managementButton->geometry().left(),
          "management button follows Batch Mode");
    action->trigger();
    auto* panel=dynamic_cast<ModelManagementDialog*>(admin.findChild<QDialog*>("modelManagementDialog"));
    CHECK(panel && !panel->isModal(),"management dialog is separate and modeless");if(!panel)return 1;
    auto* train=panel->findChild<QPushButton*>("managementTrain");
    auto* promote=panel->findChild<QPushButton*>("managementPromote");
    auto* refresh=panel->findChild<QPushButton*>("managementRefresh");
    auto label=[panel](const char* name){return panel->findChild<QLabel*>(name)->text();};
    CHECK(waitFor([&]{return train->isEnabled();}),"status loaded asynchronously");
    CHECK(server.starts==0 && server.promotions==0,"opening management never trains or promotes");
    auto* model = panel->findChild<QComboBox*>("managementModel");
    CHECK(model && model->count() == 2, "management offers UNet and MedSAM2 in the existing dialog");
    model->setCurrentIndex(1);
    CHECK(waitFor([&]{return train->isEnabled();}) && server.selectedModel.contains("model=medsam2"),
          "MedSAM2 selection requests its own lifecycle metadata");
    model->setCurrentIndex(0);
    CHECK(waitFor([&]{return train->isEnabled();}), "UNet management remains available");
    auto* reminder = panel->findChild<QSpinBox*>("fineTuneReminderCases");
    CHECK(reminder && reminder->value() == 0, "fine-tuning reminders are opt-in");
    reminder->setValue(3);
    model->setCurrentIndex(1);
    CHECK(waitFor([&]{return train->isEnabled();}) && reminder->value() == 0,
          "reminder thresholds are separate for each model family");
    model->setCurrentIndex(0);
    CHECK(waitFor([&]{return train->isEnabled();}) && reminder->value() == 3,
          "user-selected corrected-case threshold is persisted");
    int reminders = 0;
    QTimer reminderDialogs;
    QObject::connect(&reminderDialogs, &QTimer::timeout, &app, [&] {
        auto* box = qobject_cast<QMessageBox*>(QApplication::activeModalWidget());
        if (box && box->windowTitle() == "Fine-tuning Reminder") {
            ++reminders;
            CHECK(box->text().contains("3 new or revised cases"), "reminder uses backend corrected-case count");
            box->button(QMessageBox::No)->click();
        }
    });
    reminderDialogs.start(5);
    QMetaObject::invokeMethod(&admin, "offerFineTuningReminder", Qt::DirectConnection);
    CHECK(waitFor([&]{return reminders == 1;}) && server.starts == 0,
          "threshold prompts the user without starting fine-tuning");
    int readsBefore = server.reads;
    QMetaObject::invokeMethod(&admin, "offerFineTuningReminder", Qt::DirectConnection);
    waitFor([&]{return server.reads > readsBefore;});
    QApplication::processEvents();
    CHECK(reminders == 1 && server.starts == 0, "same threshold milestone does not repeatedly prompt");
    reminder->setValue(4);
    readsBefore = server.reads;
    QMetaObject::invokeMethod(&admin, "offerFineTuningReminder", Qt::DirectConnection);
    waitFor([&]{return server.reads > readsBefore;});
    QApplication::processEvents();
    CHECK(reminders == 1, "below-threshold cases do not trigger reminders");
    reminder->setValue(0);
    reminderDialogs.stop();
    auto* files = panel->findChild<QTableWidget*>("managementCases");
    auto* nextFiles = panel->findChild<QPushButton*>("nextTrainingCases");
    auto* previousFiles = panel->findChild<QPushButton*>("previousTrainingCases");
    CHECK(files && waitFor([&]{return files->rowCount() == 50 && nextFiles->isEnabled();}), "submitted files load in bounded pages");
    CHECK(files->item(0,0)->text() == "knee.png" && files->item(0,1)->text() == "Femur, Tibia" &&
          files->item(0,2)->text() == "12" && files->item(0,3)->text() == "Incorporated",
          "training table shows filename, labels, overlap and per-model status");
    nextFiles->click();
    CHECK(waitFor([&]{return files->rowCount() == 1 && previousFiles->isEnabled();}) && files->item(0,0)->text() == "last.png",
          "training table can navigate to later uploaded files");
    CHECK(!nextFiles->isEnabled(), "last training page disables Next");
    auto* exportList = panel->findChild<QPushButton*>("exportTrainingCases");
    QTemporaryDir exported;
    const auto csvPath = exported.filePath("training.csv");
    QTimer exportDialog;
    QObject::connect(&exportDialog, &QTimer::timeout, &app, [&] {
        if (auto* dialog = qobject_cast<QFileDialog*>(QApplication::activeModalWidget())) {
            dialog->selectFile(csvPath);
            QMetaObject::invokeMethod(dialog, "accept", Qt::DirectConnection);
        }
    });
    exportDialog.start(5);
    CHECK(exportList && exportList->isEnabled(), "file list export available");
    exportList->click();
    CHECK(waitFor([&]{return QFile::exists(csvPath) && exportList->isEnabled();}), "CSV export completes");
    exportDialog.stop();
    QFile csv(csvPath); csv.open(QIODevice::ReadOnly);
    const auto contents = csv.readAll();
    CHECK(contents.count('\n') == 52 && contents.contains("last.png") && contents.contains("knee.png"),
          "export from last table page includes all 51 files plus header");
    CHECK(contents.contains("\"Femur, Tibia\"") && contents.contains("\"Incorporated\",\"unet\""),
          "CSV quotes multi-label cells and includes selected model training status");
    model->setCurrentIndex(1);
    CHECK(waitFor([&]{return files->rowCount() == 50 && nextFiles->isEnabled();}) &&
          server.casesPath.contains("model=medsam2") && files->item(0,3)->text() == "Pending",
          "switching model resets the file page and retrieves its incorporation state");
    model->setCurrentIndex(0);
    CHECK(waitFor([&]{return train->isEnabled() && files->rowCount() == 50 && !previousFiles->isEnabled();}), "UNet table returns to its first page");
    CHECK(label("managementProduction").contains("production001"),"production metadata displayed");
    CHECK(label("managementDataset").contains("15") && label("managementDataset").contains("3"),"backend dataset counts displayed");
    auto* metrics=panel->findChild<QTableWidget*>("managementMetrics");
    CHECK(metrics->item(0,2)->text()=="0.9200","candidate Dice comparison displayed");
    bool approve=false;int confirmations=0,notices=0;
    QTimer dialogs;
    QObject::connect(&dialogs,&QTimer::timeout,&app,[&]{
        if(auto* box=qobject_cast<QMessageBox*>(QApplication::activeModalWidget())){
            if(box->property("handled").toBool())return;box->setProperty("handled",true);
            if(box->standardButtons().testFlag(QMessageBox::Cancel)){
                ++confirmations;box->button(approve?QMessageBox::Yes:QMessageBox::Cancel)->click();
            }else{++notices;box->accept();}
        }
    });dialogs.start(5);
    train->click();CHECK(confirmations==1 && server.starts==0,"training requires explicit confirmation");
    approve=true;train->click();train->click();
    CHECK(waitFor([&]{return server.starts==1 && refresh->isEnabled();}),"manual action starts exactly one job");
    CHECK(!train->isEnabled() && !promote->isEnabled(),"duplicate training and promotion disabled during job");
    bool responsive=false;QTimer::singleShot(0,&app,[&]{responsive=true;});
    CHECK(waitFor([&]{return responsive;}),"UI event loop remains responsive while training");
    server.state["training_active"]=false;server.state["can_start"]=true;
    server.state["job"]=QJsonObject{{"status","failed"},{"error","Dataset validation failed"}};
    refresh->click();CHECK(waitFor([&]{return train->isEnabled();}),"failed job permits retry");
    CHECK(label("managementJob").contains("Dataset validation failed"),"training error shown without changing production");
    auto candidate=server.state["candidate"].toObject();candidate["valid"]=false;server.state["candidate"]=candidate;
    refresh->click();waitFor([&]{return refresh->isEnabled();});CHECK(!promote->isEnabled(),"invalid candidate disables promotion");
    server.state["candidate"]=QJsonValue();refresh->click();waitFor([&]{return refresh->isEnabled();});CHECK(!promote->isEnabled(),"missing candidate disables promotion");
    candidate["valid"]=true;server.state["candidate"]=candidate;
    refresh->click();waitFor([&]{return promote->isEnabled();});
    approve=false;promote->click();CHECK(server.promotions==0,"promotion cancel sends no request");
    approve=true;server.failPromotion=true;int previousNotices=notices;promote->click();
    CHECK(waitFor([&]{return notices>previousNotices && promote->isEnabled();}),"promotion failure surfaced");
    CHECK(label("managementProduction").contains("production001"),"failed promotion retains production display");
    server.failPromotion=false;previousNotices=notices;promote->click();
    CHECK(waitFor([&]{return notices>previousNotices && refresh->isEnabled();}),"successful promotion refreshes metadata");
    CHECK(server.promotionPath=="/models/candidate002/promote","explicit candidate version sent");
    CHECK(label("managementProduction").contains("Production on disk: candidate002") && label("managementProduction").contains("Loaded for inference: production001") && label("managementProduction").contains("restart"),"disk and loaded versions distinguished");
    server.close();refresh->click();
    CHECK(waitFor([&]{return refresh->isEnabled();}) && !train->isEnabled() && !promote->isEnabled(),"offline backend safely disables management actions");
    CHECK(label("managementBackend").contains("unavailable"),"offline status explained");
    std::printf("%d management test failures\n",failures);return failures?1:0;
}
