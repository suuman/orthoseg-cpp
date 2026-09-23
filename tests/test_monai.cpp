#include "MonaiClient.h"
#include "MainWindow.h"
#include <QApplication>
#include <QElapsedTimer>
#include <QFile>
#include <QFileDialog>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QTcpServer>
#include <QTcpSocket>
#include <QTemporaryDir>
#include <QThread>
#include <QTimer>
#include <QStatusBar>
#include <opencv2/imgcodecs.hpp>
#include <cstdio>
#include <memory>

using namespace orthoseg;
static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("FAIL: %s\n", m); ++failures; } else std::printf("ok  : %s\n", m); } while(0)
static bool waitFor(const std::function<bool()>& predicate, int timeout = 5000) {
    QElapsedTimer clock; clock.start();
    while (!predicate() && clock.elapsed() < timeout) {
        QApplication::processEvents(); QThread::msleep(1);
    }
    return predicate();
}
static QByteArray read(const QString& path) { QFile f(path); if (!f.open(QIODevice::ReadOnly)) return {}; return f.readAll(); }
static bool same(const cv::Mat& a, const cv::Mat& b) { return a.size() == b.size() && a.type() == b.type() && cv::norm(a,b,cv::NORM_INF) == 0; }
static QByteArray png(const cv::Mat& mat) {
    std::vector<uchar> bytes; cv::imencode(".png", mat, bytes);
    return QByteArray(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}
static void rejects(const std::function<void()>& fn, const char* message) {
    bool rejected = false;
    try { fn(); } catch (const std::exception&) { rejected = true; }
    CHECK(rejected, message);
}

struct Server : QTcpServer {
    QByteArray response, contentType = "image/png", lastBody, lastPath;
    int status = 200, requests = 0, uploads = 0;
    bool stall = false;
    Server() {
        if (!listen(QHostAddress::LocalHost)) throw std::runtime_error("Cannot bind test server");
        connect(this, &QTcpServer::newConnection, this, [this] {
            while (hasPendingConnections()) {
                auto* socket = nextPendingConnection();
                auto buffer = std::make_shared<QByteArray>();
                connect(socket, &QTcpSocket::disconnected, socket, &QObject::deleteLater);
                connect(socket, &QTcpSocket::readyRead, this, [this, socket, buffer] {
                    buffer->append(socket->readAll());
                    int split = buffer->indexOf("\r\n\r\n");
                    if (split < 0 || socket->property("handled").toBool()) return;
                    const auto header = buffer->left(split);
                    int length = 0;
                    for (auto line : header.split('\n'))
                        if (line.toLower().startsWith("content-length:")) length = line.mid(15).trimmed().toInt();
                    if (buffer->size() < split + 4 + length) return;
                    socket->setProperty("handled", true);
                    ++requests;
                    lastBody = buffer->mid(split+4, length);
                    lastPath = header.split(' ').value(1);
                    if (lastPath == "/training/cases") ++uploads;
                    if (stall) return;
                    QByteArray body = response, type = contentType;
                    if (lastPath == "/health" && status == 200) { body = "{\"status\":\"ok\",\"model_loaded\":true}"; type = "application/json"; }
                    if (lastPath == "/training/cases" && status == 200) { body = "{\"status\":\"accepted\",\"new_cases_since_last_training\":1}"; type = "application/json"; }
                    socket->write("HTTP/1.1 " + QByteArray::number(status) + " Result\r\nContent-Type: " + type +
                                  "\r\nX-Model-Version: test_v1\r\nConnection: close\r\nContent-Length: " + QByteArray::number(body.size()) + "\r\n\r\n" + body);
                    socket->disconnectFromHost();
                });
            }
        });
    }
    QUrl url() const { return QUrl(QString("http://127.0.0.1:%1").arg(serverPort())); }
    QByteArray field(const QByteArray& name) const {
        const auto start = lastBody.indexOf("name=\"" + name + "\"");
        const auto data = lastBody.indexOf("\r\n\r\n", start);
        const auto end = lastBody.indexOf("\r\n--", data+4);
        return start >= 0 && data >= 0 && end >= 0 ? lastBody.mid(data+4,end-data-4) : QByteArray();
    }
};

// Drive real dialogs using the application's existing test conventions.
struct Dialogs : QObject {
    QTimer timer;
    QString file;
    bool add = false, replace = true;
    int trainingPrompts = 0, errors = 0;
    Dialogs() {
        connect(&timer, &QTimer::timeout, this, [this] {
            auto* modal = QApplication::activeModalWidget();
            if (auto* dialog = qobject_cast<QFileDialog*>(modal)) {
                if (!file.isEmpty()) {
                    if (auto* edit = dialog->findChild<QLineEdit*>("fileNameEdit")) edit->setText(file);
                    QMetaObject::invokeMethod(dialog, "accept", Qt::DirectConnection);
                }
            } else if (auto* box = qobject_cast<QMessageBox*>(modal)) {
                if (box->property("handled").toBool()) return;
                box->setProperty("handled", true);
                if (box->windowTitle() == "Add to AI Training") {
                    ++trainingPrompts;
                    for (auto* b : box->buttons()) if (b->text() == (add ? "Add to AI Training" : "Save Only")) { b->click(); break; }
                } else if (box->standardButtons().testFlag(QMessageBox::Cancel)) {
                    box->button(replace ? QMessageBox::Yes : QMessageBox::Cancel)->click();
                } else { ++errors; box->accept(); }
            }
        });
        timer.start(5);
    }
};

int main(int argc, char** argv) {
    std::setbuf(stdout, nullptr);
    QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
    QApplication app(argc, argv);
    if (app.arguments().contains("--live")) {
        const auto args = app.arguments();
        if (args.size() != 4) return 2;
        MainWindow live;
        live.show();
        Dialogs dialogs;
        dialogs.file = args[2];
        QMetaObject::invokeMethod(&live, "onUpload", Qt::DirectConnection);
        auto* button = live.findChild<QPushButton*>("monaiSegment");
        button->click();
        CHECK(waitFor([&]{ return button->isEnabled(); }, 180000) && dialogs.errors == 0, "live MONAI prediction applied");
        auto* doc = live.document();
        CHECK(doc->hasImage() && doc->canUndo(), "live prediction is editable and undoable");
        if (dialogs.errors || !doc->hasImage()) return 1;
        doc->paintLine({4,4}, {4,4}, Label::Background, 2);
        doc->paintLine({15,15}, {15,15}, Label::Tibia, 2);
        // Exercise the unchanged AI Fill apply path, without requiring GPU weights.
        doc->aiFill().resultMask = cv::Mat::zeros(doc->mask().size(), CV_8UC1);
        doc->aiFill().resultMask.at<uchar>(20,20) = static_cast<uchar>(Label::Femur);
        doc->applyAIResult();
        QFile expected(args[3] + ".canonical.png");
        CHECK(expected.open(QIODevice::WriteOnly), "write expected current mask for backend verification");
        expected.write(encodeMonaiMask(toMonaiLabels(doc->mask(), {0,1,2})));
        expected.close();
        dialogs.add = true;
        dialogs.file = args[3];
        QMetaObject::invokeMethod(&live, "onExport", Qt::DirectConnection);
        CHECK(waitFor([&]{return live.statusBar()->currentMessage().contains("added") || dialogs.errors;}, 30000) && !dialogs.errors,
              "live corrected training case accepted");
        CHECK(QFile::exists(args[3]), "existing color export saved");
        return failures ? 1 : 0;
    }
    QTemporaryDir temp;
    cv::Mat source(24, 37, CV_16UC1, cv::Scalar(45000));
    const auto input = temp.filePath("original.png");
    cv::imwrite(input.toStdString(), source);
    auto original = read(input);
    cv::Mat canonical = cv::Mat::zeros(source.size(), CV_8UC1);
    canonical(cv::Rect(3, 3, 9, 10)).setTo(1);
    canonical(cv::Rect(22, 5, 9, 10)).setTo(2);
    CHECK(same(decodeMonaiMask(png(canonical), source.size()), canonical), "valid canonical PNG decoded");
    rejects([&]{ decodeMonaiMask(png(canonical), {2,2}); }, "dimension mismatch rejected");
    rejects([&]{ decodeMonaiMask("bad", source.size()); }, "invalid PNG rejected");
    rejects([&]{ decodeMonaiMask(png(cv::Mat(source.size(), CV_8UC1, cv::Scalar(3))), source.size()); }, "invalid class rejected");
    rejects([&]{ decodeMonaiMask(png(cv::Mat(source.size(), CV_8UC3, cv::Scalar(1,1,1))), source.size()); }, "RGB labels rejected");
    rejects([&]{ decodeMonaiMask(png(source), source.size()); }, "16-bit labels rejected");
    auto mapped = fromMonaiLabels(canonical, {0,5,8});
    CHECK(mapped.at<uchar>(4,4) == 5 && mapped.at<uchar>(6,23) == 8, "noncanonical editor label IDs mapped semantically");
    CHECK(same(toMonaiLabels(mapped, {0,5,8}), canonical), "reverse mapping uses current editor labels");
    rejects([&]{ fromMonaiLabels(canonical, {0,-1,8}); }, "missing Femur rejected");
    rejects([&]{ fromMonaiLabels(canonical, {0,5,-1}); }, "missing Tibia rejected");

    Server server;
    server.response = png(canonical);
    MonaiClient client(nullptr, server.url(), 150);
    bool done = false; QString error;
    client.health([&](auto json, auto err) { error=err; done=true; CHECK(json.value("model_loaded").toBool(), "health success"); });
    CHECK(waitFor([&]{return done;} ) && error.isEmpty(), "health callback completes");
    done=false;
    client.segment(original, source.size(), [&](auto mask, auto version, auto err) { error=err; done=true; CHECK(same(mask,canonical), "HTTP segmentation decoded"); CHECK(version=="test_v1", "model version header retained"); });
    CHECK(!client.segment(original, source.size(), {}), "duplicate inference rejected");
    CHECK(waitFor([&]{return done;}) && error.isEmpty(), "inference asynchronous completion");
    CHECK(server.field("image") == original && server.field("case_id") == monaiCaseId(original), "original 16-bit bytes and stable hash sent");
    canonical.at<uchar>(0,0)=2;
    done=false;
    client.submitTrainingCase(original,canonical,"original.png","test_v1",[&](auto,auto err){error=err;done=true;});
    CHECK(waitFor([&]{return done;}) && error.isEmpty(), "training upload accepted");
    CHECK(server.field("image")==original && same(decodeMonaiMask(server.field("mask"),source.size()),canonical), "training uploads current corrected mask with original bytes");
    CHECK(server.field("model_version")=="test_v1", "training includes model traceability");
    server.status=503;server.response="{\"detail\":\"No production model\"}";
    done=false;client.segment(original,source.size(),[&](auto,auto,auto err){error=err;done=true;});
    CHECK(waitFor([&]{return done;}) && error.contains("No production model"), "HTTP errors surfaced");
    server.status=200;server.stall=true;
    done=false;client.segment(original,source.size(),[&](auto,auto,auto err){error=err;done=true;});
    CHECK(waitFor([&]{return done;}) && !error.isEmpty() && !client.segmentRunning(), "timeout restores client state");
    server.stall=false;
    auto offlineUrl=server.url();server.close();
    MonaiClient offline(nullptr,offlineUrl,150);
    done=false;offline.health([&](auto,auto err){error=err;done=true;});
    CHECK(waitFor([&]{return done;}) && error.contains("unavailable"), "backend offline handled");
    CHECK(server.listen(QHostAddress::LocalHost), "restart local test server");
    server.response=png(canonical);
    qputenv("MONAI_BACKEND_URL", server.url().toString().toUtf8());

    MainWindow window;
    window.show();
    Dialogs dialogs;
    dialogs.file=input;
    QMetaObject::invokeMethod(&window,"onUpload",Qt::DirectConnection);
    CHECK(window.document()->originalPng().size()==size_t(original.size()), "document retains PNG bytes at load");
    QFile::remove(input); // Requests must not reopen or depend on the source file.
    auto* button=window.findChild<QPushButton*>("monaiSegment");
    CHECK(button!=nullptr,"AI Segment action exists");
    auto* doc=window.document();
    doc->paintLine({20,20},{20,20},Label::Fibula,2);
    auto before=doc->mask().clone();
    button->click();
    CHECK(!button->isEnabled(),"AI Segment duplicate action disabled");
    CHECK(waitFor([&]{return button->isEnabled();}),"UI inference finishes");
    CHECK(doc->mask().at<uchar>(4,4)==1 && doc->mask().at<uchar>(6,23)==2,"MONAI applied to editable document");
    CHECK(doc->mask().at<uchar>(20,20)==3,"unrelated Fibula preserved");
    doc->undo();CHECK(same(doc->mask(),before),"prediction is one undo operation");
    button->click();waitFor([&]{return button->isEnabled();});
    doc->paintLine({4,4},{4,4},Label::Background,2);
    doc->paintLine({15,15},{15,15},Label::Tibia,2);
    auto corrected=toMonaiLabels(doc->mask(),{0,1,2});
    int uploads=server.uploads;
    dialogs.file=temp.filePath("save-only.png");dialogs.add=false;
    QMetaObject::invokeMethod(&window,"onExport",Qt::DirectConnection);
    CHECK(QFile::exists(dialogs.file) && server.uploads==uploads,"Save Only preserves export and sends nothing");
    int prompts=dialogs.trainingPrompts;
    dialogs.file=temp.filePath("same.png");
    QMetaObject::invokeMethod(&window,"onExport",Qt::DirectConnection);
    CHECK(dialogs.trainingPrompts==prompts,"unchanged declined annotation not prompted again");
    doc->paintLine({16,16},{16,16},Label::Femur,2);
    corrected=toMonaiLabels(doc->mask(),{0,1,2});
    dialogs.file=temp.filePath("corrected.png");dialogs.add=true;
    QMetaObject::invokeMethod(&window,"onExport",Qt::DirectConnection);
    CHECK(waitFor([&]{return server.uploads==uploads+1;}),"revised annotation can be submitted");
    CHECK(server.field("image")==original,"UI sends retained original PNG even after source file deletion");
    CHECK(same(decodeMonaiMask(server.field("mask"),source.size()),corrected),"UI training mask includes manual corrections, not initial prediction");
    waitFor([&]{return window.statusBar()->currentMessage().contains("added");});
    CHECK(server.field("model_version")=="test_v1","UI preserves model version through edits");
    prompts=dialogs.trainingPrompts;dialogs.file=temp.filePath("unchanged.png");
    QMetaObject::invokeMethod(&window,"onExport",Qt::DirectConnection);
    CHECK(dialogs.trainingPrompts==prompts,"unchanged submitted annotation not prompted again");
    doc->paintLine({17,17},{17,17},Label::Tibia,2);
    server.status=500;server.response="{\"detail\":\"disk failure\"}";
    dialogs.file=temp.filePath("upload-failure.png");int errors=dialogs.errors;
    QMetaObject::invokeMethod(&window,"onExport",Qt::DirectConnection);
    CHECK(waitFor([&]{return dialogs.errors>errors;}) && QFile::exists(dialogs.file),"failed upload leaves successful export intact");
    server.status=200;server.response=png(canonical);
    dialogs.file=temp.filePath("retry.png");uploads=server.uploads;
    QMetaObject::invokeMethod(&window,"onExport",Qt::DirectConnection);
    CHECK(waitFor([&]{return server.uploads>uploads;}),"failed upload can retry on next export");
    waitFor([&]{return window.statusBar()->currentMessage().contains("added");});
    dialogs.file=temp.filePath("bad.unsupported");
    prompts=dialogs.trainingPrompts; uploads=server.uploads;
    doc->paintLine({10,20},{10,20},Label::Femur,2);
    QMetaObject::invokeMethod(&window,"onExport",Qt::DirectConnection);
    CHECK(dialogs.trainingPrompts==prompts && server.uploads==uploads, "failed export never prompts or uploads");
    before=doc->mask().clone();
    server.response=png(cv::Mat(source.size(),CV_8UC1,cv::Scalar(255)));
    errors=dialogs.errors;button->click();
    CHECK(waitFor([&]{return dialogs.errors>errors;}) && same(doc->mask(),before),"invalid inference leaves existing annotation unchanged");
    server.response=png(canonical);
    button->click();doc->paintLine({30,20},{30,20},Label::Femur,2);before=doc->mask().clone();
    errors=dialogs.errors;
    CHECK(waitFor([&]{return dialogs.errors>errors;}) && same(doc->mask(),before),"edits during inference cause stale result to be discarded");
    std::printf("%d MONAI test failures\n",failures);
    return failures?1:0;
}
