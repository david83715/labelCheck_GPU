#pragma once
#include <opencv2/opencv.hpp>

#include <QMainWindow>
#include <QPushButton>
#include <QComboBox>
#include <QLabel>
#include <QProgressBar>
#include <QLineEdit>
#include <QThread>
#include <QDir>
#include <thread>
#include <atomic>
#include <QStatusBar>

#include "darknet.hpp"

// 前置宣告
class labelCheck;

// Worker 類的宣告
class Worker : public QObject
{
    Q_OBJECT

private:
    QString outputDir;
    Darknet::VStr imagePaths;
    labelCheck *parent; // 用於調用主類的方法

public:
    Worker(const QString &outputDir, Darknet::VStr imagePaths, labelCheck *parent);
    ~Worker() = default;

public slots:
    void runDarknetDetection();

signals:
    void updateProgress(int value);
    void detectionError();
    void finished();
};

// labelCheck 類
class labelCheck : public QMainWindow
{
    Q_OBJECT

private:
    QLabel *instructLabel = nullptr;
    QComboBox *rollNumComboBox = nullptr;
    QPushButton *refreshButton = nullptr;
    QPushButton *getDirButton = nullptr;
    QLineEdit *outputDirLineEdit = nullptr;
    QPushButton *checkButton = nullptr;
    QProgressBar *progbar = nullptr;

    QThread *thread = nullptr;
    Worker *worker = nullptr;

    QStatusBar *statusBar;

public:
    QDir inputDir;
    QString outputDir;

    labelCheck(QWidget *parent = nullptr);
    void getOutputDir();
    void updateRollNum();
    void startDarknetDetection();
    std::vector<cv::Mat> splitImageWithOverlap(const cv::Mat &img, int patch_width, int patch_height);
    bool copyFile(const std::string &source, const std::string &destination);
    std::vector<std::string> getImagePaths(const std::string &directory);

    void resetCheckButton();

    ~labelCheck();

private slots:
    void updateProgress(int value);
    void updateStatusMessage(const QString &message);
};