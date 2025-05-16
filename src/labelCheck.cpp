#include "labelCheck.h"
#include "darknet.hpp"
#include "darknet_image.hpp"
#include <iostream>
#include <filesystem>
#include <chrono>

#include <QApplication>
#include <QTimer>
#include <QGridLayout>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QSettings>
#include <QStandardPaths>
#include <QMessageBox>

// 定義單張圖片的處理函數
void processImage(Darknet::NetworkPtr net, const std::string &imagePath, const std::string &outputDir, labelCheck *parent, std::atomic<int> &completedImages, int totalImages, Worker *worker)
{
    cv::Mat image = cv::imread(imagePath);
    if (image.empty())
    {
        qDebug() << "Failed to load " << imagePath;
        return;
    }

    std::vector<cv::Mat> images = parent->splitImageWithOverlap(image, 500, 500);
    if (images.empty())
    {
        qDebug() << "No valid patches for " << imagePath;
        return;
    }

    bool to_copy = false;
    for (cv::Mat &img : images)
    {
        Darknet::Predictions predictions = Darknet::predict(net, img);
        if (predictions.empty())
            continue;

        for (const auto &prediction : predictions)
        {
            auto it = prediction.prob.find(prediction.best_class);
            float confidence = (it != prediction.prob.end()) ? it->second : -1.0f;
            if (confidence > 0.8f)
            {
                qDebug() << confidence;
                to_copy = true;
                size_t pos = imagePath.find_last_of("\\");
                std::string filename = imagePath.substr(pos + 1);
                pos = filename.find_last_of(".");
                filename = filename.replace(pos, 4, std::format("_{:.3f}.jpg", confidence));
                cv::imwrite(outputDir + '\\' + filename, img);
                break;
            }
        }
        if (to_copy)
            break;
    }

    if (to_copy)
    {
        size_t pos = imagePath.find_last_of("\\");
        std::string filename = imagePath.substr(pos + 1);
        parent->copyFile(imagePath, outputDir + '\\' + filename);
    }

    // 更新進度
    int progress = static_cast<int>(++completedImages * 100 / totalImages);
    emit worker->updateProgress(progress);
}

// Worker 的實現
Worker::Worker(const QString &outputDir, Darknet::VStr imagePaths, labelCheck *parent)
    : outputDir(outputDir), imagePaths(std::move(imagePaths)), parent(parent) {}

void Worker::runDarknetDetection()
{
    auto start = std::chrono::high_resolution_clock::now();

    try
    {
        // 設定模型與權重檔案路徑
        std::string config_file = "model\\yolov4-tiny-class_3i.cfg";
        std::string weights_file = "model\\backup\\yolov4-tiny-class_3i_last.weights";
        std::string names_file = "model\\class.names";

        // 決定線程數（通常為 CPU 核心數）
        // const int numThreads = std::thread::hardware_concurrency();
        const int numThreads = std::min(static_cast<int>(std::thread::hardware_concurrency()), 4);       // 限制線程數量
        const int imagesPerThread = (static_cast<int>(imagePaths.size()) + numThreads - 1) / numThreads; // 平均分配圖片

        std::vector<std::thread> threads;
        std::atomic<int> completedImages(0); // 原子計數器，用於進度更新

        for (int t = 0; t < numThreads; ++t)
        {
            // 為每個線程創建獨立的 Darknet 模型
            try
            {
                Darknet::NetworkPtr net = Darknet::load_neural_network(config_file, names_file, weights_file);
                // 計算該線程負責的圖片範圍
                size_t startIdx = t * imagesPerThread;
                size_t endIdx = std::min(startIdx + imagesPerThread, imagePaths.size());

                threads.emplace_back([net, startIdx, endIdx, this, &completedImages]() mutable
                                     {
                                         for (size_t i = startIdx; i < endIdx; ++i)
                                             processImage(net, this->imagePaths[i], outputDir.toStdString(), parent, completedImages, static_cast<int>(this->imagePaths.size()), this);
                                         Darknet::free_neural_network(net); // 釋放模型
                                     });
            }
            catch (const std::exception &e)
            {
                qDebug() << "Error: " << QString::fromLocal8Bit(e.what());
                qDebug() << "Failed to load Darknet network for thread " << t;
                emit detectionError();
                emit finished();
                return;
            }
        }

        // 等待所有線程完成
        for (auto &thread : threads)
        {
            thread.join();
        }

        emit updateProgress(100);
        qDebug() << "Done.\n";
        parent->resetCheckButton();
    }
    catch (const std::exception &e)
    {
        qDebug() << "Error: " << QString::fromLocal8Bit(e.what());
    }

    // 計算耗時
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    qDebug() << "Time cost: " << duration.count() / 60000. << " min";

    emit finished();
}

// labelCheck 的實現
labelCheck::labelCheck(QWidget *parent)
    : QMainWindow(parent)
{
    QSettings settings("settings.ini", QSettings::IniFormat);

    if (!settings.value("input/directory").isValid() || settings.value("input/directory").toString() == "")
    {
        settings.setValue("input/directory", "");
        QMessageBox::information(this,                                          // 父物件，通常是 this
                                 "Not configured",                              // 標題
                                 "Please set the roll image directory first!"); // 內容
        QTimer::singleShot(0, qApp, &QCoreApplication::quit);
    }
    inputDir = QDir(settings.value("input/directory").toString().replace("/", "\\"));
    // inputDir.setFilter(QDir::Dirs | QDir::NoDotAndDotDot);

    resize(settings.value("window/size", QSize(600, 150)).toSize());
    move(settings.value("window/position", QPoint(100, 100)).toPoint());

    instructLabel = new QLabel("Select roll number to check:", this);
    instructLabel->setAlignment(Qt::AlignCenter);

    rollNumComboBox = new QComboBox(this);
    rollNumComboBox->addItems(inputDir.entryList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Reversed));

    refreshButton = new QPushButton("Refresh", this);
    connect(refreshButton, &QPushButton::clicked, this, &labelCheck::updateRollNum);

    getDirButton = new QPushButton("Search", this);
    connect(getDirButton, &QPushButton::clicked, this, &labelCheck::getOutputDir);

    QString desktopPath = QStandardPaths::writableLocation(QStandardPaths::DesktopLocation).append("/roll_images_with_label").replace("/", "\\");
    outputDir = settings.value("output/directory", desktopPath).toString();

    outputDirLineEdit = new QLineEdit(this);
    outputDirLineEdit->setReadOnly(true);
    outputDirLineEdit->setText(outputDir);

    checkButton = new QPushButton("Check", this);
    connect(checkButton, &QPushButton::clicked, this, &labelCheck::startDarknetDetection);

    progbar = new QProgressBar(this);
    progbar->setRange(0, 100);

    QHBoxLayout *rollNumHLayout = new QHBoxLayout;
    rollNumHLayout->addWidget(rollNumComboBox);
    rollNumHLayout->addWidget(refreshButton);
    rollNumHLayout->setStretch(0, 1);

    QHBoxLayout *getDirHLayout = new QHBoxLayout;
    getDirHLayout->addWidget(outputDirLineEdit);
    getDirHLayout->addWidget(getDirButton);
    getDirHLayout->setStretch(0, 1);

    QFormLayout *formLayout = new QFormLayout;
    formLayout->setLabelAlignment(Qt::AlignRight);
    formLayout->addRow(QString("Roll Num:"), rollNumHLayout);
    formLayout->addRow(QString("Output Dir:"), getDirHLayout);

    QGridLayout *gridLayout = new QGridLayout;
    gridLayout->addWidget(instructLabel, 0, 0, 1, 2);
    gridLayout->addLayout(formLayout, 1, 0, 1, 2);
    gridLayout->addWidget(progbar, 2, 0);
    gridLayout->addWidget(checkButton, 2, 1);
    gridLayout->setRowStretch(1, 1);
    gridLayout->setColumnStretch(1, 0);

    QWidget *widget = new QWidget(this);
    widget->setLayout(gridLayout);

    setCentralWidget(widget);

    // 初始化狀態列
    statusBar = new QStatusBar(this);
    setStatusBar(statusBar);
    statusBar->showMessage("Ready");
}

void labelCheck::getOutputDir()
{
    QString DIR = QFileDialog::getExistingDirectory(
        nullptr,                                                     // 父窗口，通常設為 nullptr 或你的主窗口
        "Choose output directory",                                   // 對話框標題
        ".",                                                         // 預設開啟的路徑
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks // 選項：只顯示資料夾，不解析符號連結
    );

    if (!DIR.isEmpty())
    {
        outputDir = DIR;
        // 統一使用反斜槓 '\'
        outputDir.replace("/", "\\");
        outputDirLineEdit->setText(outputDir);
        updateStatusMessage("Output directory selected: " + outputDir);
    }
}

void labelCheck::updateRollNum()
{
    rollNumComboBox->clear();
    rollNumComboBox->addItems(inputDir.entryList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Reversed));
    updateStatusMessage("Roll number list updated");
}

void labelCheck::startDarknetDetection()
{
    checkButton->setDisabled(true);
    checkButton->setText("Processing...");
    instructLabel->setText(QString("Roll ").append(rollNumComboBox->currentText()).append(" processing..."));
    updateStatusMessage(QString("Processing Roll ").append(rollNumComboBox->currentText()).append("..."));

    QString destinationDir = outputDir;
    destinationDir.append("\\").append(rollNumComboBox->currentText()).append("\\L2");
    QDir qdir = QDir(destinationDir);
    if (!qdir.exists())
        qdir.mkpath(destinationDir);

    // 載入圖片
    QString rollDir = inputDir.absolutePath();
    rollDir.append("\\").append(rollNumComboBox->currentText()).append("\\L2");
    Darknet::VStr imagePaths = this->getImagePaths(rollDir.toStdString());
    if (imagePaths.empty())
    {
        qDebug() << "No images to process.";
        updateStatusMessage("No images found to process");
        resetCheckButton();
        return;
    }

    thread = new QThread(this);
    worker = new Worker(destinationDir, std::move(imagePaths), this);
    worker->moveToThread(thread);

    connect(thread, &QThread::started, worker, &Worker::runDarknetDetection);
    connect(worker, &Worker::updateProgress, this, &labelCheck::updateProgress);
    connect(worker, &Worker::finished, thread, &QThread::quit);
    connect(worker, &Worker::finished, worker, &QObject::deleteLater);
    connect(worker, &Worker::detectionError, this, [this]()
            { QTimer::singleShot(5000, this, [this]()
                                 {
                                     resetCheckButton(); // 恢復按鈕
                                 }); });
    connect(thread, &QThread::finished, thread, &QObject::deleteLater);

    progbar->setValue(0);
    thread->start();
}

std::vector<cv::Mat> labelCheck::splitImageWithOverlap(const cv::Mat &origin_img, int patch_width, int patch_height)
{
    try
    {
        cv::Mat gray;
        cv::cvtColor(origin_img, gray, cv::COLOR_BGR2GRAY);

        // 找到左右邊界
        int left = 0, right = gray.cols - 1;
        bool foundLeft = false, foundRight = false;

        // 從左邊開始找第一個非白色像素
        for (int x = 0; x < gray.cols; ++x)
        {
            for (int y = 0; y < gray.rows; ++y)
            {
                if (gray.at<uchar>(y, x) < 250)
                { // 假設白色像素值接近 255
                    left = x;
                    foundLeft = true;
                    break;
                }
            }
            if (foundLeft)
                break;
        }

        // 從右邊開始找第一個非白色像素
        for (int x = gray.cols - 1; x >= 0; --x)
        {
            for (int y = 0; y < gray.rows; ++y)
            {
                if (gray.at<uchar>(y, x) < 250)
                {
                    foundRight = true;
                    break;
                }
                else
                    right = x;
            }
            if (foundRight)
                break;
        }

        // 裁剪圖片
        cv::Mat img;
        if (foundLeft && foundRight && left < right)
        {
            cv::Rect roi(left, 0, right - left + 1, origin_img.rows); // 定義裁剪區域
            img = origin_img(roi);                                    // 裁剪
        }
        else
        {
            std::vector<cv::Mat> err = {};
            return err;
        }

        std::vector<cv::Mat> patches;

        int img_width = img.cols;
        int img_height = img.rows;

        // 計算垂直與水平 patch 起始位置
        std::vector<int> y_starts;
        std::vector<int> x_starts;

        // 垂直方向
        if (img_height >= patch_height)
        {
            for (int y = 0; y + patch_height <= img_height; y += patch_height)
                y_starts.push_back(y);
            if (y_starts.empty() || y_starts.back() + patch_height < img_height)
                y_starts.push_back(std::max(0, img_height - patch_height));
        }

        // 水平方向
        if (img_width >= patch_width)
        {
            for (int x = 0; x + patch_width <= img_width; x += patch_width)
                x_starts.push_back(x);
            if (x_starts.empty() || x_starts.back() + patch_width < img_width)
                x_starts.push_back(std::max(0, img_width - patch_width));
        }

        if (patch_width > img.cols || patch_height > img.rows)
        {
            std::vector<cv::Mat> err = {};
            return err;
        }

        // 切割區塊
        for (int y : y_starts)
        {
            for (int x : x_starts)
            {
                cv::Rect roi(x, y, patch_width, patch_height);
                patches.push_back(img(roi).clone());
            }
        }

        return patches;
    }
    catch (const std::exception &e)
    {
        qDebug() << "Error: " << QString::fromLocal8Bit(e.what());
        std::vector<cv::Mat> err = {};
        return err;
    }
}

bool labelCheck::copyFile(const std::string &source, const std::string &destination)
{
    namespace fs = std::filesystem;

    try
    {
        if (fs::exists(destination))
            fs::remove(destination);
        fs::copy_file(source, destination);
        // fs::copy_file(source, destination, fs::copy_options::overwrite_existing);
        qDebug() << "Copy to " << QString::fromStdString(destination);
        return true;
    }
    catch (const fs::filesystem_error &e)
    {
        qDebug() << "File system error: " << QString::fromLocal8Bit(e.what());
        return false;
    }
    catch (const std::exception &e)
    {
        qDebug() << "Error: " << QString::fromLocal8Bit(e.what());
        return false;
    }
}

std::vector<std::string> labelCheck::getImagePaths(const std::string &directory)
{
    namespace fs = std::filesystem;
    std::vector<std::string> imagePaths;

    // 常見的圖片副檔名集合
    std::set<std::string> imageExtensions = {".jpg", ".jpeg", ".png", ".bmp"};

    try
    {
        // 遍歷目錄中的所有檔案
        for (const auto &entry : fs::directory_iterator(directory))
        {
            if (entry.is_regular_file())
            { // 只處理一般檔案
                std::string path = entry.path().string();
                std::string ext = entry.path().extension().string();

                // 將副檔名轉為小寫，以便不區分大小寫比對
                for (char &c : ext)
                    c = std::tolower(c);

                // 檢查是否為圖片檔案
                if (imageExtensions.count(ext) > 0)
                {
                    // 獲取絕對路徑並添加到列表
                    imagePaths.push_back(fs::absolute(entry.path()).string());
                }
            }
        }
    }
    catch (const fs::filesystem_error &e)
    {
        qDebug() << "Error: " << QString::fromLocal8Bit(e.what()) << '\n';
    }

    return imagePaths;
}

void labelCheck::resetCheckButton()
{
    if (!checkButton->isEnabled())
    {
        checkButton->setEnabled(true);
        checkButton->setText("Check");
        instructLabel->setText("Select roll number to check:");
        updateStatusMessage("Ready");
    }
}

void labelCheck::updateProgress(int value)
{
    progbar->setValue(value);
}

void labelCheck::updateStatusMessage(const QString &message)
{
    statusBar->showMessage(message);
}

labelCheck::~labelCheck()
{
    QSettings settings("settings.ini", QSettings::IniFormat);

    settings.setValue("window/size", size());
    settings.setValue("window/position", pos());
    settings.setValue("output/directory", outputDir);

    if (thread && thread->isRunning())
    {
        delete thread;
        delete worker;
    }
}