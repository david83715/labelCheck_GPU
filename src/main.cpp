#include "labelCheck.h"
// #include <windows.h>

#include <QApplication>
#pragma comment(lib, "user32.lib")

int main(int argc, char *argv[])
{
    // SetConsoleOutputCP(CP_UTF8); // 強制輸出編碼為UTF-8以顯示中文

    QApplication a(argc, argv);
    labelCheck w;
    w.show();
    return a.exec();
}