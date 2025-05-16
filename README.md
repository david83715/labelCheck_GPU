# LabelCheck GPU

這是一個基於 YOLOv4-tiny 的標籤檢測工具，使用 GPU 加速進行影像處理。

### 含有標籤的大圖
<img src="./image/example1.jpg" alt="含有標籤的大圖">
<img src="./image/example2.jpg" alt="含有標籤的大圖">
<img src="./image/example3.jpg" alt="含有標籤的大圖">
<img src="./image/example4.jpg" alt="含有標籤的大圖">

### 標籤
<div style="display: flex; gap: 10px;">
    <img src="./image/example1_cut.jpg" alt="標籤" style="width: 200px; height: 200px;">
    <img src="./image/example2_cut.jpg" alt="標籤" style="width: 200px; height: 200px;">
    <img src="./image/example3_cut.jpg" alt="標籤" style="width: 200px; height: 200px;">
    <img src="./image/example4_cut.jpg" alt="標籤" style="width: 200px; height: 200px;">
</div>

## 功能特點

- 使用 YOLOv4-tiny 進行物件檢測
- darknet 使用 https://github.com/hank-ai/darknet.git
- GPU 加速處理
- 支援批次處理圖片
- 自動分割大圖進行處理
- 多執行緒處理

## 系統需求

- Windows 10 或更新版本
- CUDA 支援的 GPU
- CMake 3.0 或更新版本
- Qt 6.x
- OpenCV

## 建置說明

1. 確保已安裝所有必要依賴
2. 執行以下命令：
```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

## 使用說明

1. 執行程式，初次執行時要先於settings.ini設定檢驗大圖所在路徑
2. 選擇要檢查的卷號
3. 選擇輸出目錄
4. 點擊 "Check" 開始處理

## 注意事項

- 需有安裝CUDA與Cudnn等運行GPU的所需的環境
- 需先配置並編譯OpenCV與darknet，步驟可參考 https://github.com/hank-ai/darknet.git 或是 [darknetBuildStep.md](darknetBuildStep.md)
- 請確保有足夠的 GPU 記憶體