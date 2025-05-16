# Darknet 編譯步驟 (GPU 版本)

## 1. 下載與配置 OpenCV

### 1.1 下載 OpenCV
- 從 [OpenCV 官網](https://opencv.org/releases/) 下載 Sources 版本
- 注意：使用 MinGW 編譯時需要下載 Sources，使用 MSVC 編譯則可直接下載 Windows 版本

### 1.2 建立工作目錄
```bash
# 假設解壓縮在 D:\，則存在路徑如 D:\opencv-4.11.0
# 建立以下目錄結構：
D:\opencv_workspace
    ├── build-mingw
    ├── opencv_contrib
    └── sources
```

### 1.3 配置檔案
1. 將 `D:\opencv-4.11.0` 中的內容全部移至 `sources` 中
2. 下載 opencv_contrib：
```bash
git clone --depth 1 --branch 4.11.0 https://github.com/opencv/opencv_contrib.git
```

## 2. 配置與編譯 OpenCV (使用 MSVC)

### 2.1 前置準備
1. 下載並安裝 Visual Studio Community 2022
2. 在安裝過程中，選擇「使用 C++ 的桌面開發」工作負載
3. 確保包含 C++ 編譯器和 Windows SDK

### 2.2 編譯步驟
1. 啟動「x64 Native Tools Command Prompt for VS 2022」
2. 建立並進入編譯目錄：
```bash
mkdir D:\opencv_workspace\build-vc
cd D:\opencv_workspace\build-vc
```

3. 配置 CMake（選擇以下其中一個版本）：

   **CPU 版本**：
   ```bash
   cmake ../sources -G "Ninja" \
       -DBUILD_opencv_world=ON \
       -DCMAKE_INSTALL_PREFIX=D:/opencv_workspace/build \
       -DOPENCV_EXTRA_MODULES_PATH=D:/opencv_workspace/opencv_contrib/modules \
       -DCMAKE_BUILD_TYPE=Release \
       -DWITH_CUDA=OFF
   ```

   **GPU 版本**（編譯時間較長）：
   ```bash
   cmake ../sources -G "Ninja" \
       -DBUILD_opencv_world=ON \
       -DCMAKE_INSTALL_PREFIX=D:/opencv_workspace/build \
       -DOPENCV_EXTRA_MODULES_PATH=D:/opencv_workspace/opencv_contrib/modules \
       -DCMAKE_BUILD_TYPE=Release \
       -DWITH_CUDA=ON \
       -DCUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4"
   ```

4. 編譯與安裝：
```bash
ninja -j8
ninja install -j8
```

## 3. 下載與編譯 Darknet (使用 MSVC)

### 3.1 下載 Darknet
```bash
git clone https://github.com/hank-ai/darknet.git
cd darknet
mkdir build
cd build
```

### 3.2 配置 CMake
- 如果需要指定 DARKNET_CUDA_ARCHITECTURES 來配合不同型號 GPU 的 Compute Capability：
  - 在 `darknet\CM_dependencies.cmake` 第 44 行修改
  - 預設值為 "native"
  - GPU 的 Compute Capability 可查詢 [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda-gpus)

```bash
cmake .. -G "Ninja" \
    -DCMAKE_BUILD_TYPE=Release \
    -DOpenCV_DIR="D:\opencv_workspace\build\x64\vc17\lib" \
    -DDARKNET_TRY_CUDA=ON \
    -DCMAKE_CUDA_COMPILER="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe"
```

### 3.3 修改安裝配置
1. 修改 `D:\Code\labelCheck\darknet\build` 中所有的 `cmake_install.cmake`：
   - 第 5 行 CMAKE_INSTALL_PREFIX 的值改為 `"D:/Code/labelCheck/darknet"`

2. 修改 `D:\Code\labelCheck\darknet\build\src-cli\cmake_install.cmake`：
   - 第 52 行 DIRECTORIES 區域中新增以下路徑：
     ```
     "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin"
     "D:/opencv_workspace/build/x64/vc17/bin"
     ```

### 3.4 編譯與安裝
```bash
ninja -j8
ninja install -j8
```