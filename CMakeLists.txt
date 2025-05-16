cmake_minimum_required(VERSION 3.5) # CMake install : https://cmake.org/download/
project(labelCheck_GPU LANGUAGES CXX)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_PREFIX_PATH "C:/Qt/6.9.0/msvc2022_64") # Qt Kit Dir
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR "D:/opencv_workspace/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS}) # 連結opencv
include_directories(${CMAKE_SOURCE_DIR}/darknet/include) # 連結darknet.h
link_directories(${CMAKE_SOURCE_DIR}/darknet/bin) # 連結libdarknet.dll
link_directories(${CMAKE_SOURCE_DIR}/darknet/lib) # 連結darknet.lib

find_package(Qt6 COMPONENTS Widgets REQUIRED) # Qt COMPONENTS
aux_source_directory(./src srcs)

# Specify MSVC UTF-8 encoding   
add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

add_executable(${PROJECT_NAME}
    # WIN32 # If you need a terminal for debug, please comment this statement 
    ${srcs} 
) 
target_link_libraries(${PROJECT_NAME} PRIVATE Qt6::Widgets ${OpenCV_LIBS} darknet) # Qt6 Shared Library + opencv + darknet