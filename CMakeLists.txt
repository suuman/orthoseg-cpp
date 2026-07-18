cmake_minimum_required(VERSION 3.16)
project(OrthoSeg VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# Pure-OpenCV segmentation engine (no Qt) so it can be unit tested headlessly.
add_library(orthoseg_core STATIC
    src/SegmentationEngine.cpp
)
target_include_directories(orthoseg_core PUBLIC src ${OpenCV_INCLUDE_DIRS})
target_link_libraries(orthoseg_core PUBLIC ${OpenCV_LIBS})

add_executable(orthoseg
    src/main.cpp
    src/Document.cpp
    src/CanvasWidget.cpp
    src/MainWindow.cpp
)
target_link_libraries(orthoseg PRIVATE orthoseg_core Qt6::Widgets)

enable_testing()
add_executable(test_seg tests/test_seg.cpp)
target_link_libraries(test_seg PRIVATE orthoseg_core)
add_test(NAME segmentation COMMAND test_seg)
