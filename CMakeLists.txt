cmake_minimum_required(VERSION 3.16)
project(OrthoSeg VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs dnn)

# Compile the existing inference implementation without changing its standalone build.
add_library(medsam2_ocv STATIC ocv/src/medsam2_ocv.cpp)
target_include_directories(medsam2_ocv PUBLIC ocv/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(medsam2_ocv PUBLIC ${OpenCV_LIBS})

# Pure-OpenCV segmentation engine (no Qt) so it can be unit tested headlessly.
add_library(orthoseg_core STATIC
    src/SegmentationEngine.cpp
    src/AIFill.cpp
)
target_include_directories(orthoseg_core PUBLIC src ocv/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(orthoseg_core PUBLIC ${OpenCV_LIBS})

add_library(orthoseg_ai STATIC src/MedSAM2Inference.cpp src/AIFillController.cpp)
target_link_libraries(orthoseg_ai PUBLIC orthoseg_core medsam2_ocv Qt6::Core)
target_compile_definitions(orthoseg_ai PUBLIC
    ORTHOSEG_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/ocv/models")

add_executable(orthoseg
    src/main.cpp
    src/Document.cpp
    src/CanvasWidget.cpp
    src/MainWindow.cpp
)
target_link_libraries(orthoseg PRIVATE orthoseg_ai Qt6::Widgets)

enable_testing()
add_executable(test_ai tests/test_ai.cpp)
target_link_libraries(test_ai PRIVATE orthoseg_ai)
add_test(NAME ai COMMAND test_ai)
set_tests_properties(ai PROPERTIES TIMEOUT 15)
add_executable(test_seg tests/test_seg.cpp)
target_link_libraries(test_seg PRIVATE orthoseg_core)
add_test(NAME segmentation COMMAND test_seg)

add_executable(test_document tests/test_document.cpp src/Document.cpp)
target_link_libraries(test_document PRIVATE orthoseg_core)
add_test(NAME document COMMAND test_document)

add_executable(test_ui
    tests/test_ui.cpp
    src/Document.cpp
    src/CanvasWidget.cpp
    src/MainWindow.cpp
)
target_link_libraries(test_ui PRIVATE orthoseg_ai Qt6::Widgets)
add_test(NAME ui COMMAND test_ui)
set_tests_properties(ui PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen" TIMEOUT 30)
