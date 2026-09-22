cmake_minimum_required(VERSION 3.18)
project(OrthoSeg VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOMOC_SEARCH_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Build options
option(BUILD_TESTS "Build test suites" ON)
option(BUILD_EXAMPLES "Build standalone example tools" ON)

if(DEFINED BUILD_TESTING)
  set(BUILD_TESTS ${BUILD_TESTING})
endif()

# Default to custom OpenCV 5 installation if not specified and available
if(NOT OpenCV_DIR)
    if(EXISTS "/home/suman/soft/opencv/install_new/lib/cmake/opencv5")
        set(OpenCV_DIR "/home/suman/soft/opencv/install_new/lib/cmake/opencv5")
    elseif(EXISTS "/usr/local/lib/cmake/opencv5")
        set(OpenCV_DIR "/usr/local/lib/cmake/opencv5")
    endif()
endif()

find_package(Qt6 REQUIRED COMPONENTS Widgets)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs dnn)

message(STATUS "OpenCV found version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

include_directories(include)

# ---------------------------------------------------------------------------
# Libraries
# ---------------------------------------------------------------------------

# MedSAM2 OpenCV DNN implementation
add_library(medsam2_ocv STATIC src/medsam2_ocv.cpp)
target_include_directories(medsam2_ocv PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(medsam2_ocv PUBLIC ${OpenCV_LIBS})

# Pure-OpenCV segmentation engine (no Qt) so it can be unit tested headlessly.
add_library(orthoseg_core STATIC
    src/SegmentationEngine.cpp
    src/AIFill.cpp
)
target_include_directories(orthoseg_core PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(orthoseg_core PUBLIC ${OpenCV_LIBS})

# AI fill controller and worker
add_library(orthoseg_ai STATIC
    src/MedSAM2Inference.cpp
    src/AIFillController.cpp
    include/AIFillController.h
)
target_include_directories(orthoseg_ai PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(orthoseg_ai PUBLIC orthoseg_core medsam2_ocv Qt6::Core)
target_compile_definitions(orthoseg_ai PUBLIC
    ORTHOSEG_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/models")

# ---------------------------------------------------------------------------
# Executables
# ---------------------------------------------------------------------------

# Main Qt desktop application
add_executable(orthoseg
    src/main.cpp
    src/Document.cpp
    src/CanvasWidget.cpp
    src/MainWindow.cpp
    include/CanvasWidget.h
    include/MainWindow.h
)
target_include_directories(orthoseg PRIVATE include)
target_link_libraries(orthoseg PRIVATE orthoseg_ai Qt6::Widgets)

# Standalone MedSAM2 CLI inference tool
if(BUILD_EXAMPLES)
    add_executable(medsam2_ocv_infer examples/medsam2_main.cpp)
    target_include_directories(medsam2_ocv_infer PRIVATE include)
    target_link_libraries(medsam2_ocv_infer PRIVATE medsam2_ocv ${OpenCV_LIBS})
    target_link_options(medsam2_ocv_infer PRIVATE "-Wl,--disable-new-dtags")
endif()

# ---------------------------------------------------------------------------
# RPATH Configuration for standalone execution with custom OpenCV / CUDA
# ---------------------------------------------------------------------------
set(ALL_RPATHS "")
if(OpenCV_DIR)
    get_filename_component(OPENCV_LIB_DIR "${OpenCV_DIR}/../.." ABSOLUTE)
    if(EXISTS "${OPENCV_LIB_DIR}")
        list(APPEND ALL_RPATHS "${OPENCV_LIB_DIR}")
    endif()
endif()

if(EXISTS "/usr/local/cuda/lib64")
    list(APPEND ALL_RPATHS "/usr/local/cuda/lib64")
endif()
if(EXISTS "/usr/local/cuda-13.4/targets/x86_64-linux/lib")
    list(APPEND ALL_RPATHS "/usr/local/cuda-13.4/targets/x86_64-linux/lib")
endif()

if(ALL_RPATHS)
    if(BUILD_EXAMPLES)
        set_target_properties(medsam2_ocv_infer PROPERTIES
            INSTALL_RPATH "${ALL_RPATHS}"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    endif()
    target_link_options(orthoseg PRIVATE "-Wl,--disable-new-dtags")
    set_target_properties(orthoseg PROPERTIES
        INSTALL_RPATH "${ALL_RPATHS}"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
if(BUILD_TESTS)
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
        include/CanvasWidget.h
        include/MainWindow.h
    )
    target_link_libraries(test_ui PRIVATE orthoseg_ai Qt6::Widgets)
    add_test(NAME ui COMMAND test_ui)
    set_tests_properties(ui PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen" TIMEOUT 30)

    if(ALL_RPATHS)
        set_target_properties(test_ai PROPERTIES
            INSTALL_RPATH "${ALL_RPATHS}"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
        set_target_properties(test_ui PROPERTIES
            INSTALL_RPATH "${ALL_RPATHS}"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    endif()
endif()
