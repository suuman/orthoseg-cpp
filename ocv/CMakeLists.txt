cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(medsam2_ocv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()

# Default to custom OpenCV 5 installation if not specified
if(NOT OpenCV_DIR)
    if(EXISTS "/home/suman/soft/opencv/install_new/lib/cmake/opencv5")
        set(OpenCV_DIR "/home/suman/soft/opencv/install_new/lib/cmake/opencv5")
    elseif(EXISTS "/usr/local/lib/cmake/opencv5")
        set(OpenCV_DIR "/usr/local/lib/cmake/opencv5")
    endif()
endif()

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV found version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(medsam2_ocv_infer
    src/medsam2_ocv.cpp
    src/main.cpp
)

target_link_libraries(medsam2_ocv_infer PRIVATE ${OpenCV_LIBRARIES})
target_link_options(medsam2_ocv_infer PRIVATE "-Wl,--disable-new-dtags")
set_property(TARGET medsam2_ocv_infer PROPERTY CXX_STANDARD 17)

# Embed RPATH to OpenCV 5 shared libraries and CUDA so executable runs standalone
get_filename_component(OPENCV_LIB_DIR "${OpenCV_DIR}/../.." ABSOLUTE)
set(ALL_RPATHS "${OPENCV_LIB_DIR}")

if(EXISTS "/usr/local/cuda/lib64")
    list(APPEND ALL_RPATHS "/usr/local/cuda/lib64")
endif()
if(EXISTS "/usr/local/cuda-13.4/targets/x86_64-linux/lib")
    list(APPEND ALL_RPATHS "/usr/local/cuda-13.4/targets/x86_64-linux/lib")
endif()

set_target_properties(medsam2_ocv_infer PROPERTIES
    INSTALL_RPATH "${ALL_RPATHS}"
    BUILD_WITH_INSTALL_RPATH TRUE
)
