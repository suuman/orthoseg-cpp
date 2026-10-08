#pragma once
#include <opencv2/dnn.hpp>
#include <string>

// Repair missing ONNX branch shape metadata in memory for OpenCV 5 import.
// Keep model operations (including cubic Resize) and source files unchanged.
cv::dnn::Net readMedSAM2ONNX(const std::string& path);
