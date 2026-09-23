#include "segmentor.hpp"
#include <iostream>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <thread>

namespace fs = std::filesystem;

namespace xray_ocv {

std::unique_ptr<XRaySegmentor> XRaySegmentor::create(const std::string& onnx_model_path,
                                                      const std::string& device_str,
                                                      float alpha,
                                                      bool draw_contours) {
    AccelerationMode mode = AccelerationMode::CUDA_FP16; // default for RTX 4050

    if (device_str == "cpu") {
        mode = AccelerationMode::CPU;
    } else if (device_str == "cuda_fp32" || device_str == "cuda32") {
        mode = AccelerationMode::CUDA_FP32;
    } else if (device_str == "cuda" || device_str == "cuda_fp16" || device_str == "auto") {
        if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
            mode = AccelerationMode::CUDA_FP16; // leverage RTX 4050 Tensor Cores
        } else {
            std::cout << "[WARN] CUDA device not detected. Falling back to multi-threaded CPU." << std::endl;
            mode = AccelerationMode::CPU;
        }
    }

    return std::make_unique<XRaySegmentor>(onnx_model_path, mode, alpha, draw_contours);
}

XRaySegmentor::XRaySegmentor(const std::string& onnx_model_path,
                             AccelerationMode mode,
                             float alpha,
                             bool draw_contours)
    : mode_(mode),
      alpha_(alpha),
      draw_contours_(draw_contours) {

    if (!fs::exists(onnx_model_path)) {
        throw std::runtime_error("ONNX model file not found: " + onnx_model_path);
    }

    std::cout << "[INFO] Loading ONNX model with OpenCV 5 DNN: " << onnx_model_path << std::endl;
    net_ = cv::dnn::readNetFromONNX(onnx_model_path);
    if (net_.empty()) {
        throw std::runtime_error("Failed to load model from: " + onnx_model_path);
    }

    // Configure hardware acceleration
    if (mode_ == AccelerationMode::CUDA_FP16 || mode_ == AccelerationMode::CUDA_FP32) {
        int cuda_count = cv::cuda::getCudaEnabledDeviceCount();
        if (cuda_count > 0) {
            cv::cuda::DeviceInfo dev_info(0);
            std::cout << "[INFO] Hardware Acceleration: NVIDIA GPU detected -> "
                      << dev_info.name() << " (Compute Capability: "
                      << dev_info.majorVersion() << "." << dev_info.minorVersion() << ")"
                      << std::endl;

            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            if (mode_ == AccelerationMode::CUDA_FP16) {
                std::cout << "[INFO] Enabling CUDA FP16 Tensor Core acceleration (RTX 4050 optimized)." << std::endl;
                net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
            } else {
                std::cout << "[INFO] Enabling CUDA FP32 acceleration." << std::endl;
                net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            }
        } else {
            std::cout << "[WARN] No CUDA device available. Falling back to CPU backend." << std::endl;
            mode_ = AccelerationMode::CPU;
        }
    }

    if (mode_ == AccelerationMode::CPU) {
        unsigned int num_cores = std::thread::hardware_concurrency();
        std::cout << "[INFO] Hardware Acceleration: AMD Ryzen 9 HX multi-threaded CPU ("
                  << num_cores << " threads, TBB + AVX MLAS enabled)." << std::endl;
        cv::setNumThreads(static_cast<int>(num_cores));
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    std::cout << "[INFO] OpenCV 5 DNN engine initialized successfully." << std::endl;
}

void XRaySegmentor::preprocess(const cv::Mat& orig_gray,
                               cv::Mat& img_infer,
                               int& infer_w,
                               int& infer_h,
                               float& scale) const {
    const int orig_h = orig_gray.rows;
    const int orig_w = orig_gray.cols;

    infer_h = TARGET_HEIGHT; // Always 2048 for inference

    if (orig_h == TARGET_HEIGHT) {
        infer_w = orig_w;
        scale = 1.0f;
        orig_gray.copyTo(img_infer);
    } else {
        scale = static_cast<float>(TARGET_HEIGHT) / static_cast<float>(orig_h);
        infer_w = std::max(1, static_cast<int>(std::round(static_cast<float>(orig_w) * scale)));
        // Bilinear resize to height 2048 preserving aspect ratio
        cv::resize(orig_gray, img_infer, cv::Size(infer_w, infer_h), 0, 0, cv::INTER_LINEAR);
    }
}

cv::Mat XRaySegmentor::run_sliding_window(const cv::Mat& img_infer, int infer_w, int infer_h) {
    // 1. Intensity Normalization: Z-Score (x - mean) / std matching nnUNet
    cv::Mat img_f;
    img_infer.convertTo(img_f, CV_32FC1);

    cv::Scalar mean_s, std_s;
    cv::meanStdDev(img_f, mean_s, std_s);
    const float mean_val = static_cast<float>(mean_s[0]);
    const float std_val = std::max(static_cast<float>(std_s[0]), 1e-8f);

    cv::Mat img_norm = (img_f - mean_val) / std_val;

    // Case A: Image width <= patch tile width (768) -> Center pad to 768
    if (infer_w <= TILE_WIDTH) {
        const int pad_left = (TILE_WIDTH - infer_w) / 2;
        const int pad_right = TILE_WIDTH - infer_w - pad_left;

        cv::Mat padded = cv::Mat::zeros(infer_h, TILE_WIDTH, CV_32FC1);
        img_norm.copyTo(padded(cv::Rect(pad_left, 0, infer_w, infer_h)));

        // Create 4D Blob: [1, 1, 2048, 768]
        cv::Mat blob = cv::dnn::blobFromImage(padded);
        net_.setInput(blob);
        cv::Mat out = net_.forward(); // Shape: [1, 3, 2048, 768]

        // Crop center padding and compute argmax across classes (0, 1, 2)
        cv::Mat pred(infer_h, infer_w, CV_8UC1);
        const float* p0 = out.ptr<float>(0, 0);
        const float* p1 = out.ptr<float>(0, 1);
        const float* p2 = out.ptr<float>(0, 2);

        for (int r = 0; r < infer_h; ++r) {
            uint8_t* p_dst = pred.ptr<uint8_t>(r);
            const int row_offset = r * TILE_WIDTH;
            for (int c = 0; c < infer_w; ++c) {
                const int idx = row_offset + pad_left + c;
                float v0 = p0[idx];
                float v1 = p1[idx];
                float v2 = p2[idx];

                uint8_t best_c = 0;
                float max_v = v0;
                if (v1 > max_v) {
                    max_v = v1;
                    best_c = 1; // Femur
                }
                if (v2 > max_v) {
                    best_c = 2; // Tibia
                }
                p_dst[c] = best_c;
            }
        }
        return pred;
    }

    // Case B: Image width > 768 -> Sliding window inference with Gaussian weighting
    const float tile_step_size = 0.5f;
    const float target_step = static_cast<float>(TILE_WIDTH) * tile_step_size;
    const int num_steps = static_cast<int>(std::ceil(static_cast<float>(infer_w - TILE_WIDTH) / target_step)) + 1;
    const float actual_step = static_cast<float>(infer_w - TILE_WIDTH) / static_cast<float>(num_steps - 1);

    std::vector<int> steps(num_steps);
    for (int i = 0; i < num_steps; ++i) {
        steps[i] = static_cast<int>(std::round(actual_step * static_cast<float>(i)));
    }

    // 1D Gaussian weighting along tile width
    std::vector<float> gaussian_w(TILE_WIDTH);
    const float center = static_cast<float>(TILE_WIDTH - 1) / 2.0f;
    const float sigma = static_cast<float>(TILE_WIDTH) / 6.0f; // 3 sigma covering half window

    for (int x = 0; x < TILE_WIDTH; ++x) {
        const float diff = (static_cast<float>(x) - center) / sigma;
        gaussian_w[x] = std::max(std::exp(-0.5f * diff * diff), 1e-4f);
    }

    // Accumulators for logits and weights
    const size_t total_elements = static_cast<size_t>(infer_h) * infer_w;
    std::vector<float> accum_c0(total_elements, 0.0f);
    std::vector<float> accum_c1(total_elements, 0.0f);
    std::vector<float> accum_c2(total_elements, 0.0f);
    std::vector<float> accum_w(total_elements, 0.0f);

    for (int start_col : steps) {
        cv::Mat tile = img_norm(cv::Rect(start_col, 0, TILE_WIDTH, infer_h));
        cv::Mat blob = cv::dnn::blobFromImage(tile);
        net_.setInput(blob);
        cv::Mat out = net_.forward();

        const float* p0 = out.ptr<float>(0, 0);
        const float* p1 = out.ptr<float>(0, 1);
        const float* p2 = out.ptr<float>(0, 2);

        for (int r = 0; r < infer_h; ++r) {
            const int src_row = r * TILE_WIDTH;
            const int dst_row = r * infer_w;
            for (int c = 0; c < TILE_WIDTH; ++c) {
                const float w = gaussian_w[c];
                const int src_idx = src_row + c;
                const int dst_idx = dst_row + start_col + c;

                accum_c0[dst_idx] += p0[src_idx] * w;
                accum_c1[dst_idx] += p1[src_idx] * w;
                accum_c2[dst_idx] += p2[src_idx] * w;
                accum_w[dst_idx] += w;
            }
        }
    }

    // Compute argmax across accumulated channels
    cv::Mat pred(infer_h, infer_w, CV_8UC1);
    for (int r = 0; r < infer_h; ++r) {
        uint8_t* p_dst = pred.ptr<uint8_t>(r);
        const int dst_row = r * infer_w;
        for (int c = 0; c < infer_w; ++c) {
            const int idx = dst_row + c;
            float v0 = accum_c0[idx];
            float v1 = accum_c1[idx];
            float v2 = accum_c2[idx];

            uint8_t best_c = 0;
            float max_v = v0;
            if (v1 > max_v) {
                max_v = v1;
                best_c = 1;
            }
            if (v2 > max_v) {
                best_c = 2;
            }
            p_dst[c] = best_c;
        }
    }

    return pred;
}

void XRaySegmentor::create_visualization(const cv::Mat& orig_bgr,
                                         const cv::Mat& label_mask,
                                         cv::Mat& overlay_bgr,
                                         cv::Mat& color_mask_bgr) const {
    const int h = label_mask.rows;
    const int w = label_mask.cols;

    color_mask_bgr = cv::Mat::zeros(h, w, CV_8UC3);
    overlay_bgr = orig_bgr.clone();

    // Fill color mask:
    // Label 1 (Femur): Green (0, 255, 0)
    // Label 2 (Tibia): Red (0, 0, 255)
    for (int r = 0; r < h; ++r) {
        const uint8_t* p_lbl = label_mask.ptr<uint8_t>(r);
        cv::Vec3b* p_col = color_mask_bgr.ptr<cv::Vec3b>(r);
        for (int c = 0; c < w; ++c) {
            if (p_lbl[c] == LABEL_FEMUR) {
                p_col[c] = cv::Vec3b(0, 255, 0); // Green
            } else if (p_lbl[c] == LABEL_TIBIA) {
                p_col[c] = cv::Vec3b(0, 0, 255); // Red in BGR
            }
        }
    }

    // Alpha blend where label > 0
    const float alpha = alpha_;
    const float beta = 1.0f - alpha;

    for (int r = 0; r < h; ++r) {
        const uint8_t* p_lbl = label_mask.ptr<uint8_t>(r);
        const cv::Vec3b* p_col = color_mask_bgr.ptr<cv::Vec3b>(r);
        const cv::Vec3b* p_orig = orig_bgr.ptr<cv::Vec3b>(r);
        cv::Vec3b* p_over = overlay_bgr.ptr<cv::Vec3b>(r);

        for (int c = 0; c < w; ++c) {
            if (p_lbl[c] > 0) {
                p_over[c][0] = cv::saturate_cast<uint8_t>(beta * p_orig[c][0] + alpha * p_col[c][0]);
                p_over[c][1] = cv::saturate_cast<uint8_t>(beta * p_orig[c][1] + alpha * p_col[c][1]);
                p_over[c][2] = cv::saturate_cast<uint8_t>(beta * p_orig[c][2] + alpha * p_col[c][2]);
            }
        }
    }

    // Draw boundary contours
    if (draw_contours_) {
        cv::Mat mask_femur = (label_mask == LABEL_FEMUR);
        cv::Mat mask_tibia = (label_mask == LABEL_TIBIA);

        std::vector<std::vector<cv::Point>> contours_f, contours_t;
        cv::findContours(mask_femur, contours_f, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::findContours(mask_tibia, contours_t, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::drawContours(overlay_bgr, contours_f, -1, cv::Scalar(128, 255, 0), 2);
        cv::drawContours(overlay_bgr, contours_t, -1, cv::Scalar(60, 60, 255), 2);
    }
}

InferenceResult XRaySegmentor::predict(const cv::Mat& input_image, const std::string& input_name) {
    if (input_image.empty()) {
        throw std::runtime_error("Input image is empty!");
    }

    InferenceResult res;
    res.input_path = input_name;

    if (input_image.channels() == 1) {
        res.original_gray = input_image.clone();
        cv::cvtColor(input_image, res.original_bgr, cv::COLOR_GRAY2BGR);
    } else if (input_image.channels() == 3) {
        res.original_bgr = input_image.clone();
        cv::cvtColor(input_image, res.original_gray, cv::COLOR_BGR2GRAY);
    } else if (input_image.channels() == 4) {
        cv::cvtColor(input_image, res.original_bgr, cv::COLOR_BGRA2BGR);
        cv::cvtColor(res.original_bgr, res.original_gray, cv::COLOR_BGR2GRAY);
    } else {
        throw std::runtime_error("Unsupported number of channels: " + std::to_string(input_image.channels()));
    }

    res.orig_height = res.original_gray.rows;
    res.orig_width = res.original_gray.cols;

    // 1. Resize to height 2048 preserving aspect ratio
    cv::Mat img_infer;
    float scale = 1.0f;
    preprocess(res.original_gray, img_infer, res.infer_width, res.infer_height, scale);

    std::cout << "[INFO] Image: " << res.orig_width << "x" << res.orig_height
              << " -> Inference size: " << res.infer_width << "x" << res.infer_height
              << " (scale=" << scale << ")" << std::endl;

    // 2. Perform OpenCV DNN model inference
    cv::Mat pred_infer = run_sliding_window(img_infer, res.infer_width, res.infer_height);

    // 3. Resize prediction back to original dimensions using NEAREST neighbor interpolation
    if (res.orig_height == res.infer_height && res.orig_width == res.infer_width) {
        res.label_mask = pred_infer.clone();
    } else {
        cv::resize(pred_infer, res.label_mask, cv::Size(res.orig_width, res.orig_height), 0, 0, cv::INTER_NEAREST);
    }

    // 4. Create color mask and overlay on original image
    create_visualization(res.original_bgr, res.label_mask, res.overlay_bgr, res.color_mask_bgr);

    return res;
}

InferenceResult XRaySegmentor::predict(const std::string& image_path) {
    if (!fs::exists(image_path)) {
        throw std::runtime_error("Image file not found: " + image_path);
    }

    cv::Mat img = cv::imread(image_path, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        throw std::runtime_error("Failed to read image with OpenCV: " + image_path);
    }

    if (img.depth() == CV_16U) {
        img.convertTo(img, CV_8U, 1.0 / 256.0);
    }

    return predict(img, image_path);
}

InferenceResult XRaySegmentor::predict_and_save(const std::string& image_path,
                                                const std::string& output_dir,
                                                const std::string& output_label,
                                                const std::string& output_overlay,
                                                const std::string& base_name) {
    InferenceResult res = predict(image_path);

    std::string stem = base_name;
    if (stem.empty()) {
        stem = fs::path(image_path).stem().string();
    }

    if (!output_label.empty()) {
        fs::path p(output_label);
        if (p.has_parent_path()) {
            fs::create_directories(p.parent_path());
        }
        cv::imwrite(output_label, res.label_mask);
        std::cout << "[INFO] Saved label image -> " << output_label << std::endl;
    }

    if (!output_overlay.empty()) {
        fs::path p(output_overlay);
        if (p.has_parent_path()) {
            fs::create_directories(p.parent_path());
        }
        cv::imwrite(output_overlay, res.overlay_bgr);
        std::cout << "[INFO] Saved overlay image -> " << output_overlay << std::endl;
    }

    if (!output_dir.empty() || (output_label.empty() && output_overlay.empty())) {
        std::string target_dir = output_dir.empty() ? "./inference_outputs" : output_dir;
        res.save(target_dir, stem, true, false);
    }

    return res;
}

void InferenceResult::save(const std::string& output_dir,
                           const std::string& base_name,
                           bool save_color_mask,
                           bool save_stacked) const {
    fs::create_directories(output_dir);

    std::string stem = base_name;
    if (stem.empty()) {
        if (!input_path.empty()) {
            stem = fs::path(input_path).stem().string();
        } else {
            stem = "xray_inference";
        }
    }

    // 1. Save label mask: uint8 with values {0, 1, 2}
    std::string label_path = (fs::path(output_dir) / (stem + "_label.png")).string();
    cv::imwrite(label_path, label_mask);
    std::cout << "[INFO] Saved label -> " << label_path << std::endl;

    // 2. Save overlay image
    std::string overlay_path = (fs::path(output_dir) / (stem + "_overlay.png")).string();
    cv::imwrite(overlay_path, overlay_bgr);
    std::cout << "[INFO] Saved overlay -> " << overlay_path << std::endl;

    // 3. Save color mask
    if (save_color_mask) {
        std::string color_mask_path = (fs::path(output_dir) / (stem + "_color_mask.png")).string();
        cv::imwrite(color_mask_path, color_mask_bgr);
        std::cout << "[INFO] Saved color_mask -> " << color_mask_path << std::endl;
    }

    // 4. Save per-channel binary masks (ch1: femur, ch2: tibia)
    if (save_stacked) {
        fs::path stacked_dir = fs::path(output_dir) / (stem + "_channels");
        fs::create_directories(stacked_dir);

        cv::Mat femur_mask = (label_mask == LABEL_FEMUR);
        cv::Mat tibia_mask = (label_mask == LABEL_TIBIA);

        cv::imwrite((stacked_dir / "ch1_femur.png").string(), femur_mask);
        cv::imwrite((stacked_dir / "ch2_tibia.png").string(), tibia_mask);
        std::cout << "[INFO] Saved stacked channels -> " << stacked_dir.string() << std::endl;
    }
}

} // namespace xray_ocv
