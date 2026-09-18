#include "medsam2_ocv.h"

#include <iostream>
#include <chrono>

MedSAM2Engine::MedSAM2Engine(
    const std::string& encoder_path,
    const std::string& decoder_path,
    const std::string& device_str
) : device_str_(device_str),
    backend_id_(cv::dnn::DNN_BACKEND_OPENCV),
    target_id_(cv::dnn::DNN_TARGET_CPU) {

    if (device_str == "cuda" && cv::cuda::getCudaEnabledDeviceCount() > 0) {
        backend_id_ = cv::dnn::DNN_BACKEND_CUDA;
        target_id_ = cv::dnn::DNN_TARGET_CUDA;
        device_str_ = "cuda";
        std::cout << "[MedSAM2 OpenCV5 C++] Using device: CUDA (Device count: "
                  << cv::cuda::getCudaEnabledDeviceCount() << ")" << std::endl;
    } else {
        backend_id_ = cv::dnn::DNN_BACKEND_OPENCV;
        target_id_ = cv::dnn::DNN_TARGET_CPU;
        device_str_ = "cpu";
        std::cout << "[MedSAM2 OpenCV5 C++] Using device: CPU" << std::endl;
    }

    std::cout << "[MedSAM2 OpenCV5 C++] Loading Image Encoder: " << encoder_path << std::endl;
    encoder_ = cv::dnn::readNetFromONNX(encoder_path);
    if (encoder_.empty()) {
        std::cerr << "Error: Failed to load Image Encoder ONNX model: " << encoder_path << std::endl;
    }
    encoder_.setPreferableBackend(backend_id_);
    encoder_.setPreferableTarget(target_id_);

    std::cout << "[MedSAM2 OpenCV5 C++] Loading Mask Decoder: " << decoder_path << std::endl;
    decoder_ = cv::dnn::readNetFromONNX(decoder_path);
    if (decoder_.empty()) {
        std::cerr << "Error: Failed to load Mask Decoder ONNX model: " << decoder_path << std::endl;
    }
    decoder_.setPreferableBackend(backend_id_);
    decoder_.setPreferableTarget(target_id_);
}

SegmentationResult MedSAM2Engine::segment(
    const cv::Mat& image_rgb,
    const std::vector<Prompt>& prompts,
    bool pad_to_square,
    int pad_size,
    float score_thresh
) {
    SegmentationResult result;
    result.width = image_rgb.cols;
    result.height = image_rgb.rows;
    result.label_map = cv::Mat::zeros(result.height, result.width, CV_8UC1);

    if (prompts.empty() || image_rgb.empty()) {
        if (prompts.empty()) {
            std::cerr << "[MedSAM2 OpenCV5 C++] Warning: No prompts provided." << std::endl;
        }
        return result;
    }

    // 1. Generic geometry mapping: accepts ANY image size (e.g. 512x512, 128x1024, 2048x1536)
    PadInfo pad = computePadInfo(image_rgb.cols, image_rgb.rows, pad_size, pad_to_square);
    cv::Mat feed_img = padImageToSquare(image_rgb, pad);

    // 2. Prepare input image blob (1, 3, S, S) float32 in [0, 255]
    cv::Mat feed_float;
    feed_img.convertTo(feed_float, CV_32FC3);
    cv::Mat blob = cv::dnn::blobFromImage(
        feed_float,
        1.0,
        cv::Size(pad.target_size, pad.target_size),
        cv::Scalar(0, 0, 0),
        false, // swapRB = false because feed_img is already RGB
        false,
        CV_32F
    );

    // 3. Image Encoder forward pass
    auto t0 = std::chrono::high_resolution_clock::now();
    encoder_.setInput(blob, "image");
    std::vector<cv::Mat> enc_outs;
    std::vector<std::string> enc_out_names = {"image_embed", "high_res_0", "high_res_1"};
    encoder_.forward(enc_outs, enc_out_names);
    auto t1 = std::chrono::high_resolution_clock::now();
    double enc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[MedSAM2 OpenCV5 C++] Image (" << image_rgb.cols << "x" << image_rgb.rows
              << ") encoded in " << enc_ms << " ms"
              << " [scale: " << pad.scale_x << ", canvas offset: (" << pad.x0 << ", " << pad.y0 << ")]"
              << std::endl;

    cv::Mat image_embed = enc_outs[0];
    cv::Mat high_res_0 = enc_outs[1];
    cv::Mat high_res_1 = enc_outs[2];

    // 4. Run Mask Decoder for each prompt
    std::vector<cv::Mat> logit_stack;
    std::vector<int> class_order;
    std::vector<std::string> dec_out_names = {"mask", "iou_pred"};

    for (const auto& prompt : prompts) {
        // Transform box from original image space [0, W] x [0, H] to canvas space [0, S] x [0, S]
        float bx0 = prompt.box.x0 * pad.scale_x + static_cast<float>(pad.x0);
        float by0 = prompt.box.y0 * pad.scale_y + static_cast<float>(pad.y0);
        float bx1 = prompt.box.x1 * pad.scale_x + static_cast<float>(pad.x0);
        float by1 = prompt.box.y1 * pad.scale_y + static_cast<float>(pad.y0);

        // Clamp to valid canvas dimensions
        bx0 = std::clamp(bx0, 0.0f, static_cast<float>(pad.target_size - 1));
        by0 = std::clamp(by0, 0.0f, static_cast<float>(pad.target_size - 1));
        bx1 = std::clamp(bx1, 0.0f, static_cast<float>(pad.target_size - 1));
        by1 = std::clamp(by1, 0.0f, static_cast<float>(pad.target_size - 1));

        float box_arr[4] = {bx0, by0, bx1, by1};
        cv::Mat box_mat(1, 4, CV_32F, box_arr);

        cv::Mat mask_input;
        float has_mask_val = 0.0f;
        if (prompt.has_mask && !prompt.mask_logits.empty()) {
            mask_input = prompt.mask_logits;
            has_mask_val = 1.0f;
        } else {
            int sz[4] = {1, 1, 256, 256};
            mask_input = cv::Mat::zeros(4, sz, CV_32F);
            has_mask_val = 0.0f;
        }

        float has_mask_arr[1] = {has_mask_val};
        cv::Mat has_mask_mat(1, 1, CV_32F, has_mask_arr);

        decoder_.setInput(image_embed, "image_embed");
        decoder_.setInput(high_res_0, "high_res_0");
        decoder_.setInput(high_res_1, "high_res_1");
        decoder_.setInput(box_mat, "box");
        decoder_.setInput(mask_input, "mask_input");
        decoder_.setInput(has_mask_mat, "has_mask");

        std::vector<cv::Mat> dec_outs;
        auto dt0 = std::chrono::high_resolution_clock::now();
        decoder_.forward(dec_outs, dec_out_names);
        auto dt1 = std::chrono::high_resolution_clock::now();
        double dec_ms = std::chrono::duration<double, std::milli>(dt1 - dt0).count();

        float iou = dec_outs[1].at<float>(0, 0);
        result.iou_scores[prompt.class_id] = iou;

        // dec_outs[0] has shape [1, 1, S, S]
        int S = pad.target_size;
        cv::Mat mask_mat(S, S, CV_32F, dec_outs[0].ptr<float>());
        logit_stack.push_back(mask_mat.clone());
        class_order.push_back(prompt.class_id);

        std::cout << "  Prompt class " << prompt.class_id
                  << " orig box: [" << prompt.box.x0 << ", " << prompt.box.y0
                  << ", " << prompt.box.x1 << ", " << prompt.box.y1
                  << "] -> canvas box: [" << bx0 << ", " << by0 << ", " << bx1 << ", " << by1 << "]"
                  << " -> Predicted IoU: " << iou
                  << " (decoded in " << dec_ms << " ms)" << std::endl;
    }

    // 5. Combine predictions into multiclass label map via competitive argmax
    int S = pad.target_size;
    cv::Mat padded_mask = cv::Mat::zeros(S, S, CV_8UC1);

    if (!logit_stack.empty()) {
        for (int r = 0; r < S; ++r) {
            uint8_t* out_row = padded_mask.ptr<uint8_t>(r);
            for (int c = 0; c < S; ++c) {
                float max_val = -1e9f;
                int best_cls = 0;
                for (size_t i = 0; i < logit_stack.size(); ++i) {
                    float v = logit_stack[i].at<float>(r, c);
                    if (v > max_val) {
                        max_val = v;
                        best_cls = class_order[i];
                    }
                }
                if (max_val > score_thresh) {
                    out_row[c] = static_cast<uint8_t>(best_cls);
                }
            }
        }

        // 6. Unpad & unscale back to EXACT original image dimensions (W x H)
        result.label_map = unpadMask(padded_mask, pad);
    }

    return result;
}
