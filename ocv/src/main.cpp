#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <chrono>
#include <iomanip>
#include <sstream>

#include "image_utils.h"
#include "medsam2_ocv.h"

void printUsage(const char* prog) {
    std::cout << "\n=======================================================\n"
              << "MedSAM2 OpenCV 5 + CUDA Standalone Inference Tool\n"
              << "=======================================================\n"
              << "Usage: " << prog << " --image <path> [options]\n\n"
              << "Required Arguments:\n"
              << "  --image <path>         Path to input image (any size / aspect ratio / format)\n"
              << "  --out_mask <path>      Path to output refined mask (PNG, matches original image size)\n\n"
              << "Prompt Options (provide at least one):\n"
              << "  --prompt_mask <path>   Ground-truth or coarse mask PNG (derives prompts per class)\n"
              << "  --class_ids <list>     Comma-separated class IDs to process (e.g. \"1,2\"; default: auto-detected)\n"
              << "  --box \"x0,y0,x1,y1\"    Bounding box in original image coords (or \"cid,x0,y0,x1,y1\")\n"
              << "  --class_id <int>       Class ID for the preceding box (default: 1)\n"
              << "  --box2 \"x0,y0,x1,y1\"   Second bounding box\n"
              << "  --class_id2 <int>      Class ID for second box (default: 2)\n"
              << "  --boxes <string>       Multiple boxes formatted as \"cid,x0,y0,x1,y1;cid,x0,y0,x1,y1\"\n\n"
              << "Output Options:\n"
              << "  --out_overlay <path>   Path to output overlay visualization (PNG, matches original image size)\n"
              << "  --alpha <float>        Overlay blending opacity in [0, 1] (default: 0.45)\n\n"
              << "Model & Processing Options:\n"
              << "  --encoder <path>       Image Encoder ONNX model (default: models/medsam2_image_encoder.onnx)\n"
              << "  --decoder <path>       Mask Decoder ONNX model (default: models/medsam2_mask_decoder.onnx)\n"
              << "  --device <cuda|cpu>    Compute device (default: cuda if available)\n"
              << "  --pad_to_square <bool> Center-pad to preserve aspect ratio (default: 1/true)\n"
              << "  --pad_size <int>       Target square canvas resolution (default: 1024)\n"
              << "  --box_pad <int>        Grow derived bounding boxes by N pixels (default: 0)\n"
              << "  --score_thresh <float> Logit threshold for foreground mask (default: 0.0)\n"
              << "=======================================================\n" << std::endl;
}

// Helper to parse comma/space separated box strings
bool parseBoxString(const std::string& str, int default_cid, int& out_cid, Box& out_box) {
    if (str.empty()) return false;
    std::string s = str;
    std::replace(s.begin(), s.end(), ',', ' ');
    std::replace(s.begin(), s.end(), ':', ' ');
    std::stringstream ss(s);

    std::vector<float> vals;
    float v;
    while (ss >> v) {
        vals.push_back(v);
    }

    if (vals.size() == 4) {
        out_cid = default_cid;
        out_box.x0 = vals[0];
        out_box.y0 = vals[1];
        out_box.x1 = vals[2];
        out_box.y1 = vals[3];
        return true;
    } else if (vals.size() == 5) {
        out_cid = static_cast<int>(vals[0]);
        out_box.x0 = vals[1];
        out_box.y0 = vals[2];
        out_box.x1 = vals[3];
        out_box.y1 = vals[4];
        return true;
    }
    return false;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string image_path;
    std::string prompt_mask_path;
    std::string class_ids_str;
    std::vector<std::pair<std::string, int>> raw_boxes;
    std::string boxes_str;
    std::string out_mask_path = "output_mask.png";
    std::string out_overlay_path;
    std::string encoder_path = "models/medsam2_image_encoder.onnx";
    std::string decoder_path = "models/medsam2_mask_decoder.onnx";
    std::string device_str = "cuda";
    bool pad_to_square = true;
    int pad_size = 1024;
    int box_pad = 0;
    float score_thresh = 0.0f;
    float alpha = 0.45f;

    int current_cid = 1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--image" && i + 1 < argc) image_path = argv[++i];
        else if (arg == "--prompt_mask" && i + 1 < argc) prompt_mask_path = argv[++i];
        else if (arg == "--class_ids" && i + 1 < argc) class_ids_str = argv[++i];
        else if (arg == "--box" && i + 1 < argc) {
            std::string bstr = argv[++i];
            raw_boxes.push_back({bstr, current_cid});
        }
        else if (arg == "--class_id" && i + 1 < argc) {
            current_cid = std::stoi(argv[++i]);
            if (!raw_boxes.empty()) {
                raw_boxes.back().second = current_cid;
            }
        }
        else if (arg == "--box2" && i + 1 < argc) {
            std::string bstr = argv[++i];
            raw_boxes.push_back({bstr, 2});
        }
        else if (arg == "--class_id2" && i + 1 < argc) {
            int cid2 = std::stoi(argv[++i]);
            if (!raw_boxes.empty()) {
                raw_boxes.back().second = cid2;
            }
        }
        else if (arg == "--boxes" && i + 1 < argc) boxes_str = argv[++i];
        else if (arg == "--out_mask" && i + 1 < argc) out_mask_path = argv[++i];
        else if (arg == "--out_overlay" && i + 1 < argc) out_overlay_path = argv[++i];
        else if (arg == "--alpha" && i + 1 < argc) alpha = std::stof(argv[++i]);
        else if (arg == "--encoder" && i + 1 < argc) encoder_path = argv[++i];
        else if (arg == "--decoder" && i + 1 < argc) decoder_path = argv[++i];
        else if (arg == "--device" && i + 1 < argc) device_str = argv[++i];
        else if (arg == "--pad_to_square" && i + 1 < argc) {
            std::string val = argv[++i];
            pad_to_square = (val == "1" || val == "true" || val == "True");
        }
        else if (arg == "--pad_size" && i + 1 < argc) pad_size = std::stoi(argv[++i]);
        else if (arg == "--box_pad" && i + 1 < argc) box_pad = std::stoi(argv[++i]);
        else if (arg == "--score_thresh" && i + 1 < argc) score_thresh = std::stof(argv[++i]);
        else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
    }

    if (image_path.empty()) {
        std::cerr << "Error: --image argument is required." << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    // Auto-locate models if paths do not exist directly
    auto fileExists = [](const std::string& p) {
        FILE* f = fopen(p.c_str(), "r");
        if (f) { fclose(f); return true; }
        return false;
    };

    if (!fileExists(encoder_path)) {
        if (fileExists("models/medsam2_image_encoder.onnx"))
            encoder_path = "models/medsam2_image_encoder.onnx";
        else if (fileExists("../models/medsam2_image_encoder.onnx"))
            encoder_path = "../models/medsam2_image_encoder.onnx";
        else if (fileExists("/home/suman/mapmed/MedSamCustom/ocv/models/medsam2_image_encoder.onnx"))
            encoder_path = "/home/suman/mapmed/MedSamCustom/ocv/models/medsam2_image_encoder.onnx";
    }

    if (!fileExists(decoder_path)) {
        if (fileExists("models/medsam2_mask_decoder.onnx"))
            decoder_path = "models/medsam2_mask_decoder.onnx";
        else if (fileExists("../models/medsam2_mask_decoder.onnx"))
            decoder_path = "../models/medsam2_mask_decoder.onnx";
        else if (fileExists("/home/suman/mapmed/MedSamCustom/ocv/models/medsam2_mask_decoder.onnx"))
            decoder_path = "/home/suman/mapmed/MedSamCustom/ocv/models/medsam2_mask_decoder.onnx";
    }

    // 1. Load Image (generic size, 8/16-bit, color/grayscale)
    std::cout << "[MedSAM2 OpenCV5 C++] Loading image: " << image_path << std::endl;
    cv::Mat img = loadImageRGB(image_path);
    if (img.empty()) {
        std::cerr << "Error: Failed to load image from " << image_path << std::endl;
        return 1;
    }
    std::cout << "  Original Image Dimensions: " << img.cols << " x " << img.rows << " (3 channels)" << std::endl;

    // 2. Prepare Prompts
    std::vector<Prompt> prompts;
    cv::Mat prompt_mask_img;

    // Parse user-specified class IDs if provided
    std::set<int> requested_cids;
    if (!class_ids_str.empty()) {
        std::stringstream ss(class_ids_str);
        std::string token;
        while (std::getline(ss, token, ',')) {
            if (!token.empty()) requested_cids.insert(std::stoi(token));
        }
    }

    if (!prompt_mask_path.empty()) {
        std::cout << "[MedSAM2 OpenCV5 C++] Loading prompt mask: " << prompt_mask_path << std::endl;
        prompt_mask_img = loadLabelPNG(prompt_mask_path);
        if (!prompt_mask_img.empty()) {
            // Resize prompt mask if dimension doesn't match image
            if (prompt_mask_img.cols != img.cols || prompt_mask_img.rows != img.rows) {
                std::cout << "  Notice: Resizing prompt mask from " << prompt_mask_img.cols << "x" << prompt_mask_img.rows
                          << " to match image size " << img.cols << "x" << img.rows << std::endl;
                cv::resize(prompt_mask_img, prompt_mask_img, img.size(), 0, 0, cv::INTER_NEAREST);
            }

            // Discover all unique non-zero classes in prompt mask
            std::set<uint8_t> detected_classes;
            for (int r = 0; r < prompt_mask_img.rows; ++r) {
                const uint8_t* row = prompt_mask_img.ptr<uint8_t>(r);
                for (int c = 0; c < prompt_mask_img.cols; ++c) {
                    if (row[c] > 0) {
                        if (requested_cids.empty() || requested_cids.count(row[c])) {
                            detected_classes.insert(row[c]);
                        }
                    }
                }
            }

            PadInfo pad = computePadInfo(img.cols, img.rows, pad_size, pad_to_square);

            for (uint8_t cid : detected_classes) {
                Box box;
                if (boxFromMask(prompt_mask_img, cid, box, box_pad)) {
                    Prompt p;
                    p.class_id = cid;
                    p.box = box;
                    p.has_mask = true;
                    p.mask_logits = makeMaskLogits(prompt_mask_img, cid, pad);
                    prompts.push_back(p);
                    std::cout << "  Derived prompt for class " << int(cid) << " box: ["
                              << box.x0 << ", " << box.y0 << ", " << box.x1 << ", " << box.y1 << "]" << std::endl;
                }
            }
        }
    }

    // Parse individual --box / --box2 prompts
    for (const auto& entry : raw_boxes) {
        int cid = entry.second;
        Box b;
        if (parseBoxString(entry.first, cid, cid, b)) {
            Prompt p;
            p.class_id = cid;
            p.box = b;
            p.has_mask = false;
            prompts.push_back(p);
            std::cout << "  Manual box prompt class " << cid << ": ["
                      << b.x0 << ", " << b.y0 << ", " << b.x1 << ", " << b.y1 << "]" << std::endl;
        }
    }

    // Parse batch --boxes "cid,x0,y0,x1,y1;..."
    if (!boxes_str.empty()) {
        std::stringstream ss(boxes_str);
        std::string bstr;
        while (std::getline(ss, bstr, ';')) {
            if (bstr.empty()) continue;
            int cid = 1;
            Box b;
            if (parseBoxString(bstr, cid, cid, b)) {
                Prompt p;
                p.class_id = cid;
                p.box = b;
                p.has_mask = false;
                prompts.push_back(p);
                std::cout << "  Batch box prompt class " << cid << ": ["
                          << b.x0 << ", " << b.y0 << ", " << b.x1 << ", " << b.y1 << "]" << std::endl;
            }
        }
    }

    if (prompts.empty()) {
        std::cerr << "Error: No valid prompts provided! Provide --prompt_mask or --box \"x0,y0,x1,y1\"" << std::endl;
        return 1;
    }

    // 3. Initialize MedSAM2 OpenCV 5 Engine
    MedSAM2Engine engine(encoder_path, decoder_path, device_str);

    // 4. Run Segmentation (handles arbitrary size -> outputs exact original dimensions)
    auto start_time = std::chrono::high_resolution_clock::now();
    SegmentationResult result = engine.segment(img, prompts, pad_to_square, pad_size, score_thresh);
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    std::cout << "[MedSAM2 OpenCV5 C++] Total inference time: " << total_ms << " ms" << std::endl;

    // Verify output dimension matches original image
    if (result.width != img.cols || result.height != img.rows ||
        result.label_map.cols != img.cols || result.label_map.rows != img.rows) {
        std::cerr << "Warning: Output size (" << result.width << "x" << result.height
                  << ") does not match original image (" << img.cols << "x" << img.rows << ")!" << std::endl;
    }

    // 5. Save Output Mask
    if (saveMaskPNG(out_mask_path, result.label_map)) {
        std::cout << "[MedSAM2 OpenCV5 C++] Saved refined mask to: " << out_mask_path
                  << " (size: " << result.label_map.cols << " x " << result.label_map.rows << ")" << std::endl;
    } else {
        std::cerr << "Error: Failed to save output mask to: " << out_mask_path << std::endl;
    }

    // 6. Save Output Overlay
    if (!out_overlay_path.empty()) {
        if (saveOverlayPNG(out_overlay_path, img, result.label_map, alpha)) {
            std::cout << "[MedSAM2 OpenCV5 C++] Saved overlay visualization to: " << out_overlay_path
                      << " (size: " << img.cols << " x " << img.rows << ")" << std::endl;
        } else {
            std::cerr << "Error: Failed to save overlay to: " << out_overlay_path << std::endl;
        }
    }

    // 7. Compute & Print Metrics if Prompt Mask was provided
    if (!prompt_mask_img.empty() && prompt_mask_img.cols == result.width && prompt_mask_img.rows == result.height) {
        std::cout << "\n=======================================================" << std::endl;
        std::cout << "Segmentation Evaluation against Prompt Mask (" << result.width << " x " << result.height << "):" << std::endl;

        std::set<uint8_t> eval_cids;
        for (const auto& p : prompts) {
            eval_cids.insert(static_cast<uint8_t>(p.class_id));
        }

        std::vector<float> dices;
        std::vector<float> ious;
        for (uint8_t cid : eval_cids) {
            float d = computeDice(result.label_map, prompt_mask_img, cid);
            float u = computeIoU(result.label_map, prompt_mask_img, cid);
            dices.push_back(d);
            ious.push_back(u);
            std::string name = (cid == 1 ? "Femur" : (cid == 2 ? "Tibia" : ("Class " + std::to_string(cid))));
            std::cout << "  Class " << int(cid) << " (" << name << "):"
                      << "  Dice = " << std::fixed << std::setprecision(4) << d
                      << ", IoU = " << u << std::endl;
        }
        if (!dices.empty()) {
            float mean_d = 0.0f, mean_u = 0.0f;
            for (size_t i = 0; i < dices.size(); ++i) {
                mean_d += dices[i];
                mean_u += ious[i];
            }
            mean_d /= static_cast<float>(dices.size());
            mean_u /= static_cast<float>(ious.size());
            std::cout << "  --------------------------------------------------" << std::endl;
            std::cout << "  Overall Mean: Dice = " << mean_d << ", IoU = " << mean_u << std::endl;
        }
        std::cout << "=======================================================\n" << std::endl;
    }

    std::cout << "Done." << std::endl;
    return 0;
}
