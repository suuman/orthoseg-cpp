#ifndef XRAY_OCV_SEGMENTOR_HPP
#define XRAY_OCV_SEGMENTOR_HPP

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/cuda.hpp>

namespace xray_ocv {

constexpr int LABEL_BACKGROUND = 0;
constexpr int LABEL_FEMUR = 1;
constexpr int LABEL_TIBIA = 2;

constexpr int TARGET_HEIGHT = 2048;
constexpr int TILE_WIDTH = 768;

/**
 * @brief Structure holding inference outputs and original image data.
 */
struct InferenceResult {
    cv::Mat label_mask;        // (orig_H, orig_W) CV_8UC1 with discrete values {0, 1, 2}
    cv::Mat overlay_bgr;       // (orig_H, orig_W) CV_8UC3 alpha-blended overlay on original image
    cv::Mat color_mask_bgr;    // (orig_H, orig_W) CV_8UC3 colored visual mask (Femur: Green, Tibia: Red)
    cv::Mat original_bgr;      // (orig_H, orig_W) CV_8UC3 original image in BGR
    cv::Mat original_gray;     // (orig_H, orig_W) CV_8UC1 original grayscale image

    int orig_height = 0;
    int orig_width = 0;
    int infer_height = 0;
    int infer_width = 0;
    std::string input_path;

    /**
     * @brief Save inference results to disk.
     */
    void save(const std::string& output_dir,
              const std::string& base_name = "",
              bool save_color_mask = true,
              bool save_stacked = false) const;
};

/**
 * @brief Pure OpenCV 5 Inference Engine for 2D X-ray Bone Segmentation.
 * No PyTorch or LibTorch dependencies.
 * Leverages native hardware acceleration:
 *   - NVIDIA RTX 4050 GPU via CUDA & cuDNN (FP16 / FP32)
 *   - AMD Ryzen 9 HX CPU via multi-threaded TBB & AVX/FMA MLAS
 */
class XRaySegmentor {
public:
    enum class AccelerationMode {
        CUDA_FP16,   // RTX 4050 Tensor Cores (fastest, lowest memory)
        CUDA_FP32,   // RTX 4050 CUDA cores standard FP32
        CPU          // Ryzen 9 multi-threaded TBB / AVX
    };

    /**
     * @brief Constructor
     * @param onnx_model_path Path to exported ONNX model file.
     * @param mode Hardware acceleration mode (CUDA_FP16, CUDA_FP32, or CPU).
     * @param alpha Blending alpha factor for overlay (0.0 to 1.0).
     * @param draw_contours Whether to draw boundary contours on overlay.
     */
    explicit XRaySegmentor(const std::string& onnx_model_path,
                           AccelerationMode mode = AccelerationMode::CUDA_FP16,
                           float alpha = 0.4f,
                           bool draw_contours = true);

    /**
     * @brief Factory constructor using string identifier ("auto", "cuda", "cuda_fp16", "cpu")
     */
    static std::unique_ptr<XRaySegmentor> create(const std::string& onnx_model_path,
                                                  const std::string& device_str = "auto",
                                                  float alpha = 0.4f,
                                                  bool draw_contours = true);

    /**
     * @brief Run inference on an image from disk.
     */
    InferenceResult predict(const std::string& image_path);

    /**
     * @brief Run inference on an in-memory cv::Mat image.
     */
    InferenceResult predict(const cv::Mat& input_image, const std::string& input_name = "");

    /**
     * @brief Predict and save outputs to files/directory.
     */
    InferenceResult predict_and_save(const std::string& image_path,
                                    const std::string& output_dir,
                                    const std::string& output_label = "",
                                    const std::string& output_overlay = "",
                                    const std::string& base_name = "");

    // Getters and setters
    float get_alpha() const { return alpha_; }
    void set_alpha(float a) { alpha_ = a; }
    bool get_draw_contours() const { return draw_contours_; }
    void set_draw_contours(bool d) { draw_contours_ = d; }
    AccelerationMode get_mode() const { return mode_; }

private:
    cv::dnn::Net net_;
    AccelerationMode mode_;
    float alpha_;
    bool draw_contours_;

    void preprocess(const cv::Mat& orig_gray,
                    cv::Mat& img_infer,
                    int& infer_w,
                    int& infer_h,
                    float& scale) const;

    cv::Mat run_sliding_window(const cv::Mat& img_infer, int infer_w, int infer_h);

    void create_visualization(const cv::Mat& orig_bgr,
                              const cv::Mat& label_mask,
                              cv::Mat& overlay_bgr,
                              cv::Mat& color_mask_bgr) const;
};

} // namespace xray_ocv

#endif // XRAY_OCV_SEGMENTOR_HPP
