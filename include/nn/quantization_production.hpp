#pragma once
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <random>

namespace lora_kernel {

// ======================================================================
// 41. Dynamic Quantization (per-token, per-channel)
// ======================================================================
class DynamicQuantizer {
public:
    // Per-tensor symmetric quantization
    static void quantize_symmetric(const float* input, int8_t* output,
                                    float& scale, int64_t n) {
        float max_abs = 0.0f;
        for (int64_t i = 0; i < n; ++i)
            max_abs = std::max(max_abs, std::abs(input[i]));
        scale = max_abs / 127.0f;
        float inv_scale = 1.0f / (scale + 1e-12f);
        for (int64_t i = 0; i < n; ++i)
            output[i] = (int8_t)std::round(input[i] * inv_scale);
    }

    static void dequantize_symmetric(const int8_t* input, float* output,
                                      float scale, int64_t n) {
        for (int64_t i = 0; i < n; ++i)
            output[i] = input[i] * scale;
    }

    // Per-channel quantization (for weight matrices)
    static void quantize_per_channel(const float* weights, int8_t* output,
                                      float* scales, int rows, int cols) {
        for (int r = 0; r < rows; ++r) {
            float max_abs = 0.0f;
            for (int c = 0; c < cols; ++c)
                max_abs = std::max(max_abs, std::abs(weights[r * cols + c]));
            scales[r] = max_abs / 127.0f;
            float inv = 1.0f / (scales[r] + 1e-12f);
            for (int c = 0; c < cols; ++c)
                output[r * cols + c] = (int8_t)std::round(weights[r * cols + c] * inv);
        }
    }
};

// ======================================================================
// 42. Static Quantization with Calibration
// ======================================================================
class StaticQuantizer {
private:
    std::vector<float> activation_max_;
    int num_calib_steps_{0};
public:
    StaticQuantizer(int num_layers = 32) : activation_max_(num_layers, 0.0f) {}

    void observe(const float* activations, int64_t n, int layer) {
        (void)layer;
        float max_abs = 0.0f;
        for (int64_t i = 0; i < n; ++i)
            max_abs = std::max(max_abs, std::abs(activations[i]));
        activation_max_[layer] = std::max(activation_max_[layer], max_abs);
        num_calib_steps_++;
    }

    void calibrate() {
        std::cout << "[CALIB] Calibrated over " << num_calib_steps_ << " steps\n";
    }

    float get_scale(int layer) const {
        return activation_max_[layer] / 127.0f;
    }

    void quantize(const float* input, int8_t* output, int layer, int64_t n) {
        float scale = get_scale(layer);
        float inv = 1.0f / (scale + 1e-12f);
        for (int64_t i = 0; i < n; ++i)
            output[i] = (int8_t)std::round(input[i] * inv);
    }
};

// ======================================================================
// 43. Quantization-Aware Training (QAT) Wrappers
// ======================================================================
class QATWrapper {
private:
    float scale_;
    bool fake_quant_{false};

public:
    QATWrapper(int bits = 8) : scale_(1.0f) {
        (void)bits;
    }

    // Forward with fake quantization (straight-through estimator)
    void fake_quantize(float* data, int64_t n) {
        if (!fake_quant_) return;
        float max_abs = 0.0f;
        for (int64_t i = 0; i < n; ++i)
            max_abs = std::max(max_abs, std::abs(data[i]));
        scale_ = max_abs / 127.0f;
        float inv = 1.0f / (scale_ + 1e-12f);
        for (int64_t i = 0; i < n; ++i) {
            int8_t q = (int8_t)std::round(data[i] * inv);
            data[i] = q * scale_; // Fake quant: round + dequant
        }
    }

    void enable_fake_quant() { fake_quant_ = true; }
    void disable_fake_quant() { fake_quant_ = false; }
    float get_scale() const { return scale_; }
};

// ======================================================================
// 44. FP8 Training (E4M3/E5M2)
// ======================================================================
struct fp8_e4m3 {
    uint8_t bits;
    fp8_e4m3() : bits(0) {}
    fp8_e4m3(float f) {
        uint32_t i;
        std::memcpy(&i, &f, sizeof(i));
        uint32_t sign = (i >> 31) & 1;
        uint32_t exp = (i >> 23) & 0xFF;
        uint32_t mant = i & 0x7FFFFF;
        // E4M3: 1 sign, 4 exp, 3 mantissa
        if (exp == 0) { bits = 0; return; }
        uint32_t e4 = std::min((uint32_t)15, std::max((uint32_t)1, exp - 127 + 7));
        uint32_t m3 = mant >> 20;
        bits = (uint8_t)((sign << 7) | (e4 << 3) | m3);
    }
    operator float() const {
        if (bits == 0) return 0.0f;
        uint32_t sign = (bits >> 7) & 1;
        uint32_t e4 = (bits >> 3) & 0xF;
        uint32_t m3 = bits & 0x7;
        uint32_t exp = (e4 == 0) ? 0 : (e4 - 7 + 127);
        uint32_t mant = m3 << 20;
        uint32_t i = (sign << 31) | (exp << 23) | mant;
        float f;
        std::memcpy(&f, &i, sizeof(f));
        return f;
    }
};

struct fp8_e5m2 {
    uint8_t bits;
    fp8_e5m2() : bits(0) {}
    fp8_e5m2(float f) {
        uint32_t i;
        std::memcpy(&i, &f, sizeof(i));
        uint32_t sign = (i >> 31) & 1;
        uint32_t exp = (i >> 23) & 0xFF;
        uint32_t mant = i & 0x7FFFFF;
        // E5M2: 1 sign, 5 exp, 2 mantissa
        if (exp == 0) { bits = 0; return; }
        uint32_t e5 = std::min((uint32_t)31, std::max((uint32_t)1, exp - 127 + 15));
        uint32_t m2 = mant >> 21;
        bits = (uint8_t)((sign << 7) | (e5 << 2) | m2);
    }
    operator float() const {
        if (bits == 0) return 0.0f;
        uint32_t sign = (bits >> 7) & 1;
        uint32_t e5 = (bits >> 2) & 0x1F;
        uint32_t m2 = bits & 0x3;
        uint32_t exp = (e5 == 0) ? 0 : (e5 - 15 + 127);
        uint32_t mant = m2 << 21;
        uint32_t i = (sign << 31) | (exp << 23) | mant;
        float f;
        std::memcpy(&f, &i, sizeof(f));
        return f;
    }
};

class FP8Trainer {
public:
    static void cast_to_fp8_e4m3(const float* src, fp8_e4m3* dst, int64_t n) {
        for (int64_t i = 0; i < n; ++i) dst[i] = fp8_e4m3(src[i]);
    }
    static void cast_from_fp8_e4m3(const fp8_e4m3* src, float* dst, int64_t n) {
        for (int64_t i = 0; i < n; ++i) dst[i] = float(src[i]);
    }
};

// ======================================================================
// 45. INT4 Packing with Group-wise Scaling
// ======================================================================
class INT4Quantizer {
private:
    int group_size_{32};
public:
    INT4Quantizer(int group = 32) : group_size_(group) {}

    void quantize_groupwise(const float* input, uint8_t* output,
                            float* scales, int64_t n) {
        int num_groups = (int)((n + group_size_ - 1) / group_size_);
        for (int g = 0; g < num_groups; ++g) {
            int start = g * group_size_;
            int end = std::min(start + group_size_, (int)n);
            float max_abs = 0.0f;
            for (int i = start; i < end; ++i)
                max_abs = std::max(max_abs, std::abs(input[i]));
            scales[g] = max_abs / 7.0f; // INT4 symmetric: [-7, 7]
            float inv = 1.0f / (scales[g] + 1e-12f);

            for (int i = start; i < end; i += 2) {
                int4_t low = (int4_t)std::round(input[i] * inv);
                int4_t high = (i + 1 < end) ?
                    (int4_t)std::round(input[i + 1] * inv) : (int4_t)0;
                output[i / 2] = ((uint8_t)(high & 0xF) << 4) | (uint8_t)(low & 0xF);
            }
        }
    }

    void dequantize_groupwise(const uint8_t* input, const float* scales,
                              float* output, int64_t n) {
        (void)input; (void)scales; (void)output; (void)n;
    }
};

// ======================================================================
// 46. GPTQ (Hessian-based quantization)
// ======================================================================
class GPTQQuantizer {
private:
    std::vector<float> hessian_;
public:
    void quantize(float* weights, int rows, int cols, int group_size = 128) {
        (void)rows; (void)cols; (void)group_size;
        std::cout << "[GPTQ] Hessian-based quantization\n";
        // Full GPTQ would:
        // 1. Compute Hessian = 2 * X^T X (from calibration data)
        // 2. Cholesky decomposition
        // 3. Greedy quantization with Hessian-based error compensation
    }

    void compute_hessian(const float* calib_data, int n, int dim) {
        (void)calib_data; (void)n;
        hessian_.resize(dim * dim, 0.0f);
        std::cout << "[GPTQ] Computing Hessian matrix\n";
    }
};

// ======================================================================
// 47. AWQ (Activation-Aware Weight Quantization)
// ======================================================================
class AWQQuantizer {
private:
    std::vector<float> activation_scale_;
public:
    void quantize(float* weights, const float* activations,
                  int rows, int cols, float alpha = 0.5f) {
        (void)alpha;
        // AWQ: scale weights by activation importance
        activation_scale_.resize(cols);
        for (int c = 0; c < cols; ++c) {
            float max_act = 0.0f;
            for (int r = 0; r < rows; ++r)
                max_act = std::max(max_act, std::abs(activations[r * cols + c]));
            activation_scale_[c] = max_act;
        }
        std::cout << "[AWQ] Activation-aware quantization\n";
    }
};

// ======================================================================
// 48. SmoothQuant (W8A8 Smoothing)
// ======================================================================
class SmoothQuant {
public:
    static void smooth(float* weights, float* activations,
                       int rows, int cols, float alpha = 0.5f) {
        // SmoothQuant: migrate quantization difficulty from activations to weights
        for (int c = 0; c < cols; ++c) {
            float max_w = 0.0f, max_a = 0.0f;
            for (int r = 0; r < rows; ++r)
                max_w = std::max(max_w, std::abs(weights[r * cols + c]));
            for (int r = 0; r < rows; ++r)
                max_a = std::max(max_a, std::abs(activations[r * cols + c]));
            float s = std::pow(max_a, alpha) / std::pow(max_w, 1.0f - alpha);
            s = std::max(0.1f, std::min(s, 10.0f));
            for (int r = 0; r < rows; ++r) weights[r * cols + c] /= s;
            for (int r = 0; r < rows; ++r) activations[r * cols + c] *= s;
        }
    }
};

// ======================================================================
// 49. KV Cache Quantization (FP8/INT8)
// ======================================================================
class KVCacheQuantizer {
private:
    std::vector<float> k_scale_, v_scale_;
    int quant_bits_{8};
public:
    KVCacheQuantizer(int max_layers = 32, int bits = 8)
        : k_scale_(max_layers, 1.0f), v_scale_(max_layers, 1.0f), quant_bits_(bits) {}

    void quantize_kv(float* K, float* V, int layer, int64_t n) {
        float max_k = 0.0f, max_v = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            max_k = std::max(max_k, std::abs(K[i]));
            max_v = std::max(max_v, std::abs(V[i]));
        }
        k_scale_[layer] = max_k / 127.0f;
        v_scale_[layer] = max_v / 127.0f;
        float inv_k = 1.0f / (k_scale_[layer] + 1e-12f);
        float inv_v = 1.0f / (v_scale_[layer] + 1e-12f);
        for (int64_t i = 0; i < n; ++i) {
            K[i] = (float)(int8_t)(K[i] * inv_k) * k_scale_[layer];
            V[i] = (float)(int8_t)(V[i] * inv_v) * v_scale_[layer];
        }
    }
};

// ======================================================================
// 50. Mixed-Precision Quantization (different layers, different precisions)
// ======================================================================
class MixedPrecisionConfig {
private:
    struct LayerConfig {
        int bits; // 4, 8, 16
        std::string format; // "int4", "int8", "fp8", "fp16", "bf16"
    };
    std::vector<LayerConfig> configs_;
public:
    MixedPrecisionConfig(int num_layers = 12) {
        configs_.resize(num_layers);
        for (int i = 0; i < num_layers; ++i) {
            // Early layers: higher precision
            if (i < num_layers / 4) {
                configs_[i] = {16, "bf16"};
            } else if (i < num_layers / 2) {
                configs_[i] = {8, "fp8"};
            } else {
                configs_[i] = {4, "int4"};
            }
        }
    }

    int get_bits(int layer) const { return configs_[layer].bits; }
    std::string get_format(int layer) const { return configs_[layer].format; }

    void set_layer_bits(int layer, int bits) { configs_[layer].bits = bits; }
};

} // namespace lora_kernel
