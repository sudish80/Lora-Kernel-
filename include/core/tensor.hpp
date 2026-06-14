#pragma once
#include <vector>
#include <cstring>
#include <stdexcept>
#include <cstdint>
#include <cstddef>
#include <numeric>
#include <cmath>
#include <random>
#include <algorithm>
#include <array>
#include <mutex>
#include <iostream>
#include <fstream>
#include "include/core/gemm_engine.hpp"
#include "include/core/memory_manager.hpp"

#ifdef _WIN32
#include <intrin.h>
#endif

namespace lora_kernel {

// ============================================================
// CPUFeatures - runtime CPU feature detection singleton
// ============================================================
struct CPUFeatures {
    bool avx  : 1;
    bool avx2 : 1;
    bool avx512f : 1;
    bool avx512dq : 1;
    bool avx512bw : 1;
    bool avx512vl : 1;
    bool neon : 1;
    bool sve  : 1;

    static CPUFeatures detect() {
        CPUFeatures f = {};

#ifdef _WIN32
        int cpuInfo[4] = {};
        __cpuidex(cpuInfo, 0, 0);
        int maxLeaf = cpuInfo[0];

        if (maxLeaf >= 1) {
            __cpuidex(cpuInfo, 1, 0);
            f.avx = (cpuInfo[2] & (1 << 28)) != 0;
        }

        if (maxLeaf >= 7) {
            __cpuidex(cpuInfo, 7, 0);
            f.avx2 = (cpuInfo[1] & (1 << 5)) != 0;
            f.avx512f  = (cpuInfo[1] & (1 << 16)) != 0;
            f.avx512dq = (cpuInfo[1] & (1 << 17)) != 0;
            f.avx512bw = (cpuInfo[1] & (1 << 30)) != 0;
            f.avx512vl = (cpuInfo[1] & (1 << 31)) != 0;
        }
#elif defined(__linux__)
        std::ifstream cpuinfo("/proc/cpuinfo");
        if (cpuinfo.is_open()) {
            std::string line;
            while (std::getline(cpuinfo, line)) {
                if (line.find("flags") != std::string::npos || line.find("Features") != std::string::npos) {
                    if (line.find("avx")     != std::string::npos) f.avx = true;
                    if (line.find("avx2")    != std::string::npos) f.avx2 = true;
                    if (line.find("avx512f") != std::string::npos) f.avx512f = true;
                    if (line.find("avx512dq")!= std::string::npos) f.avx512dq = true;
                    if (line.find("avx512bw")!= std::string::npos) f.avx512bw = true;
                    if (line.find("avx512vl")!= std::string::npos) f.avx512vl = true;
                    if (line.find("neon")    != std::string::npos) f.neon = true;
                    if (line.find("sve")     != std::string::npos) f.sve = true;
                }
            }
        }
#elif defined(__GNUC__) || defined(__clang__)
        f.avx = __builtin_cpu_supports("avx");
        f.avx2 = __builtin_cpu_supports("avx2");
        f.avx512f  = __builtin_cpu_supports("avx512f");
        f.avx512dq = __builtin_cpu_supports("avx512dq");
        f.avx512bw = __builtin_cpu_supports("avx512bw");
        f.avx512vl = __builtin_cpu_supports("avx512vl");
#endif
        return f;
    }

    static const CPUFeatures& instance() {
        static CPUFeatures features = detect();
        return features;
    }
};

inline static bool has_avx()    { return CPUFeatures::instance().avx; }
inline static bool has_avx2()   { return CPUFeatures::instance().avx2; }
inline static bool has_avx512() { return CPUFeatures::instance().avx512f; }
inline static bool has_neon()   { return CPUFeatures::instance().neon; }
inline static bool has_sve()    { return CPUFeatures::instance().sve; }
inline static const CPUFeatures& cpu_features() { return CPUFeatures::instance(); }

enum class DType { F32, F16, BF16 };

class Tensor {
private:
    float* data_{nullptr};
    size_t num_elements_{0};
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    DType dtype_;
    bool requires_grad_{false};
    bool owns_data_{true};

    inline static MemoryManager* mem_mgr_ = nullptr;

    void allocate_storage(size_t n) {
        num_elements_ = n;
        if (n == 0) { data_ = nullptr; return; }
        if (mem_mgr_) {
            data_ = static_cast<float*>(mem_mgr_->allocate(n * sizeof(float), "tensor"));
        } else {
            data_ = new float[n]();
        }
        owns_data_ = true;
    }

    void deallocate_storage() {
        if (data_ && owns_data_) {
            if (mem_mgr_) {
                mem_mgr_->deallocate(data_);
            } else {
                delete[] data_;
            }
        }
        data_ = nullptr;
        num_elements_ = 0;
    }

    void compute_strides() {
        strides_.resize(shape_.size());
        if (shape_.empty()) return;
        strides_.back() = 1;
        for (int i = (int)shape_.size() - 2; i >= 0; --i)
            strides_[i] = strides_[i + 1] * shape_[i + 1];
    }

public:
    static void set_memory_manager(MemoryManager* mgr) { mem_mgr_ = mgr; }
    static MemoryManager* get_memory_manager() { return mem_mgr_; }

    Tensor() : dtype_(DType::F32) {}
    Tensor(const std::vector<int64_t>& shape, DType dtype = DType::F32)
        : shape_(shape), dtype_(dtype) {
        compute_strides();
        allocate_storage(static_cast<size_t>(numel()));
    }
    Tensor(const std::vector<int64_t>& shape, float init_val, DType dtype = DType::F32)
        : shape_(shape), dtype_(dtype) {
        compute_strides();
        allocate_storage(static_cast<size_t>(numel()));
        if (data_) std::fill(data_, data_ + num_elements_, init_val);
    }

    Tensor(const Tensor& other)
        : shape_(other.shape_), strides_(other.strides_),
          dtype_(other.dtype_), requires_grad_(other.requires_grad_) {
        allocate_storage(other.num_elements_);
        if (data_ && other.data_)
            std::memcpy(data_, other.data_, num_elements_ * sizeof(float));
    }

    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            deallocate_storage();
            shape_ = other.shape_;
            strides_ = other.strides_;
            dtype_ = other.dtype_;
            requires_grad_ = other.requires_grad_;
            allocate_storage(other.num_elements_);
            if (data_ && other.data_)
                std::memcpy(data_, other.data_, num_elements_ * sizeof(float));
        }
        return *this;
    }

    Tensor(Tensor&& other) noexcept
        : data_(other.data_), num_elements_(other.num_elements_),
          shape_(std::move(other.shape_)), strides_(std::move(other.strides_)),
          dtype_(other.dtype_), requires_grad_(other.requires_grad_),
          owns_data_(other.owns_data_) {
        other.data_ = nullptr;
        other.num_elements_ = 0;
        other.owns_data_ = true;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            deallocate_storage();
            data_ = other.data_;
            num_elements_ = other.num_elements_;
            shape_ = std::move(other.shape_);
            strides_ = std::move(other.strides_);
            dtype_ = other.dtype_;
            requires_grad_ = other.requires_grad_;
            owns_data_ = other.owns_data_;
            other.data_ = nullptr;
            other.num_elements_ = 0;
            other.owns_data_ = true;
        }
        return *this;
    }

    ~Tensor() { deallocate_storage(); }

    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool v) { requires_grad_ = v; }

    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }
    DType dtype() const { return dtype_; }
    int64_t ndim() const { return (int64_t)shape_.size(); }
    int64_t numel() const {
        if (shape_.empty()) return 0;
        return std::accumulate(shape_.begin(), shape_.end(),
                               1LL, std::multiplies<int64_t>());
    }
    int64_t size(int64_t dim) const {
        if (dim < 0) dim += ndim();
        return shape_[dim];
    }

    float* data() { return data_; }
    const float* data() const { return data_; }

    int64_t offset(const std::vector<int64_t>& indices) const {
        if (indices.size() != shape_.size())
            throw std::runtime_error("offset: dimension mismatch");
        int64_t off = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            off += indices[i] * strides_[i];
        }
        return off;
    }
    float& at(const std::vector<int64_t>& indices) { return data_[offset(indices)]; }
    const float& at(const std::vector<int64_t>& indices) const { return data_[offset(indices)]; }

    float& operator[](int64_t i) { return data_[i]; }
    const float& operator[](int64_t i) const { return data_[i]; }

    void reshape(const std::vector<int64_t>& new_shape) {
        int64_t n = 1;
        for (auto s : new_shape) n *= s;
        if (n != (int64_t)num_elements_) {
            deallocate_storage();
            shape_ = new_shape;
            compute_strides();
            allocate_storage((size_t)n);
        } else {
            shape_ = new_shape;
            strides_.resize(new_shape.size());
            if (!strides_.empty()) {
                strides_.back() = 1;
                for (int i = (int)new_shape.size() - 2; i >= 0; --i)
                    strides_[i] = strides_[i + 1] * new_shape[i + 1];
            }
        }
    }

    Tensor transpose() const {
        if (ndim() != 2)
            throw std::runtime_error("transpose: expected 2D tensor, got " +
                                     std::to_string(ndim()) + "D");
        Tensor result({shape_[1], shape_[0]}, dtype_);
        for (int64_t i = 0; i < shape_[0]; ++i)
            for (int64_t j = 0; j < shape_[1]; ++j)
                result.at({j, i}) = at({i, j});
        return result;
    }

    Tensor view(const std::vector<int64_t>& new_shape) const {
        Tensor t;
        t.shape_ = new_shape;
        t.strides_.resize(new_shape.size());
        t.strides_.back() = 1;
        for (int i = (int)new_shape.size() - 2; i >= 0; --i)
            t.strides_[i] = t.strides_[i + 1] * new_shape[i + 1];
        t.dtype_ = dtype_;
        t.data_ = data_;
        t.num_elements_ = num_elements_;
        t.owns_data_ = false;
        return t;
    }

    void fill(float v) { std::fill(data_, data_ + num_elements_, v); }
    void zeros() { fill(0.0f); }
    void ones() { fill(1.0f); }

    void normal(float mean = 0.0f, float stddev = 1.0f) {
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(mean, stddev);
        for (size_t i = 0; i < num_elements_; ++i) data_[i] = dist(gen);
    }
    void uniform(float low = 0.0f, float high = 1.0f) {
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(low, high);
        for (size_t i = 0; i < num_elements_; ++i) data_[i] = dist(gen);
    }

    void copy_from(const float* src, int64_t n) {
        if (n > (int64_t)num_elements_)
            throw std::runtime_error("copy_from: " + std::to_string(n) +
                                     " exceeds capacity " + std::to_string(num_elements_));
        std::memcpy(data_, src, (size_t)n * sizeof(float));
    }
    void copy_to(float* dst, int64_t n) const {
        if (n > (int64_t)num_elements_)
            throw std::runtime_error("copy_to: " + std::to_string(n) +
                                     " exceeds capacity " + std::to_string(num_elements_));
        std::memcpy(dst, data_, (size_t)n * sizeof(float));
    }

    void add(const Tensor& other) {
        if (num_elements_ != other.num_elements_)
            throw std::runtime_error("add: size mismatch " +
                                     std::to_string(num_elements_) + " vs " +
                                     std::to_string(other.num_elements_));
        for (size_t i = 0; i < num_elements_; ++i) data_[i] += other.data_[i];
    }
    void mul(const Tensor& other) {
        if (num_elements_ != other.num_elements_)
            throw std::runtime_error("mul: size mismatch " +
                                     std::to_string(num_elements_) + " vs " +
                                     std::to_string(other.num_elements_));
        for (size_t i = 0; i < num_elements_; ++i) data_[i] *= other.data_[i];
    }

    void print_shape() const {
        std::cout << "[";
        for (size_t i = 0; i < shape_.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << shape_[i];
        }
        std::cout << "]\n";
    }
};

inline void matmul(const Tensor& A, const Tensor& B, Tensor& C) {
    int64_t M = A.size(0), K = A.size(1), N = B.size(1);
    if (A.size(1) != B.size(0))
        throw std::runtime_error("matmul: dimension mismatch A[" + std::to_string(A.size(0)) + "," +
                                 std::to_string(A.size(1)) + "] B[" + std::to_string(B.size(0)) + "," +
                                 std::to_string(B.size(1)) + "]");
    if (C.size(0) != M || C.size(1) != N)
        throw std::runtime_error("matmul: output dimension mismatch");
    GemmEngine::matmul(A.data(), B.data(), C.data(), (int)M, (int)N, (int)K);
}

inline void batched_matmul(const Tensor& A, const Tensor& B, Tensor& C,
                            int64_t batch) {
    int64_t M = A.size(1), K = A.size(2), N = B.size(2);
    if (A.size(0) != batch || B.size(0) != batch)
        throw std::runtime_error("batched_matmul: batch dimension mismatch");
    if (A.size(2) != B.size(1))
        throw std::runtime_error("batched_matmul: inner dimension mismatch");
    GemmEngine::batched_gemm(A.data(), B.data(), C.data(), (int)batch, (int)M, (int)N, (int)K);
}

inline void softmax_inplace(Tensor& X) {
    int64_t rows = X.numel() / X.size(X.ndim() - 1);
    int64_t cols = X.size(X.ndim() - 1);
    for (int64_t r = 0; r < rows; ++r) {
        float* row = X.data() + r * cols;
        float max_val = *std::max_element(row, row + cols);
        double sum = 0.0;
        for (int64_t c = 0; c < cols; ++c)
            sum += std::exp((double)(row[c] - max_val));
        double inv_sum = 1.0 / (sum + 1e-12);
        for (int64_t c = 0; c < cols; ++c)
            row[c] = (float)(std::exp((double)(row[c] - max_val)) * inv_sum);
    }
}

inline void softmax_backward_inplace(const Tensor& softmax_out, Tensor& grad) {
    int64_t rows = softmax_out.numel() / softmax_out.size(softmax_out.ndim() - 1);
    int64_t cols = softmax_out.size(softmax_out.ndim() - 1);
    for (int64_t r = 0; r < rows; ++r) {
        const float* sm_row = softmax_out.data() + r * cols;
        float* grad_row = grad.data() + r * cols;
        double dot = 0.0;
        for (int64_t c = 0; c < cols; ++c)
            dot += (double)sm_row[c] * grad_row[c];
        for (int64_t c = 0; c < cols; ++c)
            grad_row[c] = sm_row[c] * (grad_row[c] - (float)dot);
    }
}

inline void rmsnorm_forward(const float* x, float* y, int n, const float* weight, float eps) {
    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) sum_sq += (double)x[i] * x[i];
    float rms = (float)(std::sqrt(sum_sq / n + eps));
    float inv_rms = 1.0f / rms;
    for (int i = 0; i < n; ++i) y[i] = (x[i] * inv_rms) * weight[i];
}

inline void rmsnorm_backward(const float* grad_output, const float* x, float* grad_x,
                              const float* weight, int n, float eps) {
    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) sum_sq += (double)x[i] * x[i];
    float rms = (float)(std::sqrt(sum_sq / n + eps));
    float inv_rms = 1.0f / rms;
    double sum_gx = 0.0;
    for (int i = 0; i < n; ++i)
        sum_gx += (double)grad_output[i] * x[i];
    float scale = (float)(sum_gx / (sum_sq + eps * n));
    for (int i = 0; i < n; ++i)
        grad_x[i] = inv_rms * (grad_output[i] - x[i] * scale) * weight[i];
}

inline void gelu_forward(const float* x, float* y, int n) {
    const float sqrt_2_over_pi = 0.7978845608f;
    for (int i = 0; i < n; ++i) {
        float c = x[i] * sqrt_2_over_pi;
        float x3 = x[i] * x[i] * x[i];
        y[i] = 0.5f * x[i] * (1.0f + std::tanh(c * (1.0f + 0.044715f * x3)));
    }
}

inline void gelu_backward(const float* grad_output, const float* x, float* grad_x, int n) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    for (int i = 0; i < n; ++i) {
        float xv = x[i];
        float x3 = xv * xv * xv;
        float tanh_arg = sqrt_2_over_pi * (xv + coeff * x3);
        float tanh_val = std::tanh(tanh_arg);
        float sech2 = 1.0f - tanh_val * tanh_val;
        float dtanh = sqrt_2_over_pi * (1.0f + 3.0f * coeff * xv * xv) * sech2;
        float dgelu = 0.5f * (1.0f + tanh_val) + 0.5f * xv * dtanh;
        grad_x[i] = grad_output[i] * dgelu;
    }
}

inline void swiglu_forward(const float* x, const float* y, float* out, int n) {
    for (int i = 0; i < n; ++i) {
        float sig = 1.0f / (1.0f + std::exp(-x[i]));
        float swish = x[i] * sig;
        out[i] = swish * y[i];
    }
}

inline void swiglu_backward(const float* grad_out, const float* x, const float* y,
                            float* grad_x, float* grad_y, int n) {
    for (int i = 0; i < n; ++i) {
        float sig = 1.0f / (1.0f + std::exp(-x[i]));
        float swish = x[i] * sig;
        float dsig = sig * (1.0f - sig);
        float dswish = sig + x[i] * dsig;
        grad_x[i] = grad_out[i] * y[i] * dswish;
        grad_y[i] = grad_out[i] * swish;
    }
}

inline void apply_causal_mask(float* scores, int seq_len) {
    for (int i = 0; i < seq_len; ++i)
        for (int j = i + 1; j < seq_len; ++j)
            scores[i * seq_len + j] = -1e9f;
}

inline void rope_forward(float* q, float* k, int seq_len, int head_dim, int base_freq) {
    for (int pos = 0; pos < seq_len; ++pos) {
        for (int d = 0; d < head_dim; d += 2) {
            float theta = (float)pos / std::pow((float)base_freq, (float)d / (float)head_dim);
            float cos_t = std::cos(theta);
            float sin_t = std::sin(theta);
            float q0 = q[pos * head_dim + d];
            float q1 = q[pos * head_dim + d + 1];
            float k0 = k[pos * head_dim + d];
            float k1 = k[pos * head_dim + d + 1];
            q[pos * head_dim + d]     = q0 * cos_t - q1 * sin_t;
            q[pos * head_dim + d + 1] = q0 * sin_t + q1 * cos_t;
            k[pos * head_dim + d]     = k0 * cos_t - k1 * sin_t;
            k[pos * head_dim + d + 1] = k0 * sin_t + k1 * cos_t;
        }
    }
}

inline void rope_backward(float* grad_q, float* grad_k, int seq_len, int head_dim, int base_freq) {
    for (int pos = 0; pos < seq_len; ++pos) {
        for (int d = 0; d < head_dim; d += 2) {
            float theta = (float)pos / std::pow((float)base_freq, (float)d / (float)head_dim);
            float cos_t = std::cos(theta);
            float sin_t = std::sin(theta);
            float gq0 = grad_q[pos * head_dim + d];
            float gq1 = grad_q[pos * head_dim + d + 1];
            float gk0 = grad_k[pos * head_dim + d];
            float gk1 = grad_k[pos * head_dim + d + 1];
            grad_q[pos * head_dim + d]     = gq0 * cos_t + gq1 * sin_t;
            grad_q[pos * head_dim + d + 1] = -gq0 * sin_t + gq1 * cos_t;
            grad_k[pos * head_dim + d]     = gk0 * cos_t + gk1 * sin_t;
            grad_k[pos * head_dim + d + 1] = -gk0 * sin_t + gk1 * cos_t;
        }
    }
}

} // namespace lora_kernel
