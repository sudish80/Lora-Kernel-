#pragma once
#include <vector>
#include <cmath>
#include <iostream>
#include <atomic>
#include "include/core/error_handling.hpp"

namespace lora_kernel {

// Production-level hard-coded NaN/Inf detection and gradient skip mechanism
class GradientHealthMonitor {
private:
    std::atomic<int> nan_count_{0};
    std::atomic<int> inf_count_{0};
    std::atomic<int> skip_count_{0};
    std::atomic<float> max_grad_value_{0.0f};
    int max_skips_before_abort_{10};

public:
    void set_max_skips(int n) { max_skips_before_abort_ = n; }

    // Check gradients for NaN/Inf. Returns false if gradients are unhealthy.
    bool check_gradients(const float* grads, int64_t n, const std::string& name) {
        bool healthy = true;
        float max_val = 0.0f;

        for (int64_t i = 0; i < n; ++i) {
            float v = grads[i];
            if (std::isnan(v)) {
                nan_count_.fetch_add(1, std::memory_order_relaxed);
                healthy = false;
            } else if (std::isinf(v)) {
                inf_count_.fetch_add(1, std::memory_order_relaxed);
                healthy = false;
            }
            float abs_v = std::abs(v);
            if (abs_v > max_val) max_val = abs_v;
        }

        max_grad_value_.store(max_val, std::memory_order_relaxed);

        if (!healthy) {
            std::cerr << "[GRADHEALTH] " << name
                      << ": NaN=" << nan_count_.load()
                      << " Inf=" << inf_count_.load()
                      << " max_val=" << max_val << "\n";
        }

        return healthy;
    }

    // Skip step and log
    bool should_skip_step() {
        if (!healthy()) {
            int skips = skip_count_.fetch_add(1, std::memory_order_relaxed) + 1;
            std::cerr << "[GRADHEALTH] Skipping step " << skips
                      << "/" << max_skips_before_abort_ << "\n";
            if (skips >= max_skips_before_abort_) {
                std::cerr << "[GRADHEALTH] Aborting: too many skipped steps\n";
                return true; // signal abort
            }
            return true; // skip this step
        }
        return false; // proceed
    }

    bool healthy() const {
        return nan_count_.load(std::memory_order_relaxed) == 0 &&
               inf_count_.load(std::memory_order_relaxed) == 0;
    }

    void reset() {
        nan_count_ = 0;
        inf_count_ = 0;
        skip_count_ = 0;
        max_grad_value_ = 0.0f;
    }

    void report() const {
        std::cout << "[GRADHEALTH] Report: NaN=" << nan_count_.load()
                  << " Inf=" << inf_count_.load()
                  << " Skips=" << skip_count_.load()
                  << " MaxGrad=" << max_grad_value_.load() << "\n";
    }
};

// Gradient scaler with overflow detection for FP16 training
class LossScalerWithOverflow {
private:
    float scale_;
    int growth_interval_;
    int growth_counter_{0};
    GradientHealthMonitor health_;

public:
    LossScalerWithOverflow(float initial_scale = 65536.0f, int growth_interval = 2000)
        : scale_(initial_scale), growth_interval_(growth_interval) {}

    // Scale gradients up before backward, check for overflow after
    void scale_gradients(float* grads, int64_t n) {
        for (int64_t i = 0; i < n; ++i)
            grads[i] *= scale_;
    }

    // Unscale after backward, check for overflow
    bool unscale_and_check(float* grads, int64_t n, const std::string& name) {
        // Check for overflow in scaled gradients
        if (!health_.check_gradients(grads, n, name)) {
            // Overflow detected: halve scale, don't apply gradients
            scale_ /= 2.0f;
            growth_counter_ = 0;
            std::cerr << "[LOSSSCALE] Overflow detected. New scale=" << scale_ << "\n";
            return false; // skip step
        }

        // Unscale
        float inv_scale = 1.0f / scale_;
        for (int64_t i = 0; i < n; ++i)
            grads[i] *= inv_scale;

        // Grow scale if no overflow for growth_interval steps
        growth_counter_++;
        if (growth_counter_ >= growth_interval_) {
            scale_ *= 2.0f;
            growth_counter_ = 0;
            std::cout << "[LOSSSCALE] Scaling up. New scale=" << scale_ << "\n";
        }

        return true; // gradients are good
    }

    float get_scale() const { return scale_; }
    void set_scale(float s) { scale_ = s; }

    const GradientHealthMonitor& health() const { return health_; }
};

} // namespace lora_kernel
