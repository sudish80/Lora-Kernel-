#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace lora_kernel {

class DynamicLossScaler {
private:
    float scale_;
    int counter_;
    int growth_interval_;
public:
    DynamicLossScaler(float initial_scale = 65536.0f, int growth_interval = 2000)
        : scale_(initial_scale), counter_(0), growth_interval_(growth_interval) {}

    void scale_gradients(float* grads, int64_t n) {
        for (int64_t i = 0; i < n; ++i)
            grads[i] *= scale_;
    }

    void unscale_gradients(float* grads, int64_t n) {
        float inv = 1.0f / scale_;
        for (int64_t i = 0; i < n; ++i)
            grads[i] *= inv;
    }

    bool has_overflow(const float* grads, int64_t n) {
        for (int64_t i = 0; i < n; ++i) {
            float v = grads[i];
            if (std::isnan(v) || std::isinf(v))
                return true;
        }
        return false;
    }

    void update(bool overflow) {
        if (overflow) {
            scale_ /= 2.0f;
            if (scale_ < 1.0f) scale_ = 1.0f;
            counter_ = 0;
        } else {
            if (++counter_ >= growth_interval_) {
                scale_ *= 2.0f;
                counter_ = 0;
            }
        }
    }

    float get_scale() const { return scale_; }
    int get_growth_interval() const { return growth_interval_; }
    void set_growth_interval(int g) { growth_interval_ = g; }
};

} // namespace lora_kernel
