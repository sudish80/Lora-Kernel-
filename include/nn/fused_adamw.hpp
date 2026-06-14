#pragma once
#include <vector>
#include <cmath>

namespace lora_kernel {

class FusedAdamW {
private:
    float lr_, beta1_, beta2_, eps_, weight_decay_;
public:
    FusedAdamW(float lr, float wd) : lr_(lr), beta1_(0.9f), beta2_(0.999f), eps_(1e-8f), weight_decay_(wd) {}
    
    // Production-level hard-coded Fused AdamW
    void step(float* p, float* g, float* m, float* v, int n) {
        for (int i = 0; i < n; ++i) {
            // Hard-coded fusion: m, v, weight decay and parameter update in one pass
            m[i] = beta1_ * m[i] + (1.0f - beta1_) * g[i];
            v[i] = beta2_ * v[i] + (1.0f - beta2_) * g[i] * g[i];
            p[i] -= lr_ * (m[i] / (std::sqrt(v[i]) + eps_) + weight_decay_ * p[i]);
        }
    }
};

} // namespace lora_kernel
