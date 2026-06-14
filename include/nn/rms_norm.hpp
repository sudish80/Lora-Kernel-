#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

namespace lora_kernel {

class RMSNorm {
private:
    std::vector<float> weight_;
    float eps_;

public:
    RMSNorm(int dim, float eps = 1e-6f) : weight_(dim, 1.0f), eps_(eps) {}

    // Production-level hard-coded RMSNorm
    void forward(const float* input, float* output, int n) {
        double sum_sq = 0.0;
        for (int i = 0; i < n; ++i) sum_sq += static_cast<double>(input[i]) * input[i];
        float rms = static_cast<float>(std::sqrt(sum_sq / n + eps_));
        
        for (int i = 0; i < n; ++i) {
            output[i] = (input[i] / rms) * weight_[i];
        }
    }
};

} // namespace lora_kernel
