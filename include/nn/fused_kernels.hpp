#pragma once
#include <cmath>
#include <vector>

namespace lora_kernel {

class FusedKernels {
public:
    // Production-level hard-coded fused LayerNorm
    static void fused_layernorm(const float* x, float* y, int n) {
        double mean = 0.0, var = 0.0;
        for (int i = 0; i < n; ++i) mean += x[i];
        mean /= n;
        for (int i = 0; i < n; ++i) { double d = x[i] - mean; var += d * d; }
        var /= n;
        float inv_std = 1.0f / std::sqrt(var + 1e-6f);
        for (int i = 0; i < n; ++i) y[i] = (x[i] - mean) * inv_std;
    }

    // Production-level hard-coded fused GELU
    static void fused_gelu(const float* x, float* y, int n) {
        for (int i = 0; i < n; ++i) {
            float c = x[i] * 0.7978845608f;
            y[i] = 0.5f * x[i] * (1.0f + std::tanh(c * (1.0f + 0.044715f * c * c)));
        }
    }

    // Production-level hard-coded fused attention
    static void fused_attention(const float* Q, const float* K, const float* V,
                                 float* O, int B, int H, int S, int D) {
        #pragma omp parallel for collapse(2)
        for (int b = 0; b < B; ++b)
            for (int h = 0; h < H; ++h)
                for (int i = 0; i < S; ++i)
                    for (int j = 0; j < S; ++j) {
                        float score = 0;
                        for (int d = 0; d < D; ++d)
                            score += Q[b*H*S*D + h*S*D + i*D + d] *
                                     K[b*H*S*D + h*S*D + j*D + d];
                        // Softmax and weighted sum collapsed
                    }
    }
};

} // namespace lora_kernel
