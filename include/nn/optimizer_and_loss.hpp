#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace lora_kernel {

class AdamOptimizer {
private:
    std::vector<float> m_, v_;
    float beta1_{0.9f}, beta2_{0.999f}, eps_{1e-8f};
    int   step_{0};

public:
    explicit AdamOptimizer(size_t size) : m_(size, 0.0f), v_(size, 0.0f) {}

    void update(float* params, const float* grads, float lr, size_t size) {
        ++step_;
        float bias_corr1 = 1.0f - std::pow(beta1_, step_);
        float bias_corr2 = 1.0f - std::pow(beta2_, step_);

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < size; ++i) {
            m_[i] = beta1_ * m_[i] + (1.0f - beta1_) * grads[i];
            v_[i] = beta2_ * v_[i] + (1.0f - beta2_) * grads[i] * grads[i];
            float m_hat = m_[i] / bias_corr1;
            float v_hat = v_[i] / bias_corr2;
            params[i] -= lr * m_hat / (std::sqrt(v_hat) + eps_);
        }
    }
};

class GradientClipper {
private:
    float max_norm_{1.0f};

public:
    void set_max_norm(float n) { max_norm_ = n; }

    float clip(float* grads, size_t size) {
        double norm_sq = 0.0;
        #pragma omp parallel for reduction(+:norm_sq) schedule(static)
        for (size_t i = 0; i < size; ++i)
            norm_sq += static_cast<double>(grads[i]) * grads[i];

        float norm = static_cast<float>(std::sqrt(norm_sq));
        if (norm > max_norm_) {
            float scale = max_norm_ / (norm + 1e-6f);
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < size; ++i)
                grads[i] *= scale;
        }
        return norm;
    }
};

class CosineScheduler {
private:
    float max_lr_, min_lr_;
    int   warmup_, total_, step_{0};

public:
    CosineScheduler(float max_lr, float min_lr, int warmup, int total)
        : max_lr_(max_lr), min_lr_(min_lr), warmup_(warmup), total_(total) {}

    float get() {
        float lr;
        if (step_ < warmup_) {
            lr = min_lr_ + (max_lr_ - min_lr_) *
                 (static_cast<float>(step_) / warmup_);
        } else {
            float prog = static_cast<float>(step_ - warmup_) /
                         static_cast<float>(total_ - warmup_);
            lr = min_lr_ + 0.5f * (max_lr_ - min_lr_) *
                 (1.0f + std::cos(static_cast<float>(M_PI) * prog));
        }
        ++step_;
        return lr;
    }

    void restart() { step_ = 0; }
};

class FusedSoftmaxCrossEntropy {
public:
    float compute(const float* logits, const int* targets,
                  int batch, int vocab) const {
        double total_loss = 0.0;
        for (int b = 0; b < batch; ++b) {
            const float* row = logits + b * vocab;
            float max_val = *std::max_element(row, row + vocab);
            double sum_exp = 0.0;
            for (int i = 0; i < vocab; ++i)
                sum_exp += std::exp(static_cast<double>(row[i] - max_val));
            double log_pt = static_cast<double>(row[targets[b]] - max_val)
                            - std::log(sum_exp + 1e-12);
            total_loss -= log_pt;
        }
        return static_cast<float>(total_loss / batch);
    }
};

} // namespace lora_kernel
