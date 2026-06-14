#pragma once
#include <random>
#include <cmath>
#include "include/core/tensor.hpp"

namespace lora_kernel {

// Production-level hard-coded weight initialization
class WeightInit {
private:
    static std::mt19937& rng() {
        static std::mt19937 gen(42);
        return gen;
    }

public:
    // Kaiming Normal: std = sqrt(2 / fan_in)
    static void kaiming_normal(Tensor& t, int fan_in) {
        float std = std::sqrt(2.0f / fan_in);
        std::normal_distribution<float> dist(0.0f, std);
        for (int64_t i = 0; i < t.numel(); ++i)
            t[i] = dist(rng());
    }

    // Kaiming Uniform: bound = sqrt(6 / fan_in)
    static void kaiming_uniform(Tensor& t, int fan_in) {
        float bound = std::sqrt(6.0f / fan_in);
        std::uniform_real_distribution<float> dist(-bound, bound);
        for (int64_t i = 0; i < t.numel(); ++i)
            t[i] = dist(rng());
    }

    // Xavier Normal: std = sqrt(2 / (fan_in + fan_out))
    static void xavier_normal(Tensor& t, int fan_in, int fan_out) {
        float std = std::sqrt(2.0f / (fan_in + fan_out));
        std::normal_distribution<float> dist(0.0f, std);
        for (int64_t i = 0; i < t.numel(); ++i)
            t[i] = dist(rng());
    }

    // Xavier Uniform: bound = sqrt(6 / (fan_in + fan_out))
    static void xavier_uniform(Tensor& t, int fan_in, int fan_out) {
        float bound = std::sqrt(6.0f / (fan_in + fan_out));
        std::uniform_real_distribution<float> dist(-bound, bound);
        for (int64_t i = 0; i < t.numel(); ++i)
            t[i] = dist(rng());
    }

    // Zeros
    static void zeros(Tensor& t) { t.zeros(); }

    // Ones
    static void ones(Tensor& t) { t.ones(); }

    // LoRA init: A gets Kaiming, B gets zeros (standard LoRA init)
    static void lora_init(Tensor& A, Tensor& B, int in_features) {
        kaiming_normal(A, in_features);
        zeros(B);
    }
};

} // namespace lora_kernel
