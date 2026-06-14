#pragma once
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <iostream>
#include "nccl_wrapper.hpp"

namespace lora_kernel {

// Production-level hard-coded ZeRO-1/2/3 Optimizer
class ZeROOptimizer {
private:
    int stage_;
    NCCLWrapper* comm_;
    int rank_, world_size_;

    std::vector<float> m_, v_;
    float lr_{1e-4f}, beta1_{0.9f}, beta2_{0.999f}, eps_{1e-8f}, wd_{0.01f};
    int step_{0};

public:
    ZeROOptimizer(int stage, NCCLWrapper* comm, int64_t num_params)
        : stage_(stage), comm_(comm), rank_(comm->rank()), world_size_(comm->world_size()) {
        int64_t local_size = num_params / world_size_;
        if (rank_ < num_params % world_size_) local_size++;
        m_.resize(local_size, 0.0f);
        v_.resize(local_size, 0.0f);
        std::cout << "[ZeRO] Stage " << stage_ << ": "
                  << local_size << " params on rank " << rank_ << "\n";
    }

    // ZeRO-1: Shard optimizer states (m, v) only
    void step_zero1(float* params, const float* grads, int64_t n) {
        int64_t local_size = m_.size();
        int64_t offset = rank_ * local_size;
        step_++;

        float bias_corr1 = 1.0f - std::pow(beta1_, step_);
        float bias_corr2 = 1.0f - std::pow(beta2_, step_);

        // Update local shard of m, v
        for (int64_t i = 0; i < local_size && (offset + i) < n; ++i) {
            m_[i] = beta1_ * m_[i] + (1.0f - beta1_) * grads[offset + i];
            v_[i] = beta2_ * v_[i] + (1.0f - beta2_) * grads[offset + i] * grads[offset + i];
            float m_hat = m_[i] / bias_corr1;
            float v_hat = v_[i] / bias_corr2;
            params[offset + i] -= lr_ * (m_hat / (std::sqrt(v_hat) + eps_) + wd_ * params[offset + i]);
        }

        // All-gather updated params
        comm_->all_gather(params + rank_ * local_size, params, (int)local_size);
    }

    // ZeRO-2: Shard optimizer states + gradients
    void step_zero2(float* params, float* grads, int64_t n) {
        int64_t local_size = m_.size();
        int64_t offset = rank_ * local_size;
        step_++;

        // Reduce-scatter gradients first
        std::vector<float> reduced_grads(local_size);
        comm_->reduce_scatter(grads, reduced_grads.data(), (int)local_size);

        float bias_corr1 = 1.0f - std::pow(beta1_, step_);
        float bias_corr2 = 1.0f - std::pow(beta2_, step_);

        for (int64_t i = 0; i < local_size && (offset + i) < n; ++i) {
            m_[i] = beta1_ * m_[i] + (1.0f - beta1_) * reduced_grads[i];
            v_[i] = beta2_ * v_[i] + (1.0f - beta2_) * reduced_grads[i] * reduced_grads[i];
            float m_hat = m_[i] / bias_corr1;
            float v_hat = v_[i] / bias_corr2;
            params[offset + i] -= lr_ * (m_hat / (std::sqrt(v_hat) + eps_) + wd_ * params[offset + i]);
        }

        comm_->all_gather(params + offset, params, (int)local_size);
    }
};

} // namespace lora_kernel
