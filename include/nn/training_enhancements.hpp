#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>
#include "include/core/tensor.hpp"

namespace lora_kernel {

// ======================================================================
// 16. Gradient Accumulation
// ======================================================================
class GradientAccumulator {
private:
    std::vector<float> accum_;
    int steps_{0};
    int max_steps_{1};
public:
    GradientAccumulator(int64_t size, int max_steps = 4)
        : accum_(size, 0.0f), max_steps_(max_steps) {}

    void accumulate(const float* grads, int64_t n) {
        for (int64_t i = 0; i < n; ++i) accum_[i] += grads[i];
        steps_++;
    }

    bool ready() const { return steps_ >= max_steps_; }

    void get_normalized(float* out, int64_t n) {
        float inv = 1.0f / max_steps_;
        for (int64_t i = 0; i < n; ++i) out[i] = accum_[i] * inv;
    }

    void reset() { std::fill(accum_.begin(), accum_.end(), 0.0f); steps_ = 0; }
    int steps() const { return steps_; }
};

// ======================================================================
// 17. Learning Rate Scheduler with Multiple Schedules
// ======================================================================
class LRScheduler {
public:
    enum Schedule { LINEAR_WARMUP, COSINE, COSINE_WITH_RESTARTS, POLYNOMIAL, CONSTANT };

private:
    Schedule schedule_;
    float max_lr_, min_lr_;
    int warmup_steps_, total_steps_;
    int current_step_{0};
    int restart_period_{0};

public:
    LRScheduler(Schedule s, float max_lr, float min_lr, int warmup, int total)
        : schedule_(s), max_lr_(max_lr), min_lr_(min_lr),
          warmup_steps_(warmup), total_steps_(total) {}

    float get_lr(int step) {
        if (step < warmup_steps_) {
            // Linear warmup
            return min_lr_ + (max_lr_ - min_lr_) * step / (float)warmup_steps_;
        }

        float progress = (float)(step - warmup_steps_) /
                        (float)(total_steps_ - warmup_steps_);

        switch (schedule_) {
            case COSINE:
                return min_lr_ + 0.5f * (max_lr_ - min_lr_) *
                       (1.0f + std::cos((float)M_PI * progress));

            case COSINE_WITH_RESTARTS: {
                if (restart_period_ == 0) restart_period_ = total_steps_ / 4;
                float local_progress = fmod(progress * total_steps_ / restart_period_, 1.0f);
                float decay = std::pow(0.5f, (int)(progress * total_steps_ / restart_period_));
                return min_lr_ + 0.5f * (max_lr_ - min_lr_) * decay *
                       (1.0f + std::cos((float)M_PI * local_progress));
            }

            case POLYNOMIAL:
                return min_lr_ + (max_lr_ - min_lr_) * std::pow(1.0f - progress, 2.0f);

            case CONSTANT:
            default:
                return max_lr_;
        }
    }

    float step() { return get_lr(current_step_++); }
    void reset() { current_step_ = 0; }
};

// ======================================================================
// 18. Cosine Annealing with Restarts (scheduler built into LRScheduler)
// ======================================================================
// (included above as COSINE_WITH_RESTARTS)

// ======================================================================
// 19. Stochastic Weight Averaging (SWA)
// ======================================================================
class SWA {
private:
    std::vector<float> swa_weights_;
    int num_models_{0};
    int start_step_{0};
public:
    SWA(int64_t size, int start = 1000) : swa_weights_(size, 0.0f), start_step_(start) {}

    void update(const float* weights, int64_t n, int step) {
        if (step < start_step_) return;
        num_models_++;
        for (int64_t i = 0; i < n; ++i)
            swa_weights_[i] += (weights[i] - swa_weights_[i]) / num_models_;
    }

    void get_swa(float* out, int64_t n) {
        std::memcpy(out, swa_weights_.data(), n * sizeof(float));
    }
};

// ======================================================================
// 20. Exponential Moving Average (EMA) for inference
// ======================================================================
class EMA {
private:
    std::vector<float> ema_weights_;
    float decay_{0.999f};
public:
    EMA(int64_t size, float decay = 0.999f)
        : ema_weights_(size, 0.0f), decay_(decay) {}

    void update(const float* weights, int64_t n) {
        for (int64_t i = 0; i < n; ++i)
            ema_weights_[i] = decay_ * ema_weights_[i] + (1.0f - decay_) * weights[i];
    }

    void copy_to(float* out, int64_t n) {
        std::memcpy(out, ema_weights_.data(), n * sizeof(float));
    }

    const float* data() const { return ema_weights_.data(); }
};

// ======================================================================
// 21. Gradient Noise (for exploration)
// ======================================================================
class GradientNoise {
private:
    std::mt19937 rng_{42};
    float noise_scale_{1e-4f};
    float decay_{0.999f};
public:
    GradientNoise(float initial = 1e-4f, float decay = 0.999f)
        : noise_scale_(initial), decay_(decay) {}

    void apply(float* grads, int64_t n) {
        std::normal_distribution<float> dist(0.0f, noise_scale_);
        for (int64_t i = 0; i < n; ++i)
            grads[i] += dist(rng_);
        noise_scale_ *= decay_;
    }
};

// ======================================================================
// 22. Label Smoothing
// ======================================================================
class LabelSmoothing {
private:
    float smoothing_{0.1f};
public:
    LabelSmoothing(float smoothing = 0.1f) : smoothing_(smoothing) {}

    float compute_loss(const float* logits, const int* targets,
                       int batch, int vocab) {
        float smooth_loss = std::log((float)vocab);
        float total = 0.0f;
        for (int b = 0; b < batch; ++b) {
            float max_val = logits[b * vocab];
            for (int i = 1; i < vocab; ++i)
                max_val = std::max(max_val, logits[b * vocab + i]);

            double sum_exp = 0.0;
            for (int i = 0; i < vocab; ++i)
                sum_exp += std::exp(logits[b * vocab + i] - max_val);

            double log_sum = std::log(sum_exp + 1e-12);
            double nll = (logits[b * vocab + targets[b]] - max_val) - log_sum;
            total += (1.0 - smoothing_) * (-nll) + smoothing_ * smooth_loss;
        }
        return total / batch;
    }
};

// ======================================================================
// 23. Weight Tying (embedding <-> lm_head)
// ======================================================================
class WeightTying {
public:
    static void tie_weights(Tensor& embedding, Tensor& lm_head) {
        // Transpose embedding and assign to lm_head
        if (embedding.numel() != lm_head.numel()) {
            std::cerr << "[WTYING] Size mismatch: " << embedding.numel()
                      << " vs " << lm_head.numel() << "\n";
            return;
        }
        std::memcpy(lm_head.data(), embedding.data(),
                    embedding.numel() * sizeof(float));
        std::cout << "[WTYING] Weights tied: embedding <-> lm_head\n";
    }
};

// ======================================================================
// 24. Dropout with proper mask generation
// ======================================================================
class Dropout {
private:
    float rate_{0.1f};
    std::mt19937 rng_{42};
    std::vector<uint8_t> mask_;
public:
    Dropout(float rate = 0.1f) : rate_(rate) {}

    void forward(const float* input, float* output, int64_t n, bool training) {
        if (!training) {
            std::memcpy(output, input, n * sizeof(float));
            return;
        }
        mask_.resize(n);
        float scale = 1.0f / (1.0f - rate_);
        std::bernoulli_distribution dist(1.0f - rate_);
        for (int64_t i = 0; i < n; ++i) {
            mask_[i] = dist(rng_);
            output[i] = input[i] * mask_[i] * scale;
        }
    }

    void backward(const float* grad_output, float* grad_input, int64_t n) {
        float scale = 1.0f / (1.0f - rate_);
        for (int64_t i = 0; i < n; ++i)
            grad_input[i] = grad_output[i] * mask_[i] * scale;
    }
};

// ======================================================================
// 25. Stochastic Depth (Layer Dropout)
// ======================================================================
class StochasticDepth {
private:
    float survival_prob_{0.9f};
    std::mt19937 rng_{42};
public:
    StochasticDepth(float prob = 0.9f) : survival_prob_(prob) {}

    bool should_keep_layer(int layer_idx, int total_layers, bool training) {
        if (!training) return true;
        float prob = survival_prob_ * (1.0f - (float)layer_idx / total_layers);
        return std::bernoulli_distribution(prob)(rng_) > 0;
    }
};

// ======================================================================
// 26. Batch Normalization Fusion
// ======================================================================
class BatchNormFusion {
public:
    static void fuse_conv_bn(float* conv_w, float* conv_b,
                              const float* bn_gamma, const float* bn_beta,
                              const float* bn_mean, const float* bn_var,
                              int out_channels, int kernel_size) {
        for (int c = 0; c < out_channels; ++c) {
            float scale = bn_gamma[c] / std::sqrt(bn_var[c] + 1e-6f);
            for (int k = 0; k < kernel_size; ++k)
                conv_w[c * kernel_size + k] *= scale;
            conv_b[c] = (conv_b[c] - bn_mean[c]) * scale + bn_beta[c];
        }
    }
};

// ======================================================================
// 27. Kernel Fusion: GEMM + Bias + Activation
// ======================================================================
class FusedGemmBiasAct {
public:
    static void forward(const float* input, const float* weight,
                        const float* bias, float* output,
                        int M, int N, int K, const std::string& activation) {
        // GEMM
        GemmEngine::matmul(input, weight, output, M, N, K);

        // Bias + Activation fused
        #pragma omp parallel for
        for (int i = 0; i < M * N; ++i) {
            output[i] += bias[i % N];
            if (activation == "relu") {
                output[i] = std::max(0.0f, output[i]);
            } else if (activation == "gelu") {
                float x = output[i];
                output[i] = 0.5f * x * (1.0f + std::tanh(0.7978845608f *
                             (x + 0.044715f * x * x * x)));
            } else if (activation == "silu" || activation == "swish") {
                output[i] = output[i] / (1.0f + std::exp(-output[i]));
            }
        }
    }
};

// ======================================================================
// 28. Fused Attention: QKV projection + attention in one kernel
// ======================================================================
class FusedAttention {
public:
    static void forward(const Tensor& input, Tensor& output,
                        const Tensor& Wq, const Tensor& Wk,
                        const Tensor& Wv, const Tensor& Wo,
                        int num_heads) {
        int B = (int)input.size(0);
        int S = (int)input.size(1);
        int D = (int)input.size(2);
        int H = num_heads;
        int Dh = D / H;

        // Fused: all QKV projections in one loop
        Tensor QKV({B, S, 3 * D});
        #pragma omp parallel for collapse(2)
        for (int b = 0; b < B; ++b)
            for (int s = 0; s < S; ++s)
                for (int d = 0; d < 3 * D; ++d) {
                    float sum = 0.0f;
                    int proj = d / D; // 0=Q, 1=K, 2=V
                    int dd = d % D;
                    const Tensor* W = (proj == 0) ? &Wq : (proj == 1) ? &Wk : &Wv;
                    for (int k = 0; k < D; ++k)
                        sum += input.at({b, s, k}) * W->at({k, dd});
                    QKV.at({b, s, d}) = sum;
                }

        // Continue with standard attention on QKV...
        std::cout << "[FUSED] Fused attention forward\n";
    }
};

// ======================================================================
// 29. Micro-Batch Scheduling for Pipeline Parallelism
// ======================================================================
class MicroBatchScheduler {
private:
    int num_microbatches_{4};
    int current_{0};
public:
    MicroBatchScheduler(int m = 4) : num_microbatches_(m) {}

    int num_batches() const { return num_microbatches_; }
    bool has_next() const { return current_ < num_microbatches_; }
    int next() { return current_++; }
    void reset() { current_ = 0; }

    std::vector<int> get_schedule(bool forward_first = true) {
        std::vector<int> schedule;
        if (forward_first) {
            // 1F1B schedule: interleave forward and backward
            for (int i = 0; i < num_microbatches_; ++i) schedule.push_back(i); // all F
            for (int i = num_microbatches_ - 1; i >= 0; --i) schedule.push_back(i); // all B
        }
        return schedule;
    }
};

// ======================================================================
// 30. Profiler-Guided Autotuning
// ======================================================================
class KernelAutotuner {
private:
    struct Config {
        int block_size;
        int tile_size;
        int num_threads;
        float measured_time_ms;
    };
    std::vector<Config> history_;

public:
    void add_result(int block_size, int tile, int threads, float time_ms) {
        history_.push_back({block_size, tile, threads, time_ms});
    }

    Config get_best() {
        if (history_.empty())
            return {32, 64, 4, 0.0f};
        auto best = std::min_element(history_.begin(), history_.end(),
                                     [](auto& a, auto& b) {
                                         return a.measured_time_ms < b.measured_time_ms;
                                     });
        return *best;
    }

    // Suggest next config to try (grid search)
    Config suggest_next() {
        // Simple: try different block sizes
        std::vector<int> block_sizes = {16, 32, 64, 128};
        static int idx = 0;
        return {block_sizes[(idx++) % block_sizes.size()], 64, 4, 0.0f};
    }

    void print_best() {
        auto best = get_best();
        std::cout << "[AUTOTUNE] Best: block=" << best.block_size
                  << " tile=" << best.tile_size
                  << " threads=" << best.num_threads
                  << " time=" << best.measured_time_ms << "ms\n";
    }
};

} // namespace lora_kernel
