#pragma once
#include "include/core/tensor.hpp"
#include "include/nn/transformer_blocks.hpp"
#include "include/nn/loss_scaling.hpp"
#include "include/nn/checkpointing.hpp"
#include "include/nn/mixed_precision.hpp"
#include "include/nn/gradient_health.hpp"
#include "include/nn/optimizer_and_loss.hpp"
#include "include/nn/training_enhancements.hpp"
#include <iostream>
#include <cmath>

namespace lora_kernel {

class TrainingPipeline {
private:
    Transformer& model_;
    DynamicLossScaler loss_scaler_;
    GradientCheckpointing checkpointing_;
    MixedPrecision mixed_prec_;

    float lr_;
    float weight_decay_;
    float max_grad_norm_;
    int step_{0};

    struct OptState {
        Tensor m, v;
        OptState(int64_t n) : m({n}), v({n}) {}
    };
    std::vector<OptState> opt_states_;

    GradientHealthMonitor health_monitor_;
    GradientClipper clipper_;

    GradientAccumulator* grad_accum_;
    SWA* swa_;
    EMA* ema_;
    std::vector<float> flat_grad_buffer_;

public:
    TrainingPipeline(Transformer& model, float lr = 1e-4f,
                     float wd = 0.01f, float max_grad_norm = 1.0f)
        : model_(model), lr_(lr), weight_decay_(wd), max_grad_norm_(max_grad_norm),
          grad_accum_(nullptr), swa_(nullptr), ema_(nullptr) {
        clipper_.set_max_norm(max_grad_norm);
        int64_t total_grads = model_.get_param_grad_count();
        opt_states_.emplace_back(total_grads);
        flat_grad_buffer_.resize(total_grads);
        std::cout << "[TRAIN] Pipeline initialized, params="
                  << model_.num_parameters() << " grad_elems=" << total_grads << "\n";
    }

    ~TrainingPipeline() {
        delete grad_accum_;
        delete swa_;
        delete ema_;
    }

    float train_step(const Tensor& input_ids, const Tensor& targets) {
        int B = (int)input_ids.size(0);
        int S = (int)input_ids.size(1);
        int V = (int)model_.vocab_size();

        // 1. Forward pass
        Tensor logits({B, S, V});
        model_.forward(input_ids, logits);

        // 2. Compute loss and gradient of loss w.r.t. logits
        float loss = 0.0f;
        Tensor grad_logits({B, S, V});
        grad_logits.zeros();

        double total_loss = 0.0;
        for (int b = 0; b < B; ++b) {
            for (int s = 0; s < S; ++s) {
                int64_t row_offset = (b * S + s) * V;
                float* row = logits.data() + row_offset;

                float max_val = row[0];
                for (int v = 1; v < V; ++v)
                    if (row[v] > max_val) max_val = row[v];

                double sum_exp = 0.0;
                for (int v = 0; v < V; ++v)
                    sum_exp += std::exp((double)(row[v] - max_val));

                float inv_sum = (float)(1.0 / (sum_exp + 1e-12));
                for (int v = 0; v < V; ++v)
                    grad_logits.at({b, s, v}) = std::exp((double)(row[v] - max_val)) * inv_sum;

                int t = (int)targets.at({b, s});
                grad_logits.at({b, s, t}) -= 1.0f;

                total_loss -= (double)(row[t] - max_val) - std::log(sum_exp + 1e-12);
            }
        }
        loss = (float)(total_loss / (B * S));

        // 3. Scale gradients and backward
        float scale = loss_scaler_.get_scale();
        for (int i = 0; i < B * S * V; ++i)
            grad_logits.data()[i] *= scale;

        model_.backward(grad_logits);

        // 4. Unscale gradients
        auto& param_grads = model_.get_param_grads();
        float inv_scale = 1.0f / scale;
        for (auto& g : param_grads) {
            int64_t n = g.numel();
            for (int64_t i = 0; i < n; ++i)
                g.data()[i] *= inv_scale;
        }

        // 5. Gradient accumulation
        if (grad_accum_) {
            int64_t offset = 0;
            for (auto& g : param_grads) {
                int64_t n = g.numel();
                std::memcpy(flat_grad_buffer_.data() + offset, g.data(), n * sizeof(float));
                offset += n;
            }
            grad_accum_->accumulate(flat_grad_buffer_.data(), (int64_t)flat_grad_buffer_.size());
        }

        bool do_step = !grad_accum_ || grad_accum_->ready();

        if (do_step) {
            // 6. Get normalized gradients from accumulator
            if (grad_accum_) {
                grad_accum_->get_normalized(flat_grad_buffer_.data(), (int64_t)flat_grad_buffer_.size());
                int64_t offset = 0;
                for (auto& g : param_grads) {
                    int64_t n = g.numel();
                    std::memcpy(g.data(), flat_grad_buffer_.data() + offset, n * sizeof(float));
                    offset += n;
                }
            }

            // 7. Gradient clipping
            for (auto& g : param_grads) {
                clipper_.clip(g.data(), g.numel());
            }

            // 8. Check gradient health
            bool healthy = true;
            for (auto& g : param_grads) {
                if (!health_monitor_.check_gradients(g.data(), g.numel(), "model")) {
                    healthy = false;
                    break;
                }
            }

            if (!healthy || health_monitor_.should_skip_step()) {
                loss_scaler_.update(true);
                model_.zero_grad();
                if (grad_accum_) grad_accum_->reset();
                step_++;
                return loss;
            }

            // 9. Optimizer step (AdamW with weight decay)
            optimizer_step(scale);

            // 10. Update loss scaler
            loss_scaler_.update(false);

            // 11. Zero gradients for next accumulation cycle
            model_.zero_grad();

            // 12. Reset accumulator
            if (grad_accum_) grad_accum_->reset();

            // 13. SWA/EMA update
            if (swa_ || ema_) {
                const auto& params = model_.get_params();
                for (auto* p : params) {
                    int64_t n = p->numel();
                    if (swa_) swa_->update(p->data(), n, step_);
                    if (ema_) ema_->update(p->data(), n);
                }
            }
        }

        step_++;
        return loss;
    }

    void set_learning_rate(float lr) { lr_ = lr; }
    int current_step() const { return step_; }

    DynamicLossScaler& get_loss_scaler() { return loss_scaler_; }
    GradientHealthMonitor& get_health_monitor() { return health_monitor_; }

    void set_gradient_accumulation_steps(int steps) {
        delete grad_accum_;
        grad_accum_ = nullptr;
        if (steps > 1) {
            int64_t total_grads = model_.get_param_grad_count();
            grad_accum_ = new GradientAccumulator(total_grads, steps);
            std::cout << "[TRAIN] Gradient accumulation enabled: " << steps << " steps\n";
        }
    }

    void init_swa(int start_step = 1000) {
        delete swa_;
        int64_t total = model_.num_parameters();
        swa_ = new SWA(total, start_step);
        std::cout << "[TRAIN] SWA enabled, start_step=" << start_step << "\n";
    }

    void init_ema(float decay = 0.999f) {
        delete ema_;
        int64_t total = model_.num_parameters();
        ema_ = new EMA(total, decay);
        std::cout << "[TRAIN] EMA enabled, decay=" << decay << "\n";
    }

    void get_swa_weights(float* out) {
        if (swa_) swa_->get_swa(out, model_.num_parameters());
    }

    void get_ema_weights(float* out) {
        if (ema_) ema_->copy_to(out, model_.num_parameters());
    }

private:
    void optimizer_step(float scale) {
        float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
        float bias_corr1 = 1.0f - std::pow(beta1, step_ + 1);
        float bias_corr2 = 1.0f - std::pow(beta2, step_ + 1);

        const auto& params = model_.get_params();
        auto& grads = model_.get_param_grads();

        int64_t flat_offset = 0;
        auto& opt = opt_states_[0];

        for (size_t p_idx = 0; p_idx < params.size(); ++p_idx) {
            Tensor* param = params[p_idx];
            Tensor& grad = grads[p_idx];
            int64_t n = param->numel();

            for (int64_t i = 0; i < n; ++i) {
                float g = grad.data()[i];
                int64_t idx = flat_offset + i;

                opt.m[idx] = beta1 * opt.m[idx] + (1.0f - beta1) * g;
                opt.v[idx] = beta2 * opt.v[idx] + (1.0f - beta2) * g * g;

                float m_hat = opt.m[idx] / bias_corr1;
                float v_hat = opt.v[idx] / bias_corr2;

                float param_val = param->data()[i];
                param_val -= lr_ * (weight_decay_ * param_val + m_hat / (std::sqrt(v_hat) + eps));
                param->data()[i] = param_val;
            }

            flat_offset += n;
        }
    }
};

} // namespace lora_kernel
