#include "include/nn/transformer_blocks.hpp"
#include <iostream>

namespace lora_kernel {

// ======================================================================
// Multi-Head Attention Forward: x[B,S,D] -> Q,K,V -> scores -> context -> O
// ======================================================================
void MultiHeadAttention::forward(const Tensor& input, Tensor& output) {
    int B = (int)input.size(0);
    int S = (int)input.size(1);
    int D = hidden_dim_;
    int H = num_heads_;
    int Dh = head_dim_;

    last_input_ = input;

    last_Q_.reshape({B, S, D});
    last_K_.reshape({B, S, D});
    last_V_.reshape({B, S, D});

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; ++b) {
        for (int s = 0; s < S; ++s) {
            for (int d = 0; d < D; ++d) {
                float sum_q = 0.0f, sum_k = 0.0f, sum_v = 0.0f;
                for (int k = 0; k < D; ++k) {
                    float x = input.at({b, s, k});
                    sum_q += x * Wq_.at({k, d});
                    sum_k += x * Wk_.at({k, d});
                    sum_v += x * Wv_.at({k, d});
                }
                last_Q_.at({b, s, d}) = sum_q + bias_q_[d];
                last_K_.at({b, s, d}) = sum_k + bias_k_[d];
                last_V_.at({b, s, d}) = sum_v + bias_v_[d];
            }
        }
    }

    for (int b = 0; b < B; ++b) {
        rope_forward(last_Q_.data() + b * S * D,
                     last_K_.data() + b * S * D,
                     S, D, 10000);
    }

    Tensor Q_heads({B * H, S, Dh});
    Tensor K_heads({B * H, S, Dh});
    Tensor V_heads({B * H, S, Dh});

    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int s = 0; s < S; ++s) {
                int bh = b * H + h;
                for (int d = 0; d < Dh; ++d) {
                    Q_heads.at({bh, s, d}) = last_Q_.at({b, s, h * Dh + d});
                    K_heads.at({bh, s, d}) = last_K_.at({b, s, h * Dh + d});
                    V_heads.at({bh, s, d}) = last_V_.at({b, s, h * Dh + d});
                }
            }
        }
    }

    int BH = B * H;
    last_scores_.reshape({BH, S, S});
    float inv_scale = 1.0f / std::sqrt((float)Dh);

    #pragma omp parallel for collapse(3)
    for (int bh = 0; bh < BH; ++bh) {
        for (int i = 0; i < S; ++i) {
            for (int j = 0; j < S; ++j) {
                float sum = 0.0f;
                for (int d = 0; d < Dh; ++d)
                    sum += Q_heads.at({bh, i, d}) * K_heads.at({bh, j, d});
                last_scores_.at({bh, i, j}) = sum * inv_scale;
            }
        }
    }

    for (int bh = 0; bh < BH; ++bh)
        apply_causal_mask(last_scores_.data() + bh * S * S, S);

    last_softmax_ = last_scores_;
    softmax_inplace(last_softmax_);

    Tensor context({BH, S, Dh});
    #pragma omp parallel for collapse(3)
    for (int bh = 0; bh < BH; ++bh) {
        for (int i = 0; i < S; ++i) {
            for (int d = 0; d < Dh; ++d) {
                float sum = 0.0f;
                for (int j = 0; j < S; ++j)
                    sum += last_softmax_.at({bh, i, j}) * V_heads.at({bh, j, d});
                context.at({bh, i, d}) = sum;
            }
        }
    }

    last_context_ = context;
    output.reshape({B, S, D});
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; ++b) {
        for (int s = 0; s < S; ++s) {
            for (int d = 0; d < D; ++d) {
                float sum = 0.0f;
                int h = d / Dh;
                int dh = d % Dh;
                sum = context.at({b * H + h, s, dh});
                output.at({b, s, d}) = sum + bias_o_[d];
            }
        }
    }
}

// ======================================================================
// Multi-Head Attention Backward
// ======================================================================
void MultiHeadAttention::backward(const Tensor& grad_output, Tensor& grad_input,
                                  Tensor& grad_Wq, Tensor& grad_Wk,
                                  Tensor& grad_Wv, Tensor& grad_Wo) {
    int B = (int)last_input_.size(0);
    int S = (int)last_input_.size(1);
    int D = hidden_dim_;
    int H = num_heads_;
    int Dh = head_dim_;
    int BH = B * H;
    Tensor grad_context({BH, S, Dh});
    for (int b = 0; b < B; ++b)
        for (int s = 0; s < S; ++s)
            for (int d = 0; d < D; ++d) {
                int h = d / Dh;
                int dh = d % Dh;
                grad_context.at({b * H + h, s, dh}) += grad_output.at({b, s, d});
            }

    // grad_Wo += context^T @ grad_context (outer product)
    grad_Wo.zeros();
    for (int h = 0; h < H; ++h) {
        for (int d = 0; d < Dh; ++d) {
            for (int d2 = 0; d2 < Dh; ++d2) {
                float sum = 0.0f;
                for (int b = 0; b < B; ++b)
                    for (int s = 0; s < S; ++s)
                        sum += last_context_.at({b * H + h, s, d}) *
                               grad_context.at({b * H + h, s, d2});
                // Accumulate into Wo gradient at the right position
                int out_d = h * Dh + d;
                int out_d2 = h * Dh + d2;
                grad_Wo.at({out_d, out_d2}) += sum;
            }
        }
    }

    // Gradient through softmax: dL/d(scores) = softmax * (dL/d(context) - sum(...))
    // First, project grad_context back through V to get grad_scores
    Tensor grad_scores({BH, S, S});
    #pragma omp parallel for collapse(3)
    for (int bh = 0; bh < BH; ++bh) {
        for (int i = 0; i < S; ++i) {
            for (int j = 0; j < S; ++j) {
                float sum = 0.0f;
                for (int d = 0; d < Dh; ++d)
                    sum += grad_context.at({bh, i, d}) * last_V_.at({bh / H, j, (bh % H) * Dh + d});
                grad_scores.at({bh, i, j}) = sum;
            }
        }
    }

    softmax_backward_inplace(last_softmax_, grad_scores);

    float inv_scale = 1.0f / std::sqrt((float)Dh);
    for (int i = 0; i < BH * S * S; ++i)
        grad_scores.data()[i] *= inv_scale;

    Tensor grad_Q_heads({BH, S, Dh});
    Tensor grad_K_heads({BH, S, Dh});
    Tensor grad_V_heads({BH, S, Dh});

    #pragma omp parallel for collapse(3)
    for (int bh = 0; bh < BH; ++bh)
        for (int j = 0; j < S; ++j)
            for (int d = 0; d < Dh; ++d) {
                float sum = 0.0f;
                for (int i = 0; i < S; ++i)
                    sum += last_softmax_.at({bh, i, j}) * grad_context.at({bh, i, d});
                grad_V_heads.at({bh, j, d}) = sum;
            }

    #pragma omp parallel for collapse(3)
    for (int bh = 0; bh < BH; ++bh)
        for (int i = 0; i < S; ++i)
            for (int d = 0; d < Dh; ++d) {
                float sum = 0.0f;
                for (int j = 0; j < S; ++j)
                    sum += grad_scores.at({bh, i, j}) * last_K_.at({bh / H, j, (bh % H) * Dh + d});
                grad_Q_heads.at({bh, i, d}) = sum;
            }

    #pragma omp parallel for collapse(3)
    for (int bh = 0; bh < BH; ++bh)
        for (int j = 0; j < S; ++j)
            for (int d = 0; d < Dh; ++d) {
                float sum = 0.0f;
                for (int i = 0; i < S; ++i)
                    sum += grad_scores.at({bh, i, j}) * last_Q_.at({bh / H, i, (bh % H) * Dh + d});
                grad_K_heads.at({bh, j, d}) = sum;
            }

    Tensor grad_Q({B, S, D}), grad_K({B, S, D}), grad_V({B, S, D});
    for (int b = 0; b < B; ++b)
        for (int s = 0; s < S; ++s)
            for (int h = 0; h < H; ++h)
                for (int d = 0; d < Dh; ++d) {
                    grad_Q.at({b, s, h * Dh + d}) = grad_Q_heads.at({b * H + h, s, d});
                    grad_K.at({b, s, h * Dh + d}) = grad_K_heads.at({b * H + h, s, d});
                    grad_V.at({b, s, h * Dh + d}) = grad_V_heads.at({b * H + h, s, d});
                }

    for (int b = 0; b < B; ++b) {
        rope_backward(grad_Q.data() + b * S * D,
                      grad_K.data() + b * S * D, S, D, 10000);
    }

    grad_Wq.zeros();
    grad_Wk.zeros();
    grad_Wv.zeros();

    for (int b = 0; b < B; ++b)
        for (int s = 0; s < S; ++s)
            for (int i = 0; i < D; ++i)
                for (int j = 0; j < D; ++j) {
                    grad_Wq.at({i, j}) += last_input_.at({b, s, i}) * grad_Q.at({b, s, j});
                    grad_Wk.at({i, j}) += last_input_.at({b, s, i}) * grad_K.at({b, s, j});
                    grad_Wv.at({i, j}) += last_input_.at({b, s, i}) * grad_V.at({b, s, j});
                }

    Tensor Wq_T = Wq_.transpose(), Wk_T = Wk_.transpose(), Wv_T = Wv_.transpose();
    grad_input.reshape({B, S, D});
    for (int b = 0; b < B; ++b)
        for (int s = 0; s < S; ++s)
            for (int i = 0; i < D; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < D; ++j) {
                    sum += grad_Q.at({b, s, j}) * Wq_T.at({j, i});
                    sum += grad_K.at({b, s, j}) * Wk_T.at({j, i});
                    sum += grad_V.at({b, s, j}) * Wv_T.at({j, i});
                }
                grad_input.at({b, s, i}) = sum;
            }
}

// ======================================================================
// FeedForward Forward: SwiGLU(input @ Wgate) * (input @ Wup) @ Wdown
// ======================================================================
void FeedForward::forward(const Tensor& input, Tensor& output) {
    int B = (int)input.size(0);
    int S = (int)input.size(1);
    int D = hidden_dim_;
    int F = ff_dim_;

    last_input_ = input;

    last_gate_.reshape({B * S, F});
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < B * S; ++i) {
        for (int j = 0; j < F; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < D; ++k)
                sum += input.data()[i * D + k] * Wgate_.at({k, j});
            last_gate_.at({i, j}) = sum + bias_gate_[j];
        }
    }

    last_up_.reshape({B * S, F});
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < B * S; ++i) {
        for (int j = 0; j < F; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < D; ++k)
                sum += input.data()[i * D + k] * Wup_.at({k, j});
            last_up_.at({i, j}) = sum + bias_up_[j];
        }
    }

    last_gated_.reshape({B * S, F});
    for (int i = 0; i < B * S * F; ++i) {
        float gate_val = last_gate_.data()[i];
        float sig = 1.0f / (1.0f + std::exp(-gate_val));
        float swish = gate_val * sig;
        last_gated_.data()[i] = swish * last_up_.data()[i];
    }

    output.reshape({B * S, D});
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < B * S; ++i) {
        for (int j = 0; j < D; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < F; ++k)
                sum += last_gated_.at({i, k}) * Wdown_.at({k, j});
            output.data()[i * D + j] = sum + bias_down_[j];
        }
    }
}

// ======================================================================
// FeedForward Backward
// ======================================================================
void FeedForward::backward(const Tensor& grad_output, Tensor& grad_input,
                           Tensor& grad_Wgate, Tensor& grad_Wup, Tensor& grad_Wdown) {
    int B = (int)last_input_.size(0);
    int S = (int)last_input_.size(1);
    int D = hidden_dim_;
    int F = ff_dim_;
    int BS = B * S;

    grad_Wdown.zeros();
    for (int i = 0; i < F; ++i)
        for (int j = 0; j < D; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < BS; ++k)
                sum += last_gated_.at({k, i}) * grad_output.data()[k * D + j];
            grad_Wdown.at({i, j}) = sum;
        }

    Tensor grad_gated({BS, F});
    Tensor Wdown_T = Wdown_.transpose();
    for (int i = 0; i < BS; ++i)
        for (int j = 0; j < F; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < D; ++k)
                sum += grad_output.data()[i * D + k] * Wdown_T.at({k, j});
            grad_gated.at({i, j}) = sum;
        }

    Tensor grad_gate({BS, F}), grad_up({BS, F});
    for (int i = 0; i < BS * F; ++i) {
        float gate_val = last_gate_.data()[i];
        float sig = 1.0f / (1.0f + std::exp(-gate_val));
        float swish = gate_val * sig;
        float dsig = sig * (1.0f - sig);
        float dswish = sig + gate_val * dsig;
        grad_gate.data()[i] = grad_gated.data()[i] * last_up_.data()[i] * dswish;
        grad_up.data()[i] = grad_gated.data()[i] * swish;
    }

    grad_Wgate.zeros();
    for (int i = 0; i < D; ++i)
        for (int j = 0; j < F; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < BS; ++k)
                sum += last_input_.data()[k * D + i] * grad_gate.at({k, j});
            grad_Wgate.at({i, j}) = sum;
        }

    grad_Wup.zeros();
    for (int i = 0; i < D; ++i)
        for (int j = 0; j < F; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < BS; ++k)
                sum += last_input_.data()[k * D + i] * grad_up.at({k, j});
            grad_Wup.at({i, j}) = sum;
        }

    Tensor Wgate_T = Wgate_.transpose();
    Tensor Wup_T = Wup_.transpose();
    grad_input.reshape({BS, D});
    for (int i = 0; i < BS; ++i)
        for (int j = 0; j < D; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < F; ++k) {
                sum += grad_gate.at({i, k}) * Wgate_T.at({k, j});
                sum += grad_up.at({i, k}) * Wup_T.at({k, j});
            }
            grad_input.data()[i * D + j] = sum;
        }
}

// ======================================================================
// TransformerBlock
// ======================================================================
TransformerBlock::TransformerBlock(const TransformerConfig& cfg)
    : attn_(cfg), ffn_(cfg.hidden_dim, cfg.hidden_dim * 4),
      Wq_({cfg.hidden_dim, cfg.hidden_dim}),
      Wk_({cfg.hidden_dim, cfg.hidden_dim}),
      Wv_({cfg.hidden_dim, cfg.hidden_dim}),
      Wo_({cfg.hidden_dim, cfg.hidden_dim}),
      Wgate_({cfg.hidden_dim, cfg.hidden_dim * 4}),
      Wup_({cfg.hidden_dim, cfg.hidden_dim * 4}),
      Wdown_({cfg.hidden_dim * 4, cfg.hidden_dim}),
      rms_weight_attn_({cfg.hidden_dim}, 1.0f),
      rms_weight_ffn_({cfg.hidden_dim}, 1.0f),
      grad_Wq_({cfg.hidden_dim, cfg.hidden_dim}),
      grad_Wk_({cfg.hidden_dim, cfg.hidden_dim}),
      grad_Wv_({cfg.hidden_dim, cfg.hidden_dim}),
      grad_Wo_({cfg.hidden_dim, cfg.hidden_dim}),
      grad_Wgate_({cfg.hidden_dim, cfg.hidden_dim * 4}),
      grad_Wup_({cfg.hidden_dim, cfg.hidden_dim * 4}),
      grad_Wdown_({cfg.hidden_dim * 4, cfg.hidden_dim}),
      grad_rms_weight_attn_({cfg.hidden_dim}),
      grad_rms_weight_ffn_({cfg.hidden_dim}) {
    // Initialize grad_weight_ptrs_ in order matching param registration
    grad_weight_ptrs_ = {
        grad_Wq_.data(), grad_Wk_.data(), grad_Wv_.data(), grad_Wo_.data(),
        grad_Wgate_.data(), grad_Wup_.data(), grad_Wdown_.data(),
        grad_rms_weight_attn_.data(), grad_rms_weight_ffn_.data()
    };
}

void TransformerBlock::forward(const Tensor& input, Tensor& output,
                                PagedKVCache* cache, int token_pos) {
    int B = (int)input.size(0);
    int S = (int)input.size(1);
    int D = (int)input.size(2);

    last_input_ = input;

    last_norm1_out_.reshape({B, S, D});
    for (int i = 0; i < B * S; ++i)
        rmsnorm_forward(input.data() + i * D,
                        last_norm1_out_.data() + i * D,
                        D, rms_weight_attn_.data(), eps_);

    PagedKVCache* active_cache = cache ? cache : kv_cache_;

    if (active_cache && S == 1) {
        float k_buf[4096], v_buf[4096];
        active_cache->load(0, token_pos, k_buf, v_buf);
        bool has_cached = false;
        for (int d = 0; d < D; ++d) {
            if (k_buf[d] != 0.0f || v_buf[d] != 0.0f) { has_cached = true; break; }
        }

        if (has_cached) {
            int total_len = token_pos + 1;
            Tensor k_cached({total_len, D}), v_cached({total_len, D});
            for (int p = 0; p < total_len; ++p) {
                active_cache->load(0, p, k_buf, v_buf);
                for (int d = 0; d < D; ++d) {
                    k_cached.at({p, d}) = k_buf[d];
                    v_cached.at({p, d}) = v_buf[d];
                }
            }

            Tensor q({1, 1, D});
            for (int d = 0; d < D; ++d) {
                float sum = 0.0f;
                for (int k = 0; k < D; ++k)
                    sum += last_norm1_out_.at({0, 0, k}) * Wq_.at({k, d});
                q.at({0, 0, d}) = sum;
            }

            float inv_scale = 1.0f / std::sqrt((float)D);
            last_attn_out_.reshape({1, 1, D});
            last_attn_out_.zeros();

            float max_val = -1e9f;
            for (int j = 0; j < total_len; ++j) {
                float score = 0.0f;
                for (int d = 0; d < D; ++d)
                    score += q.at({0, 0, d}) * k_cached.at({j, d});
                score *= inv_scale;
                max_val = std::max(max_val, score);
            }
            float sum_exp = 0.0f;
            for (int j = 0; j < total_len; ++j) {
                float score = 0.0f;
                for (int d = 0; d < D; ++d)
                    score += q.at({0, 0, d}) * k_cached.at({j, d});
                float w = std::exp(score * inv_scale - max_val);
                sum_exp += w;
                for (int d = 0; d < D; ++d)
                    last_attn_out_.at({0, 0, d}) += (w / sum_exp) * v_cached.at({j, d});
            }
        } else {
            attn_.forward(last_norm1_out_, last_attn_out_);
            const Tensor& last_k = attn_.get_last_K();
            const Tensor& last_v = attn_.get_last_V();
            for (int d = 0; d < D; ++d) {
                k_buf[d] = last_k.at({0, 0, d});
                v_buf[d] = last_v.at({0, 0, d});
            }
            active_cache->store(0, token_pos, k_buf, v_buf);
        }
    } else {
        last_attn_out_.reshape({B, S, D});
        attn_.forward(last_norm1_out_, last_attn_out_);

        if (active_cache) {
            const Tensor& last_k = attn_.get_last_K();
            const Tensor& last_v = attn_.get_last_V();
            for (int b = 0; b < B; ++b)
                for (int s = 0; s < S; ++s) {
                    float k_buf[4096], v_buf[4096];
                    for (int d = 0; d < D; ++d) {
                        k_buf[d] = last_k.at({b, s, d});
                        v_buf[d] = last_v.at({b, s, d});
                    }
                    active_cache->store(0, token_pos + s, k_buf, v_buf);
                }
        }
    }

    output.reshape({B, S, D});
    for (int i = 0; i < B * S * D; ++i)
        output.data()[i] = input.data()[i] + last_attn_out_.data()[i];

    last_norm2_out_.reshape({B, S, D});
    for (int i = 0; i < B * S; ++i)
        rmsnorm_forward(output.data() + i * D,
                        last_norm2_out_.data() + i * D,
                        D, rms_weight_ffn_.data(), eps_);

    Tensor ffn_out({B, S, D});
    ffn_.forward(last_norm2_out_, ffn_out);

    for (int i = 0; i < B * S * D; ++i)
        output.data()[i] += ffn_out.data()[i];
}

void TransformerBlock::backward(const Tensor& grad_output, Tensor& grad_input) {
    int B = (int)last_input_.size(0);
    int S = (int)last_input_.size(1);
    int D = (int)last_input_.size(2);
    int BS = B * S;
    // --- FFN backward ---
    Tensor grad_norm2({B, S, D});
    ffn_.backward(grad_output, grad_norm2,
                  grad_Wgate_, grad_Wup_, grad_Wdown_);
    // grad_norm2 = dL/dh4 where h4 = rmsnorm2(h3)

    // --- RMSNorm2 backward ---
    // h3 = last_input_ + last_attn_out_ (input to norm2)
    // rmsnorm_backward computes dL/d(input_to_norm) given dL/d(output_of_norm)
    Tensor pre_norm2({B, S, D});
    for (int i = 0; i < BS * D; ++i)
        pre_norm2.data()[i] = last_input_.data()[i] + last_attn_out_.data()[i];

    Tensor grad_h3_from_norm2({B, S, D});
    for (int i = 0; i < BS; ++i)
        rmsnorm_backward(grad_norm2.data() + i * D,
                        pre_norm2.data() + i * D,
                        grad_h3_from_norm2.data() + i * D,
                        rms_weight_ffn_.data(), D, eps_);

    // Grad for rms_weight_ffn_:
    // dL/d(weight_d) = sum_{bs} grad_norm2[bs,d] * x_normed[bs,d]
    // where x_normed = pre_norm2 / rms(pre_norm2)
    grad_rms_weight_ffn_.zeros();
    for (int i = 0; i < BS; ++i) {
        double sum_sq = 0.0;
        for (int d = 0; d < D; ++d)
            sum_sq += (double)pre_norm2.data()[i * D + d] * pre_norm2.data()[i * D + d];
        float rms = (float)std::sqrt(sum_sq / D + eps_);
        float inv_rms = 1.0f / rms;
        for (int d = 0; d < D; ++d) {
            float x_normed = pre_norm2.data()[i * D + d] * inv_rms;
            grad_rms_weight_ffn_.data()[d] += grad_norm2.data()[i * D + d] * x_normed;
        }
    }

    // Total dL/dh3 = grad_output (residual from y = h3 + h5) + grad_h3_from_norm2
    Tensor dldh3({B, S, D});
    for (int i = 0; i < BS * D; ++i)
        dldh3.data()[i] = grad_h3_from_norm2.data()[i] + grad_output.data()[i];

    // --- MHA backward ---
    Tensor grad_norm1({B, S, D});
    attn_.backward(dldh3, grad_norm1,
                   grad_Wq_, grad_Wk_, grad_Wv_, grad_Wo_);
    // grad_norm1 = dL/dh1 where h1 = rmsnorm1(h0)

    // --- RMSNorm1 backward ---
    Tensor grad_h0_from_attn({B, S, D});
    for (int i = 0; i < BS; ++i)
        rmsnorm_backward(grad_norm1.data() + i * D,
                        last_input_.data() + i * D,
                        grad_h0_from_attn.data() + i * D,
                        rms_weight_attn_.data(), D, eps_);

    // Grad for rms_weight_attn_:
    grad_rms_weight_attn_.zeros();
    for (int i = 0; i < BS; ++i) {
        double sum_sq = 0.0;
        for (int d = 0; d < D; ++d)
            sum_sq += (double)last_input_.data()[i * D + d] * last_input_.data()[i * D + d];
        float rms = (float)std::sqrt(sum_sq / D + eps_);
        float inv_rms = 1.0f / rms;
        for (int d = 0; d < D; ++d) {
            float x_normed = last_input_.data()[i * D + d] * inv_rms;
            grad_rms_weight_attn_.data()[d] += grad_norm1.data()[i * D + d] * x_normed;
        }
    }

    // Total dL/dh0 = dL/dh3 (residual from h3 = h0 + h2) + grad_h0_from_attn
    grad_input.reshape({B, S, D});
    for (int i = 0; i < BS * D; ++i)
        grad_input.data()[i] = dldh3.data()[i] + grad_h0_from_attn.data()[i];
}

// ======================================================================
// Full Transformer
// ======================================================================
Transformer::Transformer(const TransformerConfig& cfg) : cfg_(cfg) {
    token_embedding_.reshape({cfg.vocab_size, cfg.hidden_dim});
    pos_embedding_.reshape({cfg.max_seq_len, cfg.hidden_dim});
    lm_head_.reshape({cfg.hidden_dim, cfg.vocab_size});

    WeightInit::kaiming_normal(token_embedding_, cfg.hidden_dim);
    WeightInit::kaiming_normal(pos_embedding_, cfg.hidden_dim);
    WeightInit::kaiming_normal(lm_head_, cfg.hidden_dim);

    // Register all parameter tensors in params_
    params_.reserve(3 + cfg.num_layers * 9);
    params_.push_back(&token_embedding_);
    params_.push_back(&pos_embedding_);
    params_.push_back(&lm_head_);

    blocks_.reserve(cfg.num_layers);
    for (int i = 0; i < cfg.num_layers; ++i)
        blocks_.emplace_back(cfg);

    for (auto& blk : blocks_) {
        params_.push_back(blk.get_Wq_ptr());
        params_.push_back(blk.get_Wk_ptr());
        params_.push_back(blk.get_Wv_ptr());
        params_.push_back(blk.get_Wo_ptr());
        params_.push_back(blk.get_Wgate_ptr());
        params_.push_back(blk.get_Wup_ptr());
        params_.push_back(blk.get_Wdown_ptr());
        params_.push_back(blk.get_rms_weight_attn_ptr());
        params_.push_back(blk.get_rms_weight_ffn_ptr());
    }

    // Create gradient storage for each parameter (same shapes)
    for (auto* p : params_)
        param_grads_.emplace_back(p->shape(), 0.0f);
}

void Transformer::forward(const Tensor& input_ids, Tensor& logits) {
    int B = (int)input_ids.size(0);
    int S = (int)input_ids.size(1);
    int D = cfg_.hidden_dim;

    Tensor hidden({B, S, D});
    for (int b = 0; b < B; ++b)
        for (int s = 0; s < S; ++s) {
            int tok = (int)input_ids.at({b, s});
            for (int d = 0; d < D; ++d)
                hidden.at({b, s, d}) = token_embedding_.at({tok, d}) +
                                       pos_embedding_.at({s, d});
        }

    for (auto& block : blocks_)
        block.forward(hidden, hidden);

    // Save for backward
    last_hidden_ = hidden;

    logits.reshape({B, S, cfg_.vocab_size});
    for (int b = 0; b < B; ++b)
        for (int s = 0; s < S; ++s)
            for (int v = 0; v < cfg_.vocab_size; ++v) {
                float sum = 0.0f;
                for (int d = 0; d < D; ++d)
                    sum += hidden.at({b, s, d}) * lm_head_.at({d, v});
                logits.at({b, s, v}) = sum;
            }
}

void Transformer::backward(const Tensor& grad_logits) {
    int B = grad_logits.size(0);
    int S = grad_logits.size(1);
    int D = cfg_.hidden_dim;
    int V = cfg_.vocab_size;

    // 1. Backward through LM head
    // grad_hidden[b,s,d] = sum_v grad_logits[b,s,v] * lm_head_[d,v]
    Tensor grad_hidden({B, S, D});
    for (int b = 0; b < B; ++b)
        for (int s = 0; s < S; ++s)
            for (int d = 0; d < D; ++d) {
                float sum = 0.0f;
                for (int v = 0; v < V; ++v)
                    sum += grad_logits.at({b, s, v}) * lm_head_.at({d, v});
                grad_hidden.at({b, s, d}) = sum;
            }

    // grad_lm_head[d,v] = sum_{b,s} last_hidden_[b,s,d] * grad_logits[b,s,v]
    Tensor& g_lm = param_grads_[2];
    g_lm.zeros();
    for (int d = 0; d < D; ++d)
        for (int v = 0; v < V; ++v) {
            float sum = 0.0f;
            for (int b = 0; b < B; ++b)
                for (int s = 0; s < S; ++s)
                    sum += last_hidden_.at({b, s, d}) * grad_logits.at({b, s, v});
            g_lm.at({d, v}) = sum;
        }

    // 2. Backward through each block in reverse
    Tensor grad = grad_hidden;
    int blk_params = 9;
    for (int i = (int)blocks_.size() - 1; i >= 0; --i) {
        blocks_[i].backward(grad, grad);

        // Copy block's gradient tensors into param_grads_
        int base = 3 + i * blk_params;
        param_grads_[base + 0].copy_from(blocks_[i].get_grad_Wq().data(), blocks_[i].get_grad_Wq().numel());
        param_grads_[base + 1].copy_from(blocks_[i].get_grad_Wk().data(), blocks_[i].get_grad_Wk().numel());
        param_grads_[base + 2].copy_from(blocks_[i].get_grad_Wv().data(), blocks_[i].get_grad_Wv().numel());
        param_grads_[base + 3].copy_from(blocks_[i].get_grad_Wo().data(), blocks_[i].get_grad_Wo().numel());
        param_grads_[base + 4].copy_from(blocks_[i].get_grad_Wgate().data(), blocks_[i].get_grad_Wgate().numel());
        param_grads_[base + 5].copy_from(blocks_[i].get_grad_Wup().data(), blocks_[i].get_grad_Wup().numel());
        param_grads_[base + 6].copy_from(blocks_[i].get_grad_Wdown().data(), blocks_[i].get_grad_Wdown().numel());
        param_grads_[base + 7].copy_from(blocks_[i].get_grad_rms_weight_attn().data(), blocks_[i].get_grad_rms_weight_attn().numel());
        param_grads_[base + 8].copy_from(blocks_[i].get_grad_rms_weight_ffn().data(), blocks_[i].get_grad_rms_weight_ffn().numel());
    }
}

void Transformer::zero_grad() {
    for (auto& g : param_grads_)
        g.zeros();
}

int64_t Transformer::num_parameters() const {
    int64_t n = 0;
    for (auto* p : params_)
        n += p->numel();
    return n;
}

int64_t Transformer::get_param_grad_count() const {
    int64_t n = 0;
    for (auto& g : param_grads_)
        n += g.numel();
    return n;
}

std::vector<float*> Transformer::get_param_grad_pointers() {
    std::vector<float*> ptrs;
    for (auto& g : param_grads_)
        ptrs.push_back(g.data());
    return ptrs;
}

} // namespace lora_kernel
