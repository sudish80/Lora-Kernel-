#pragma once
#include <vector>
#include <cmath>
#include <memory>
#include "include/core/tensor.hpp"
#include "include/nn/weight_init.hpp"
#include "include/nn/kv_cache.hpp"

namespace lora_kernel {

struct TransformerConfig {
    int hidden_dim;
    int num_heads;
    int head_dim;
    int vocab_size;
    int max_seq_len;
    int num_layers;
    float dropout;
    int ff_dim;
};

class MultiHeadAttention {
private:
    int hidden_dim_, num_heads_, head_dim_, max_seq_len_;
    Tensor Wq_, Wk_, Wv_, Wo_;
    Tensor bias_q_, bias_k_, bias_v_, bias_o_;

    Tensor last_input_;
    Tensor last_Q_, last_K_, last_V_;
    Tensor last_scores_;
    Tensor last_softmax_;
    Tensor last_context_;

public:
    MultiHeadAttention(const TransformerConfig& cfg)
        : hidden_dim_(cfg.hidden_dim), num_heads_(cfg.num_heads),
          head_dim_(cfg.head_dim), max_seq_len_(cfg.max_seq_len),
          Wq_({hidden_dim_, hidden_dim_}), Wk_({hidden_dim_, hidden_dim_}),
          Wv_({hidden_dim_, hidden_dim_}), Wo_({hidden_dim_, hidden_dim_}),
          bias_q_({hidden_dim_}), bias_k_({hidden_dim_}),
          bias_v_({hidden_dim_}), bias_o_({hidden_dim_}) {
        WeightInit::kaiming_normal(Wq_, hidden_dim_);
        WeightInit::kaiming_normal(Wk_, hidden_dim_);
        WeightInit::kaiming_normal(Wv_, hidden_dim_);
        WeightInit::kaiming_normal(Wo_, hidden_dim_);
    }

    void forward(const Tensor& input, Tensor& output);
    void backward(const Tensor& grad_output, Tensor& grad_input,
                  Tensor& grad_Wq, Tensor& grad_Wk, Tensor& grad_Wv, Tensor& grad_Wo);

    const Tensor& get_Wq() const { return Wq_; }
    const Tensor& get_Wk() const { return Wk_; }
    const Tensor& get_Wv() const { return Wv_; }
    const Tensor& get_Wo() const { return Wo_; }
    const Tensor& get_last_K() const { return last_K_; }
    const Tensor& get_last_V() const { return last_V_; }
    Tensor* get_Wq_ptr() { return &Wq_; }
    Tensor* get_Wk_ptr() { return &Wk_; }
    Tensor* get_Wv_ptr() { return &Wv_; }
    Tensor* get_Wo_ptr() { return &Wo_; }
};

class FeedForward {
private:
    int hidden_dim_, ff_dim_;
    Tensor Wgate_, Wup_, Wdown_;
    Tensor bias_gate_, bias_up_, bias_down_;

    Tensor last_input_;
    Tensor last_gate_, last_up_, last_gated_;
    Tensor last_swish_;

public:
    FeedForward(int hidden_dim, int ff_dim)
        : hidden_dim_(hidden_dim), ff_dim_(ff_dim),
          Wgate_({hidden_dim_, ff_dim_}), Wup_({hidden_dim_, ff_dim_}),
          Wdown_({ff_dim_, hidden_dim_}),
          bias_gate_({ff_dim_}), bias_up_({ff_dim_}), bias_down_({hidden_dim_}) {
        WeightInit::kaiming_normal(Wgate_, hidden_dim_);
        WeightInit::kaiming_normal(Wup_, hidden_dim_);
        WeightInit::kaiming_normal(Wdown_, ff_dim_);
    }

    void forward(const Tensor& input, Tensor& output);
    void backward(const Tensor& grad_output, Tensor& grad_input,
                  Tensor& grad_Wgate, Tensor& grad_Wup, Tensor& grad_Wdown);

    const Tensor& get_Wgate() const { return Wgate_; }
    const Tensor& get_Wup() const { return Wup_; }
    const Tensor& get_Wdown() const { return Wdown_; }
    Tensor* get_Wgate_ptr() { return &Wgate_; }
    Tensor* get_Wup_ptr() { return &Wup_; }
    Tensor* get_Wdown_ptr() { return &Wdown_; }
};

class TransformerBlock {
private:
    MultiHeadAttention attn_;
    FeedForward ffn_;
    float eps_{1e-6f};

    Tensor Wq_, Wk_, Wv_, Wo_;
    Tensor Wgate_, Wup_, Wdown_;
    Tensor rms_weight_attn_, rms_weight_ffn_;

    Tensor grad_Wq_, grad_Wk_, grad_Wv_, grad_Wo_;
    Tensor grad_Wgate_, grad_Wup_, grad_Wdown_;
    Tensor grad_rms_weight_attn_, grad_rms_weight_ffn_;

    Tensor last_input_;
    Tensor last_norm1_out_;
    Tensor last_attn_out_;
    Tensor last_norm2_out_;

    PagedKVCache* kv_cache_{nullptr};

public:
    std::vector<float*> grad_weight_ptrs_;

    TransformerBlock(const TransformerConfig& cfg);

    void forward(const Tensor& input, Tensor& output,
                 PagedKVCache* cache = nullptr, int token_pos = 0);
    void backward(const Tensor& grad_output, Tensor& grad_input);

    void set_kv_cache(PagedKVCache* cache) { kv_cache_ = cache; }
    PagedKVCache* get_kv_cache() const { return kv_cache_; }

    Tensor* get_Wq_ptr() { return &Wq_; }
    Tensor* get_Wk_ptr() { return &Wk_; }
    Tensor* get_Wv_ptr() { return &Wv_; }
    Tensor* get_Wo_ptr() { return &Wo_; }
    Tensor* get_Wgate_ptr() { return &Wgate_; }
    Tensor* get_Wup_ptr() { return &Wup_; }
    Tensor* get_Wdown_ptr() { return &Wdown_; }
    Tensor* get_rms_weight_attn_ptr() { return &rms_weight_attn_; }
    Tensor* get_rms_weight_ffn_ptr() { return &rms_weight_ffn_; }

    const Tensor& get_grad_Wq() const { return grad_Wq_; }
    const Tensor& get_grad_Wk() const { return grad_Wk_; }
    const Tensor& get_grad_Wv() const { return grad_Wv_; }
    const Tensor& get_grad_Wo() const { return grad_Wo_; }
    const Tensor& get_grad_Wgate() const { return grad_Wgate_; }
    const Tensor& get_grad_Wup() const { return grad_Wup_; }
    const Tensor& get_grad_Wdown() const { return grad_Wdown_; }
    const Tensor& get_grad_rms_weight_attn() const { return grad_rms_weight_attn_; }
    const Tensor& get_grad_rms_weight_ffn() const { return grad_rms_weight_ffn_; }

    const std::vector<float*>& get_grad_ptrs() const { return grad_weight_ptrs_; }
};

class Transformer {
private:
    Tensor token_embedding_, pos_embedding_;
    Tensor lm_head_;
    std::vector<TransformerBlock> blocks_;
    TransformerConfig cfg_;
    Tensor last_hidden_;

    std::vector<Tensor*> params_;
    std::vector<Tensor> param_grads_;

public:
    Transformer(const TransformerConfig& cfg);

    void forward(const Tensor& input_ids, Tensor& logits);
    void backward(const Tensor& grad_logits);
    void zero_grad();

    int64_t num_parameters() const;
    int64_t get_param_grad_count() const;
    std::vector<float*> get_param_grad_pointers();
    int64_t vocab_size() const { return cfg_.vocab_size; }

    const std::vector<Tensor*>& get_params() const { return params_; }
    std::vector<Tensor>& get_param_grads() { return param_grads_; }
    TransformerConfig get_config() const { return cfg_; }
};

} // namespace lora_kernel
