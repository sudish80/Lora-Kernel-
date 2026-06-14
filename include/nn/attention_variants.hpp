#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <iostream>
#include "include/core/tensor.hpp"

namespace lora_kernel {

// ======================================================================
// 31. Multi-Query Attention (MQA): single KV head shared across all Q heads
// ======================================================================
class MultiQueryAttention {
public:
    static void forward(const float* Q, const float* K, const float* V,
                        float* O, int B, int H, int S, int D) {
        // K, V are [B, 1, S, D] (single head), broadcast over H
        int KV_heads = 1;
        float inv_scale = 1.0f / std::sqrt((float)D);

        #pragma omp parallel for collapse(2)
        for (int b = 0; b < B; ++b) {
            for (int h = 0; h < H; ++h) {
                for (int i = 0; i < S; ++i) {
                    float max_val = -1e9f;
                    for (int j = 0; j < S; ++j) {
                        float score = 0;
                        for (int d = 0; d < D; ++d)
                            score += Q[b * H * S * D + h * S * D + i * D + d] *
                                     K[b * KV_heads * S * D + 0 * S * D + j * D + d];
                        score *= inv_scale;
                        if (j <= i) { // causal
                            max_val = std::max(max_val, score);
                        }
                    }
                    // Softmax + weighted sum (simplified)
                    float sum_exp = 0;
                    for (int j = 0; j <= i; ++j) {
                        float score = 0;
                        for (int d = 0; d < D; ++d)
                            score += Q[b * H * S * D + h * S * D + i * D + d] *
                                     K[b * KV_heads * S * D + 0 * S * D + j * D + d];
                        score = std::exp(score * inv_scale - max_val);
                        sum_exp += score;
                        for (int d = 0; d < D; ++d)
                            O[b * H * S * D + h * S * D + i * D + d] +=
                                (score / sum_exp) * V[b * KV_heads * S * D + 0 * S * D + j * D + d];
                    }
                }
            }
        }
    }
};

// ======================================================================
// 32. Grouped-Query Attention (GQA): KV heads = Q heads / group_size
// ======================================================================
class GroupedQueryAttention {
public:
    static void forward(const float* Q, const float* K, const float* V,
                        float* O, int B, int H, int G, int S, int D) {
        // H Q heads, G KV heads. Each KV head serves H/G Q heads.
        int kv_heads = G;
        int q_per_kv = H / G;
        float inv_scale = 1.0f / std::sqrt((float)D);

        #pragma omp parallel for collapse(2)
        for (int b = 0; b < B; ++b)
            for (int g = 0; g < G; ++g)
                for (int i = 0; i < S; ++i) {
                    float max_val = -1e9f;
                    for (int j = 0; j <= i; ++j) {
                        float score = 0;
                        for (int d = 0; d < D; ++d)
                            score += Q[b * H * S * D + (g * q_per_kv) * S * D + i * D + d] *
                                     K[b * kv_heads * S * D + g * S * D + j * D + d];
                        score *= inv_scale;
                        max_val = std::max(max_val, score);
                    }
                    // Apply to all Q heads in this group
                    for (int qh = 0; qh < q_per_kv; ++qh) {
                        float sum_exp = 0;
                        for (int j = 0; j <= i; ++j) {
                            float score = 0;
                            for (int d = 0; d < D; ++d)
                                score += Q[b * H * S * D + (g * q_per_kv + qh) * S * D + i * D + d] *
                                         K[b * kv_heads * S * D + g * S * D + j * D + d];
                            score = std::exp(score * inv_scale - max_val);
                            sum_exp += score;
                            for (int d = 0; d < D; ++d)
                                O[b * H * S * D + (g * q_per_kv + qh) * S * D + i * D + d] +=
                                    (score / sum_exp) * V[b * kv_heads * S * D + g * S * D + j * D + d];
                        }
                    }
                }
    }
};

// ======================================================================
// 33. Sliding Window Attention
// ======================================================================
class SlidingWindowAttention {
private:
    int window_size_{512};
public:
    SlidingWindowAttention(int window = 512) : window_size_(window) {}

    void forward(const float* Q, const float* K, const float* V, float* O,
                 int B, int H, int S, int D) {
        float inv_scale = 1.0f / std::sqrt((float)D);
        #pragma omp parallel for collapse(2)
        for (int b = 0; b < B; ++b)
            for (int h = 0; h < H; ++h)
                for (int i = 0; i < S; ++i) {
                    int start = std::max(0, i - window_size_);
                    float max_val = -1e9f;
                    for (int j = start; j <= i; ++j) {
                        float score = 0;
                        for (int d = 0; d < D; ++d)
                            score += Q[b*H*S*D + h*S*D + i*D + d] *
                                     K[b*H*S*D + h*S*D + j*D + d];
                        max_val = std::max(max_val, score * inv_scale);
                    }
                    float sum_exp = 0;
                    float weighted[D];
                    std::memset(weighted, 0, D * sizeof(float));
                    for (int j = start; j <= i; ++j) {
                        float score = 0;
                        for (int d = 0; d < D; ++d)
                            score += Q[b*H*S*D + h*S*D + i*D + d] *
                                     K[b*H*S*D + h*S*D + j*D + d];
                        float w = std::exp(score * inv_scale - max_val);
                        sum_exp += w;
                        for (int d = 0; d < D; ++d)
                            weighted[d] += w * V[b*H*S*D + h*S*D + j*D + d];
                    }
                    for (int d = 0; d < D; ++d)
                        O[b*H*S*D + h*S*D + i*D + d] = weighted[d] / (sum_exp + 1e-12f);
                }
    }
};

// ======================================================================
// 34. Dilated Attention
// ======================================================================
class DilatedAttention {
private:
    int dilation_{2};
public:
    DilatedAttention(int dilation = 2) : dilation_(dilation) {}

    void forward(const float* Q, const float* K, const float* V, float* O,
                 int B, int H, int S, int D) {
        float inv_scale = 1.0f / std::sqrt((float)D);
        #pragma omp parallel for collapse(2)
        for (int b = 0; b < B; ++b)
            for (int h = 0; h < H; ++h)
                for (int i = 0; i < S; ++i) {
                    float max_val = -1e9f;
                    // Attend to every `dilation`-th previous token
                    for (int j = (i % dilation_); j <= i; j += dilation_) {
                        float score = 0;
                        for (int d = 0; d < D; ++d)
                            score += Q[b*H*S*D + h*S*D + i*D + d] *
                                     K[b*H*S*D + h*S*D + j*D + d];
                        max_val = std::max(max_val, score * inv_scale);
                    }
                    float sum_exp = 0;
                    float weighted[D] = {0};
                    for (int j = (i % dilation_); j <= i; j += dilation_) {
                        float score = 0;
                        for (int d = 0; d < D; ++d)
                            score += Q[b*H*S*D + h*S*D + i*D + d] *
                                     K[b*H*S*D + h*S*D + j*D + d];
                        float w = std::exp(score * inv_scale - max_val);
                        sum_exp += w;
                        for (int d = 0; d < D; ++d)
                            weighted[d] += w * V[b*H*S*D + h*S*D + j*D + d];
                    }
                    for (int d = 0; d < D; ++d)
                        O[b*H*S*D + h*S*D + i*D + d] = weighted[d] / (sum_exp + 1e-12f);
                }
    }
};

// ======================================================================
// 35. Cross-Attention (Encoder-Decoder)
// ======================================================================
class CrossAttention {
public:
    static void forward(const float* Q, const float* K, const float* V,
                        float* O, int B, int H, int Q_len, int KV_len, int D) {
        float inv_scale = 1.0f / std::sqrt((float)D);
        #pragma omp parallel for collapse(2)
        for (int b = 0; b < B; ++b)
            for (int h = 0; h < H; ++h) {
                for (int i = 0; i < Q_len; ++i) {
                    float max_val = -1e9f;
                    for (int j = 0; j < KV_len; ++j) {
                        float score = 0;
                        for (int d = 0; d < D; ++d)
                            score += Q[b*H*Q_len*D + h*Q_len*D + i*D + d] *
                                     K[b*H*KV_len*D + h*KV_len*D + j*D + d];
                        max_val = std::max(max_val, score * inv_scale);
                    }
                    float sum_exp = 0;
                    float weighted[D] = {0};
                    for (int j = 0; j < KV_len; ++j) {
                        float score = 0;
                        for (int d = 0; d < D; ++d)
                            score += Q[b*H*Q_len*D + h*Q_len*D + i*D + d] *
                                     K[b*H*KV_len*D + h*KV_len*D + j*D + d];
                        float w = std::exp(score * inv_scale - max_val);
                        sum_exp += w;
                        for (int d = 0; d < D; ++d)
                            weighted[d] += w * V[b*H*KV_len*D + h*KV_len*D + j*D + d];
                    }
                    for (int d = 0; d < D; ++d)
                        O[b*H*Q_len*D + h*Q_len*D + i*D + d] = weighted[d] / (sum_exp + 1e-12f);
                }
            }
    }
};

// ======================================================================
// 36. ALiBi Position Bias
// ======================================================================
class ALiBi {
public:
    static void apply(float* scores, int H, int S, float slope_base = 0.5f) {
        // ALiBi: bias = -slope * |i - j|
        for (int h = 0; h < H; ++h) {
            float slope = 1.0f / std::pow(2.0f, (float)(h + 1));
            for (int i = 0; i < S; ++i)
                for (int j = 0; j < S; ++j)
                    scores[h * S * S + i * S + j] += slope * (j - i);
        }
    }
};

// ======================================================================
// 37. Relative Position Bias
// ======================================================================
class RelativePositionBias {
private:
    std::vector<float> bias_table_;
    int max_distance_;
public:
    RelativePositionBias(int max_dist = 128, int num_heads = 12)
        : max_distance_(max_dist) {
        bias_table_.resize(num_heads * (2 * max_dist - 1));
        for (size_t i = 0; i < bias_table_.size(); ++i)
            bias_table_[i] = std::sin((float)i / bias_table_.size() * 2 * M_PI);
    }

    void apply(float* scores, int H, int S) {
        for (int h = 0; h < H; ++h)
            for (int i = 0; i < S; ++i)
                for (int j = 0; j < S; ++j) {
                    int rel = j - i + max_distance_ - 1;
                    if (rel >= 0 && rel < 2 * max_distance_ - 1)
                        scores[h * S * S + i * S + j] +=
                            bias_table_[h * (2 * max_distance_ - 1) + rel];
                }
    }
};

// ======================================================================
// 38. Top-K Sparse Attention with Block Sparsity
// ======================================================================
class BlockSparseAttention {
private:
    int block_size_{32};
    int top_k_blocks_{4};
public:
    BlockSparseAttention(int block = 32, int topk = 4)
        : block_size_(block), top_k_blocks_(topk) {}

    void forward(const float* Q, const float* K, const float* V, float* O,
                 int B, int H, int S, int D) {
        int num_blocks = (S + block_size_ - 1) / block_size_;
        float inv_scale = 1.0f / std::sqrt((float)D);

        #pragma omp parallel for collapse(2)
        for (int b = 0; b < B; ++b)
            for (int h = 0; h < H; ++h) {
                // Compute block-level scores (average over block)
                float block_scores[num_blocks * num_blocks] = {0};
                for (int bi = 0; bi < num_blocks; ++bi)
                    for (int bj = 0; bj < num_blocks; ++bj)
                        for (int i = bi * block_size_; i < std::min((bi + 1) * block_size_, S); ++i)
                            for (int j = bj * block_size_; j < std::min((bj + 1) * block_size_, S); ++j) {
                                float score = 0;
                                for (int d = 0; d < D; ++d)
                                    score += Q[b*H*S*D + h*S*D + i*D + d] *
                                             K[b*H*S*D + h*S*D + j*D + d];
                                block_scores[bi * num_blocks + bj] += score;
                            }

                // Keep top-K blocks per row
                // (simplified: use dense attention within top blocks)
                std::memset(O + b*H*S*D + h*S*D, 0, S * D * sizeof(float));

                for (int bi = 0; bi < num_blocks; ++bi) {
                    // Find top-K block columns
                    std::vector<std::pair<float, int>> sorted;
                    for (int bj = 0; bj <= bi; ++bj)
                        sorted.push_back({block_scores[bi * num_blocks + bj], bj});
                    std::sort(sorted.begin(), sorted.end(),
                              std::greater<std::pair<float, int>>());

                    // Attend only within top-K blocks (plus causal masking)
                    for (int k = 0; k < std::min(top_k_blocks_, (int)sorted.size()); ++k) {
                        int bj = sorted[k].second;
                        for (int i = bi * block_size_; i < std::min((bi + 1) * block_size_, S); ++i)
                            for (int j = bj * block_size_; j < std::min((bj + 1) * block_size_, i + 1); ++j) {
                                float score = 0;
                                for (int d = 0; d < D; ++d)
                                    score += Q[b*H*S*D + h*S*D + i*D + d] *
                                             K[b*H*S*D + h*S*D + j*D + d];
                                O[b*H*S*D + h*S*D + i*D + 0] += score; // simplified
                            }
                    }
                }
            }
    }
};

// ======================================================================
// 39. Local-Global Hybrid Attention
// ======================================================================
class HybridAttention {
private:
    int local_window_{64};
    int global_interval_{16};
public:
    HybridAttention(int local = 64, int global = 16)
        : local_window_(local), global_interval_(global) {}

    void forward(const float* Q, const float* K, const float* V, float* O,
                 int B, int H, int S, int D) {
        float inv_scale = 1.0f / std::sqrt((float)D);
        #pragma omp parallel for collapse(2)
        for (int b = 0; b < B; ++b)
            for (int h = 0; h < H; ++h)
                for (int i = 0; i < S; ++i) {
                    // Local: attend to nearby tokens
                    int local_start = std::max(0, i - local_window_);
                    // Global: also attend to special global tokens (every interval)
                    std::memset(O + b*H*S*D + h*S*D + i*D, 0, D * sizeof(float));

                    float max_val = -1e9f;
                    for (int j = local_start; j <= i; ++j) {
                        float score = 0;
                        for (int d = 0; d < D; ++d)
                            score += Q[b*H*S*D + h*S*D + i*D + d] *
                                     K[b*H*S*D + h*S*D + j*D + d];
                        max_val = std::max(max_val, score * inv_scale);
                    }
                    // Also attend to global tokens (stride)
                    for (int j = 0; j <= i; j += global_interval_) {
                        float score = 0;
                        for (int d = 0; d < D; ++d)
                            score += Q[b*H*S*D + h*S*D + i*D + d] *
                                     K[b*H*S*D + h*S*D + j*D + d];
                        max_val = std::max(max_val, score * inv_scale);
                    }
                    // Softmax + weighted sum... (simplified)
                }
    }
};

// ======================================================================
// 40. Linear Attention (Performer-style with feature map)
// ======================================================================
class LinearAttention {
public:
    static void forward(const float* Q, const float* K, const float* V,
                        float* O, int B, int H, int S, int D) {
        // Feature map: phi(x) = ELU(x) + 1
        // O = phi(Q) @ (phi(K)^T @ V) / (phi(Q) @ phi(K)^T sum)

        #pragma omp parallel for collapse(2)
        for (int b = 0; b < B; ++b)
            for (int h = 0; h < H; ++h) {
                // Apply feature map to K, V sum
                float KV[D][D] = {{0}};
                float K_sum[D] = {0};

                for (int j = 0; j < S; ++j) {
                    for (int d = 0; d < D; ++d) {
                        float phi_k = (K[b*H*S*D + h*S*D + j*D + d] > 0) ?
                                      K[b*H*S*D + h*S*D + j*D + d] + 1.0f :
                                      std::exp(K[b*H*S*D + h*S*D + j*D + d]);
                        K_sum[d] += phi_k;
                        for (int d2 = 0; d2 < D; ++d2)
                            KV[d][d2] += phi_k * V[b*H*S*D + h*S*D + j*D + d2];
                    }
                }

                // Compute output
                for (int i = 0; i < S; ++i) {
                    float denom = 0;
                    for (int d = 0; d < D; ++d) {
                        float phi_q = (Q[b*H*S*D + h*S*D + i*D + d] > 0) ?
                                      Q[b*H*S*D + h*S*D + i*D + d] + 1.0f :
                                      std::exp(Q[b*H*S*D + h*S*D + i*D + d]);
                        denom += phi_q * K_sum[d];
                    }
                    for (int d2 = 0; d2 < D; ++d2) {
                        float sum = 0;
                        for (int d = 0; d < D; ++d) {
                            float phi_q = (Q[b*H*S*D + h*S*D + i*D + d] > 0) ?
                                          Q[b*H*S*D + h*S*D + i*D + d] + 1.0f :
                                          std::exp(Q[b*H*S*D + h*S*D + i*D + d]);
                            sum += phi_q * KV[d][d2];
                        }
                        O[b*H*S*D + h*S*D + i*D + d2] = sum / (denom + 1e-12f);
                    }
                }
            }
    }
};

} // namespace lora_kernel
