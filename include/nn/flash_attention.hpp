#pragma once
#include <cmath>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

namespace lora_kernel {

class FlashAttentionV2 {
public:
    static void forward(const float* Q, const float* K, const float* V,
                        float* O, int B, int H, int S, int D) {
        const int Br = 32;
        const int Bc = 32;
        float inv_sqrt_D = 1.0f / std::sqrt((float)D);

        static thread_local std::vector<float> m, l, acc;
        m.resize(S);
        l.resize(S);
        acc.resize(S * D);

        #pragma omp parallel for collapse(2)
        for (int b = 0; b < B; ++b) {
            for (int h = 0; h < H; ++h) {
                int bh = b * H + h;
                const float* q_ptr = Q + bh * S * D;
                const float* k_ptr = K + bh * S * D;
                const float* v_ptr = V + bh * S * D;
                float* o_ptr = O + bh * S * D;

                std::fill(m.begin(), m.end(), -1e9f);
                std::fill(l.begin(), l.end(), 0.0f);
                std::memset(acc.data(), 0, S * D * sizeof(float));

                for (int j_start = 0; j_start < S; j_start += Bc) {
                    int j_end = std::min(j_start + Bc, S);
                    int tj = j_end - j_start;

                    const float* K_tile = k_ptr + j_start * D;
                    const float* V_tile = v_ptr + j_start * D;

                    for (int i_start = 0; i_start < S; i_start += Br) {
                        int i_end = std::min(i_start + Br, S);
                        int ti = i_end - i_start;

                        static thread_local std::vector<float> S_ij;
                        S_ij.resize(ti * tj);
                        for (int i = 0; i < ti; ++i)
                            for (int j = 0; j < tj; ++j) {
                                float sum = 0.0f;
                                for (int d = 0; d < D; ++d)
                                    sum += q_ptr[(i_start + i) * D + d] * K_tile[j * D + d];
                                S_ij[i * tj + j] = sum * inv_sqrt_D;
                            }

                        for (int i = 0; i < ti; ++i)
                            for (int j = 0; j < tj; ++j)
                                if (i_start + i < j_start + j)
                                    S_ij[i * tj + j] = -1e9f;

                        for (int i = 0; i < ti; ++i) {
                            int gi = i_start + i;
                            float m_old = m[gi];
                            float m_new = m_old;

                            for (int j = 0; j < tj; ++j)
                                m_new = std::max(m_new, S_ij[i * tj + j]);

                            float l_new = 0.0f;
                            for (int j = 0; j < tj; ++j)
                                l_new += std::exp(S_ij[i * tj + j] - m_new);

                            float rescale = l[gi] * std::exp(m_old - m_new);
                            float norm = 1.0f / (rescale + l_new + 1e-12f);
                            for (int d = 0; d < D; ++d)
                                acc[gi * D + d] *= rescale * norm;

                            for (int d = 0; d < D; ++d) {
                                float sum = 0.0f;
                                for (int j = 0; j < tj; ++j)
                                    sum += std::exp(S_ij[i * tj + j] - m_new) * V_tile[j * D + d];
                                acc[gi * D + d] += sum * norm;
                            }

                            l[gi] = l_new;
                            m[gi] = m_new;
                        }
                    }
                }

                std::memcpy(o_ptr, acc.data(), S * D * sizeof(float));
            }
        }
    }
};

} // namespace lora_kernel
