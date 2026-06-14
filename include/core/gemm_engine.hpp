#pragma once
#include <vector>
#include <cstring>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <algorithm>

#ifdef __CUDACC__
#include <cublas_v2.h>
#endif

#ifdef __linux__
#include <dlfcn.h>
#endif

namespace lora_kernel {

// Production-level hard-coded high-performance GEMM with BLAS integration
class GemmEngine {
private:
    // Autotuned block sizes for different cache levels
    static constexpr int L1_BLOCK = 32;   // L1 cache tile
    static constexpr int L2_BLOCK = 128;  // L2 cache tile
    static constexpr int L3_BLOCK = 256;  // L3 cache tile
    static constexpr int REG_BLOCK = 4;   // Register blocking

public:
    // Production matmul: tries cuBLAS, falls back to tiled CPU
    static void matmul(const float* A, const float* B, float* C,
                       int M, int N, int K) {
        // Validate inputs
        if (!A || !B || !C) throw std::runtime_error("GEMM: null pointer");
        if (M <= 0 || N <= 0 || K <= 0) throw std::runtime_error("GEMM: invalid dims");

#ifdef __CUDACC__
        static bool cublas_tried = false;
        static bool cublas_ok = false;
        static cublasHandle_t handle;

        if (!cublas_tried) {
            cublas_tried = true;
            cublas_ok = (cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS);
        }
        if (cublas_ok) {
            float alpha = 1.0f, beta = 0.0f;
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       M, N, K, &alpha, A, M, B, K, &beta, C, M);
            return;
        }
#endif

        // CPU fallback: cache-blocked tiled matmul
        tiled_matmul(A, B, C, M, N, K);
    }

    // Tiled CPU matmul with register blocking
    static void tiled_matmul(const float* A, const float* B, float* C,
                              int M, int N, int K) {
        std::memset(C, 0, M * N * sizeof(float));

        #pragma omp parallel for collapse(2)
        for (int i0 = 0; i0 < M; i0 += L2_BLOCK) {
            for (int j0 = 0; j0 < N; j0 += L2_BLOCK) {
                int i_end = std::min(i0 + L2_BLOCK, M);
                int j_end = std::min(j0 + L2_BLOCK, N);

                for (int k0 = 0; k0 < K; k0 += L3_BLOCK) {
                    int k_end = std::min(k0 + L3_BLOCK, K);

                    // Micro-kernel with register blocking
                    for (int i = i0; i < i_end; i += REG_BLOCK) {
                        for (int k = k0; k < k_end; k += REG_BLOCK) {
                            for (int j = j0; j < j_end; ++j) {
                                // Accumulate in registers
                                float c[REG_BLOCK][REG_BLOCK] = {{0}};

                                // Inner product with register tiling
                                for (int ri = 0; ri < REG_BLOCK && i + ri < i_end; ++ri) {
                                    for (int rk = 0; rk < REG_BLOCK && k + rk < k_end; ++rk) {
                                        float a_val = A[(i + ri) * K + (k + rk)];
                                        for (int rj = 0; rj < REG_BLOCK && j + rj < j_end; ++rj) {
                                            c[ri][rj] += a_val * B[(k + rk) * N + (j + rj)];
                                        }
                                    }
                                }

                                // Write back
                                for (int ri = 0; ri < REG_BLOCK && i + ri < i_end; ++ri)
                                    for (int rj = 0; rj < REG_BLOCK && j + rj < j_end; ++rj)
                                        C[(i + ri) * N + (j + rj)] += c[ri][rj];
                            }
                        }
                    }
                }
            }
        }
    }

    // Production batched GEMM for multi-head attention
    static void batched_gemm(const float* A, const float* B, float* C,
                              int batch, int M, int N, int K) {
        #pragma omp parallel for
        for (int b = 0; b < batch; ++b) {
            matmul(A + b * M * K, B + b * K * N,
                   C + b * M * N, M, N, K);
        }
    }
};

} // namespace lora_kernel
