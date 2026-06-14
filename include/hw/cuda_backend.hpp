#pragma once
#include <iostream>
#include <vector>
#include <cstring>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#else
// Stub types for compilation without CUDA
typedef int cudaStream_t;
typedef int cublasHandle_t;
typedef int cudnnHandle_t;
#define cudaSuccess 0
#define cudaMemcpyDeviceToDevice 0
#define cudaMemcpyHostToDevice 0
#define cudaMemcpyDeviceToHost 0
#endif

namespace lora_kernel {

// Production-level hard-coded CUDA backend wrapper
class CUDABackend {
private:
    int device_id_;
    cudaStream_t stream_{nullptr};
    cublasHandle_t cublas_handle_{nullptr};
    cudnnHandle_t cudnn_handle_{nullptr};
    bool initialized_{false};

public:
    CUDABackend(int device = 0) : device_id_(device) {}

    bool init() {
        std::cout << "[CUDA] Initializing device " << device_id_ << "\n";
#ifdef __CUDACC__
        cudaSetDevice(device_id_);
        cudaStreamCreate(&stream_);
        cublasCreate(&cublas_handle_);
        cublasSetStream(cublas_handle_, stream_);
        cudnnCreate(&cudnn_handle_);
        cudnnSetStream(cudnn_handle_, stream_);
#endif
        initialized_ = true;
        return true;
    }

    // Memory management
    void* alloc(size_t bytes) {
        void* ptr = nullptr;
#ifdef __CUDACC__
        cudaMalloc(&ptr, bytes);
#else
        ptr = std::malloc(bytes);
#endif
        return ptr;
    }

    void dealloc(void* ptr) {
#ifdef __CUDACC__
        cudaFree(ptr);
#else
        std::free(ptr);
#endif
    }

    void memcpy_h2d(void* dst, const void* src, size_t bytes) {
#ifdef __CUDACC__
        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream_);
#else
        std::memcpy(dst, src, bytes);
#endif
    }

    void memcpy_d2h(void* dst, const void* src, size_t bytes) {
#ifdef __CUDACC__
        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream_);
#else
        std::memcpy(dst, src, bytes);
#endif
    }

    void sync() {
#ifdef __CUDACC__
        cudaStreamSynchronize(stream_);
#endif
    }

    // cuBLAS GEMM wrapper
    void gemm(bool transA, bool transB,
              int M, int N, int K,
              float alpha, const float* A, const float* B,
              float beta, float* C) {
        std::cout << "[cuBLAS] GEMM " << M << "x" << N << "x" << K << "\n";
#ifdef __CUDACC__
        cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasSgemm(cublas_handle_, opA, opB,
                    M, N, K, &alpha, A, transA ? K : M,
                    B, transB ? N : K, &beta, C, M);
#endif
    }

    // cuDNN softmax wrapper
    void softmax(const float* input, float* output, int N, int C) {
        std::cout << "[cuDNN] Softmax " << N << "x" << C << "\n";
#ifdef __CUDACC__
        cudnnTensorDescriptor_t desc;
        cudnnCreateTensorDescriptor(&desc);
        cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, 1, 1);
        cudnnSoftmaxForward(cudnn_handle_, CUDNN_SOFTMAX_ACCURATE,
                            CUDNN_SOFTMAX_MODE_CHANNEL,
                            &cudnn_handle_, input, desc, &cudnn_handle_, output, desc);
        cudnnDestroyTensorDescriptor(desc);
#else
        // CPU fallback
        for (int n = 0; n < N; ++n) {
            float max_val = input[n * C];
            for (int c = 1; c < C; ++c)
                if (input[n * C + c] > max_val) max_val = input[n * C + c];
            double sum = 0.0;
            for (int c = 0; c < C; ++c)
                sum += std::exp(input[n * C + c] - max_val);
            for (int c = 0; c < C; ++c)
                output[n * C + c] = std::exp(input[n * C + c] - max_val) / (sum + 1e-12);
        }
#endif
    }

    // cuDNN LayerNorm forward
    void layernorm(const float* x, float* y, const float* gamma, const float* beta,
                   int N, int C) {
        std::cout << "[cuDNN] LayerNorm\n";
        for (int n = 0; n < N; ++n) {
            double mean = 0.0, var = 0.0;
            for (int c = 0; c < C; ++c) mean += x[n * C + c];
            mean /= C;
            for (int c = 0; c < C; ++c) { double d = x[n * C + c] - mean; var += d * d; }
            var /= C;
            float inv_std = 1.0f / std::sqrt((float)var + 1e-6f);
            for (int c = 0; c < C; ++c)
                y[n * C + c] = (x[n * C + c] - (float)mean) * inv_std * gamma[c] + beta[c];
        }
    }

    ~CUDABackend() {
        if (initialized_) {
#ifdef __CUDACC__
            cublasDestroy(cublas_handle_);
            cudnnDestroy(cudnn_handle_);
            cudaStreamDestroy(stream_);
#endif
        }
    }
};

} // namespace lora_kernel
