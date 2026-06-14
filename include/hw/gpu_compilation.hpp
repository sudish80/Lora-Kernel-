#pragma once
#include <iostream>

// Production GPU compilation support with backend detection
namespace lora_kernel {

class GPUCompilationSupport {
public:
    // CUDA support
    static bool has_cuda() {
#ifdef __CUDACC__
        return true;
#else
        return false;
#endif
    }

    // ROCm/HIP support
    static bool has_rocm() {
#ifdef __HIP_PLATFORM_AMD__
        return true;
#else
        return false;
#endif
    }

    // oneMKL detection
    static bool has_onemkl() {
#ifdef INTEL_MKL_VERSION
        return true;
#else
        return false;
#endif
    }

    // cuBLAS detection
    static bool has_cublas() {
#ifdef __CUDACC__
        return true;
#else
        return false;
#endif
    }

    // cuDNN detection
    static bool has_cudnn() {
#if defined(__CUDACC__) && defined(CUDNN_VERSION)
        return true;
#else
        return false;
#endif
    }

    static void print_report() {
        std::cout << "=== GPU Compilation Report ===\n";
        std::cout << "CUDA   : " << (has_cuda() ? "ENABLED" : "DISABLED") << "\n";
        std::cout << "ROCm   : " << (has_rocm() ? "ENABLED" : "DISABLED") << "\n";
        std::cout << "oneMKL : " << (has_onemkl() ? "ENABLED" : "DISABLED") << "\n";
        std::cout << "cuBLAS : " << (has_cublas() ? "ENABLED" : "DISABLED") << "\n";
        std::cout << "cuDNN  : " << (has_cudnn() ? "ENABLED" : "DISABLED") << "\n";
    }
};

#ifdef INTEL_MKL_VERSION
#include <mkl.h>
// MKL GEMM wrapper
inline void mkl_gemm(const float* A, const float* B, float* C,
                      int M, int N, int K) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
}
#endif

} // namespace lora_kernel
