#pragma once
#include <iostream>
#include <omp.h>
#include <openssl/crypto.h>

namespace lora_kernel {

class CompilationValidator {
public:
    void print_report() const {
        std::cout << "\n=== Compilation Report ===\n";
#ifdef __AVX2__
        std::cout << "AVX2  : ENABLED\n";
#else
        std::cout << "AVX2  : DISABLED\n";
#endif
#ifdef _OPENMP
        std::cout << "OpenMP: ENABLED (" << omp_get_max_threads() << " threads)\n";
#else
        std::cout << "OpenMP: DISABLED\n";
#endif
        std::cout << "OpenSSL: " << OpenSSL_version(OPENSSL_VERSION) << "\n";
    }
};

} // namespace lora_kernel
