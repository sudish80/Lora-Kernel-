#pragma once
#include <iostream>
#include <vector>
#include <cstdint>

namespace lora_kernel {

class BoundaryProtectionGuard {
public:
    bool validate_token_array(const uint32_t* tokens, size_t length,
                              size_t max_allowed) const {
        if (!tokens) {
            std::cerr << "[BOUNDARY] Null pointer\n";
            return false;
        }
        if (length == 0 || length > max_allowed) {
            std::cerr << "[BOUNDARY] Invalid length: " << length << "\n";
            return false;
        }
        for (size_t i = 0; i < length; ++i) {
            if (tokens[i] == 0xFFFFFFFFu) {
                std::cerr << "[BOUNDARY] Corrupted token at " << i << "\n";
                return false;
            }
        }
        return true;
    }
};

class NumericShield {
private:
    static constexpr float MAX_VAL = 3.402823466e+38f;
    static constexpr float EPS     = 1e-7f;
public:
    void check_and_fix_tensor(float* data, size_t n) const {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            if (std::isnan(data[i]) || std::isinf(data[i])) { data[i] = 0.0f; continue; }
            if (std::abs(data[i]) < EPS) data[i] = 0.0f;
            if (data[i] >  MAX_VAL) data[i] =  MAX_VAL;
            if (data[i] < -MAX_VAL) data[i] = -MAX_VAL;
        }
    }
    bool has_nan(const float* data, size_t n) const {
        for (size_t i = 0; i < n; ++i) if (std::isnan(data[i])) return true;
        return false;
    }
};

} // namespace lora_kernel
