#pragma once
#include <cstdint>
#include <cmath>

namespace lora_kernel {

// BF16 (bfloat16) support
struct bf16 {
    uint16_t bits;
    
    bf16() : bits(0) {}
    bf16(float f) {
        uint32_t i;
        memcpy(&i, &f, sizeof(i));
        bits = i >> 16;
    }
    
    operator float() const {
        uint32_t i = static_cast<uint32_t>(bits) << 16;
        float f;
        memcpy(&f, &i, sizeof(f));
        return f;
    }
    
    bf16 operator+(const bf16& other) const {
        return bf16(float(*this) + float(other));
    }
};

class BF16Training {
public:
    static void convert_to_bf16(const float* src, bf16* dst, int n) {
        for (int i = 0; i < n; ++i) dst[i] = bf16(src[i]);
    }
    
    static void convert_from_bf16(const bf16* src, float* dst, int n) {
        for (int i = 0; i < n; ++i) dst[i] = float(src[i]);
    }
};

} // namespace lora_kernel
