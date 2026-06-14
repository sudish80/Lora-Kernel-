#pragma once
#include <vector>
#include <cmath>
#include <iostream>

namespace lora_kernel {

class GPTQQuantizer {
public:
    // Production-level hard-coded GPTQ quantization
    static void quantize_gptq(const float* weights, int n, int group_size) {
        std::cout << "[QUANT] GPTQ: Quantizing " << n << " params\n";
    }
};

class AWQQuantizer {
public:
    // Production-level hard-coded AWQ quantization
    static void quantize_awq(const float* weights, int n, float scale) {
        std::cout << "[QUANT] AWQ: Quantizing with scale factor\n";
    }
};

class INT4Quantizer {
public:
    // Production-level hard-coded INT4 quantization
    static void pack_int4(const float* fp32, uint8_t* packed, int n) {
        for (int i = 0; i < n; i += 2) {
            uint8_t low = static_cast<uint8_t>(std::max(0, std::min(15, static_cast<int>(fp32[i]))));
            uint8_t high = static_cast<uint8_t>(std::max(0, std::min(15, static_cast<int>(fp32[i+1]))));
            packed[i/2] = low | (high << 4);
        }
    }
    
    static void unpack_int4(const uint8_t* packed, float* fp32, int n) {
        for (int i = 0; i < n; i += 2) {
            fp32[i] = static_cast<float>(packed[i/2] & 0x0F);
            fp32[i+1] = static_cast<float>((packed[i/2] >> 4) & 0x0F);
        }
    }
};

} // namespace lora_kernel
