#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>

namespace lora_kernel {

typedef uint16_t fp16;

class MixedPrecision {
public:
    enum class DType { FP32, FP16, BF16, FP8_E4M3 };

    MixedPrecision(DType dtype = DType::FP16) : dtype_(dtype) {}

    DType get_actual_dtype() const { return dtype_; }
    void set_dtype(DType dt) { dtype_ = dt; }

    void allocate_master_weights(int64_t n) {
        master_weights_.resize(n);
    }

    float* get_master_weights() { return master_weights_.data(); }
    const float* get_master_weights() const { return master_weights_.data(); }
    int64_t master_weight_count() const { return (int64_t)master_weights_.size(); }

    void copy_to_master(const float* src, int64_t n) {
        if ((int64_t)master_weights_.size() < n) master_weights_.resize(n);
        std::memcpy(master_weights_.data(), src, n * sizeof(float));
    }

    // ============ FP16 conversion ============

    static void cast_to_fp16(const float* in, uint16_t* out, int64_t n) {
        for (int64_t i = 0; i < n; ++i)
            out[i] = float_to_half(in[i]);
    }

    static void cast_from_fp16(const uint16_t* in, float* out, int64_t n) {
        for (int64_t i = 0; i < n; ++i)
            out[i] = half_to_float(in[i]);
    }

    // ============ BF16 conversion ============

    static void cast_to_bf16(const float* in, uint16_t* out, int64_t n) {
        for (int64_t i = 0; i < n; ++i)
            out[i] = float_to_bf16(in[i]);
    }

    static void cast_from_bf16(const uint16_t* in, float* out, int64_t n) {
        for (int64_t i = 0; i < n; ++i)
            out[i] = bf16_to_float(in[i]);
    }

    // ============ FP8 E4M3 conversion ============

    static void cast_to_fp8_e4m3(const float* in, uint8_t* out, int64_t n) {
        for (int64_t i = 0; i < n; ++i)
            out[i] = float_to_fp8_e4m3(in[i]);
    }

    static void cast_from_fp8_e4m3(const uint8_t* in, float* out, int64_t n) {
        for (int64_t i = 0; i < n; ++i)
            out[i] = fp8_e4m3_to_float(in[i]);
    }

private:
    DType dtype_ = DType::FP16;
    std::vector<float> master_weights_;

    static uint16_t float_to_half(float f) {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        uint32_t sign = (bits >> 16) & 0x8000;
        uint32_t exp8 = (bits >> 23) & 0xff;
        uint32_t mant23 = bits & 0x7fffff;

        if (exp8 == 0xff) {
            if (mant23 != 0)
                return (uint16_t)(sign | 0x7e00 | (mant23 >> 13));
            return (uint16_t)(sign | 0x7c00);
        }
        if (exp8 == 0)
            return (uint16_t)sign;

        int32_t new_exp = (int32_t)exp8 - 127 + 15;
        if (new_exp >= 31)
            return (uint16_t)(sign | 0x7c00);
        if (new_exp <= 0) {
            if (new_exp <= -10) return (uint16_t)sign;
            uint32_t m = (mant23 | 0x800000) >> (1 - new_exp);
            return (uint16_t)(sign | (m >> 13));
        }

        uint32_t round_bit = (mant23 >> 12) & 1;
        uint32_t sticky = mant23 & 0xfff;
        uint32_t m10 = mant23 >> 13;
        if (round_bit && ((m10 & 1) || sticky != 0)) {
            m10++;
            if (m10 == 0x400) {
                m10 = 0;
                new_exp++;
                if (new_exp >= 31) return (uint16_t)(sign | 0x7c00);
            }
        }
        return (uint16_t)(sign | ((uint32_t)new_exp << 10) | m10);
    }

    static float half_to_float(uint16_t h) {
        uint32_t sign = (uint32_t)(h & 0x8000) << 16;
        uint32_t exp5 = (h >> 10) & 0x1f;
        uint32_t mant10 = h & 0x03ff;

        if (exp5 == 0x1f) {
            if (mant10 != 0) {
                uint32_t bits = sign | 0x7fc00000 | ((uint32_t)mant10 << 13);
                float f; std::memcpy(&f, &bits, sizeof(f)); return f;
            }
            uint32_t bits = sign | 0x7f800000;
            float f; std::memcpy(&f, &bits, sizeof(f)); return f;
        }
        if (exp5 == 0) {
            if (mant10 == 0) { float f; std::memcpy(&f, &sign, sizeof(f)); return f; }
            while ((mant10 & 0x0400) == 0) { mant10 <<= 1; exp5--; }
            exp5++;
            mant10 &= 0x03ff;
        }
        int32_t exp32 = (int32_t)exp5 - 15 + 127;
        uint32_t bits = sign | ((uint32_t)exp32 << 23) | ((uint32_t)mant10 << 13);
        float f; std::memcpy(&f, &bits, sizeof(f)); return f;
    }

    static uint16_t float_to_bf16(float f) {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        uint32_t rounding = 0x7fff;
        if ((bits & 0xffff) != 0 || ((bits >> 16) & 1))
            bits += rounding;
        return (uint16_t)(bits >> 16);
    }

    static float bf16_to_float(uint16_t bf16) {
        uint32_t bits = (uint32_t)bf16 << 16;
        float f; std::memcpy(&f, &bits, sizeof(f)); return f;
    }

    static uint8_t float_to_fp8_e4m3(float f) {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        uint32_t sign = (bits >> 24) & 0x80;
        int32_t exp8 = (int32_t)((bits >> 23) & 0xff) - 127;
        uint32_t mant23 = bits & 0x7fffff;

        if (exp8 < -6) return (uint8_t)sign;
        if (exp8 > 8) return (uint8_t)(sign | 0x7e);

        int32_t e4 = exp8 + 7;
        if (e4 <= 0) {
            uint32_t m = (mant23 | 0x800000) >> (11 - e4);
            return (uint8_t)(sign | (m >> 20));
        }
        uint32_t round_bit = (mant23 >> 19) & 1;
        uint32_t sticky = mant23 & 0x7ffff;
        uint32_t m3 = mant23 >> 20;
        if (round_bit && ((m3 & 1) || sticky != 0)) {
            m3++;
            if (m3 == 0x08) {
                m3 = 0;
                e4++;
                if (e4 > 14) return (uint8_t)(sign | 0x7e);
            }
        }
        return (uint8_t)(sign | ((uint32_t)e4 << 3) | m3);
    }

    static float fp8_e4m3_to_float(uint8_t fp8) {
        uint32_t sign = (uint32_t)(fp8 & 0x80) << 24;
        uint32_t exp4 = (fp8 >> 3) & 0x0f;
        uint32_t mant3 = fp8 & 0x07;

        if (exp4 == 0x0f) {
            uint32_t bits = sign | 0x7fc00000;
            float f; std::memcpy(&f, &bits, sizeof(f)); return f;
        }
        if (exp4 == 0) {
            if (mant3 == 0) { float f; std::memcpy(&f, &sign, sizeof(f)); return f; }
            while ((mant3 & 0x08) == 0) { mant3 <<= 1; exp4--; }
            exp4++;
            mant3 &= 0x07;
        }
        int32_t exp32 = (int32_t)exp4 - 7 + 127;
        uint32_t bits = sign | ((uint32_t)exp32 << 23) | ((uint32_t)mant3 << 20);
        float f; std::memcpy(&f, &bits, sizeof(f)); return f;
    }
};

} // namespace lora_kernel
