#pragma once
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <openssl/err.h>
#include <vector>
#include <array>
#include <cstring>
#include <iostream>
#include <fstream>
#include <iterator>

namespace lora_kernel {

class AES256CTR {
public:
    static bool encrypt(const uint8_t* key,
                        const uint8_t* plaintext, size_t len,
                        std::vector<uint8_t>& out) {
        uint8_t iv[16];
        if (RAND_bytes(iv, sizeof(iv)) != 1) return false;

        out.resize(sizeof(iv) + len);
        std::memcpy(out.data(), iv, sizeof(iv));

        EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
        if (!ctx) return false;

        int outl = 0;
        bool ok =
            EVP_EncryptInit_ex(ctx, EVP_aes_256_ctr(), nullptr, key, iv) == 1 &&
            EVP_EncryptUpdate(ctx, out.data() + sizeof(iv), &outl,
                              plaintext, (int)len) == 1;

        EVP_CIPHER_CTX_free(ctx);
        return ok;
    }

    static bool decrypt(const uint8_t* key,
                        const uint8_t* ciphertext, size_t len,
                        std::vector<uint8_t>& out) {
        if (len < 16) return false;
        const uint8_t* iv   = ciphertext;
        const uint8_t* data = ciphertext + 16;
        size_t data_len     = len - 16;

        out.resize(data_len);
        EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
        if (!ctx) return false;

        int outl = 0;
        bool ok =
            EVP_DecryptInit_ex(ctx, EVP_aes_256_ctr(), nullptr, key, iv) == 1 &&
            EVP_DecryptUpdate(ctx, out.data(), &outl, data, (int)data_len) == 1;

        EVP_CIPHER_CTX_free(ctx);
        return ok;
    }

    static std::array<uint8_t, 32> generate_key() {
        std::array<uint8_t, 32> key;
        RAND_bytes(key.data(), 32);
        return key;
    }
};

class WeightEncryptionEngine {
private:
    std::array<uint8_t, 32> key_;

public:
    WeightEncryptionEngine() { key_ = AES256CTR::generate_key(); }

    std::vector<uint8_t> encrypt_weights(const float* w, size_t n) const {
        std::vector<uint8_t> out;
        AES256CTR::encrypt(key_.data(),
                           reinterpret_cast<const uint8_t*>(w),
                           n * sizeof(float), out);
        return out;
    }

    bool decrypt_weights(const std::vector<uint8_t>& blob,
                         float* w, size_t n) const {
        std::vector<uint8_t> plain;
        if (!AES256CTR::decrypt(key_.data(), blob.data(), blob.size(), plain))
            return false;
        if (plain.size() != n * sizeof(float)) return false;
        std::memcpy(w, plain.data(), plain.size());
        return true;
    }
};

class EncryptedCheckpointStorage {
private:
    std::array<uint8_t, 32> key_;

public:
    EncryptedCheckpointStorage() { key_ = AES256CTR::generate_key(); }

    bool save(const std::string& file, const void* data, size_t size) {
        std::vector<uint8_t> enc;
        if (!AES256CTR::encrypt(key_.data(),
                                static_cast<const uint8_t*>(data),
                                size, enc)) {
            std::cerr << "[CKPT] Encryption failed for " << file << "\n";
            return false;
        }
        std::ofstream f(file, std::ios::binary);
        if (!f) { std::cerr << "[CKPT] Cannot open " << file << "\n"; return false; }
        uint64_t plain_size = static_cast<uint64_t>(size);
        f.write(reinterpret_cast<const char*>(&plain_size), sizeof(plain_size));
        f.write(reinterpret_cast<const char*>(enc.data()), enc.size());
        std::cout << "[CKPT] Encrypted checkpoint saved: " << file
                  << " (" << enc.size() << " bytes)\n";
        return true;
    }

    bool load(const std::string& file, void* data, size_t max_size) {
        std::ifstream f(file, std::ios::binary);
        if (!f) { std::cerr << "[CKPT] Cannot open " << file << "\n"; return false; }
        uint64_t plain_size;
        f.read(reinterpret_cast<char*>(&plain_size), sizeof(plain_size));
        if (plain_size > max_size) {
            std::cerr << "[CKPT] Checkpoint size mismatch\n"; return false;
        }
        std::vector<uint8_t> enc(std::istreambuf_iterator<char>(f), {});
        std::vector<uint8_t> plain;
        if (!AES256CTR::decrypt(key_.data(), enc.data(), enc.size(), plain))
            return false;
        std::memcpy(data, plain.data(), plain_size);
        return true;
    }
};

class ChecksumValidator {
private:
    uint32_t table_[256];

    void init() {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int j = 0; j < 8; ++j)
                c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            table_[i] = c;
        }
    }

public:
    ChecksumValidator() { init(); }

    uint32_t compute(const void* data, size_t size) const {
        const uint8_t* b = static_cast<const uint8_t*>(data);
        uint32_t crc = 0xFFFFFFFFu;
        for (size_t i = 0; i < size; ++i)
            crc = table_[(crc ^ b[i]) & 0xFF] ^ (crc >> 8);
        return crc ^ 0xFFFFFFFFu;
    }

    bool verify(const void* data, size_t size, uint32_t expected) const {
        return compute(data, size) == expected;
    }
};

} // namespace lora_kernel
