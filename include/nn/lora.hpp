#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <random>
#include <atomic>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <algorithm>

namespace lora_kernel {

// Forward declarations or simple includes
template<typename T>
class ThreadSafeParameterShield {
private:
    T*   data_;
    size_t size_;
    mutable std::shared_mutex mutex_;
    std::atomic<uint64_t> version_{0};

public:
    explicit ThreadSafeParameterShield(size_t sz) : size_(sz) {
        data_ = static_cast<T*>(aligned_alloc(64, sz * sizeof(T)));
        if (!data_) throw std::bad_alloc();
        std::memset(data_, 0, sz * sizeof(T));
    }
    ~ThreadSafeParameterShield() { std::free(data_); }

    ThreadSafeParameterShield(const ThreadSafeParameterShield&) = delete;
    ThreadSafeParameterShield& operator=(const ThreadSafeParameterShield&) = delete;

    class ReadGuard {
        const ThreadSafeParameterShield* p_;
        std::shared_lock<std::shared_mutex> lock_;
    public:
        explicit ReadGuard(const ThreadSafeParameterShield* p)
            : p_(p), lock_(p->mutex_) {}
        const T* get() const { return p_->data_; }
        size_t   size() const { return p_->size_; }
    };

    class WriteGuard {
        ThreadSafeParameterShield* p_;
        std::unique_lock<std::shared_mutex> lock_;
    public:
        explicit WriteGuard(ThreadSafeParameterShield* p)
            : p_(p), lock_(p->mutex_) {}
        T*     get()    { return p_->data_; }
        size_t size()   const { return p_->size_; }
        void   commit() { p_->version_.fetch_add(1, std::memory_order_release); }
    };

    ReadGuard  read()        const { return ReadGuard(this); }
    WriteGuard write()             { return WriteGuard(this); }
    uint64_t   version()     const {
        return version_.load(std::memory_order_acquire);
    }
};

class LoRALayer {
public:
    int in_features, out_features, rank;
    std::vector<float> lora_a, lora_b;
    std::vector<float> grad_a, grad_b;

    LoRALayer(int in_f, int out_f, int r)
        : in_features(in_f), out_features(out_f), rank(r),
          lora_a(r * in_f, 0.0f), lora_b(out_f * r, 0.0f),
          grad_a(r * in_f, 0.0f), grad_b(out_f * r, 0.0f) {
        float scale = std::sqrt(2.0f / in_f);
        std::mt19937 gen(std::random_device{}());
        std::normal_distribution<float> dist(0.0f, scale);
        for (auto& v : lora_a) v = dist(gen);
    }

    void forward(const float* input, float* output, int batch_size) const {
        #pragma omp parallel for schedule(static)
        for (int b = 0; b < batch_size; ++b) {
            const float* in_row  = input  + b * in_features;
            float*       out_row = output + b * out_features;

            std::vector<float> mid(rank, 0.0f);
            for (int r = 0; r < rank; ++r) {
                float acc = 0.0f;
                const float* a_row = lora_a.data() + r * in_features;
                for (int i = 0; i < in_features; ++i)
                    acc += a_row[i] * in_row[i];
                mid[r] = acc;
            }

            for (int j = 0; j < out_features; ++j) {
                float acc = 0.0f;
                for (int r = 0; r < rank; ++r)
                    acc += lora_b[j * rank + r] * mid[r];
                out_row[j] += acc;
            }
        }
    }

    void backward(const float* input, const float* grad_output,
                  int batch_size, float dropout_p = 0.0f) {
        for (int b = 0; b < batch_size; ++b) {
            const float* in_row   = input        + b * in_features;
            const float* dout_row = grad_output  + b * out_features;

            std::vector<float> mid(rank, 0.0f);
            for (int r = 0; r < rank; ++r) {
                const float* a_row = lora_a.data() + r * in_features;
                for (int i = 0; i < in_features; ++i)
                    mid[r] += a_row[i] * in_row[i];
            }

            for (int j = 0; j < out_features; ++j) {
                for (int r = 0; r < rank; ++r)
                    grad_b[j * rank + r] += dout_row[j] * mid[r];
            }

            std::vector<float> d_mid(rank, 0.0f);
            for (int r = 0; r < rank; ++r)
                for (int j = 0; j < out_features; ++j)
                    d_mid[r] += dout_row[j] * lora_b[j * rank + r];

            for (int r = 0; r < rank; ++r)
                for (int i = 0; i < in_features; ++i)
                    grad_a[r * in_features + i] += d_mid[r] * in_row[i];
        }
    }

    void update(float lr) {
        for (size_t i = 0; i < lora_a.size(); ++i) {
            lora_a[i] -= lr * grad_a[i];
            grad_a[i]  = 0.0f;
        }
        for (size_t i = 0; i < lora_b.size(); ++i) {
            lora_b[i] -= lr * grad_b[i];
            grad_b[i]  = 0.0f;
        }
    }
};

} // namespace lora_kernel
