#pragma once
#include <chrono>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <iostream>

namespace lora_kernel {

class MemoryLeakDetector {
private:
    struct Record { size_t size; std::chrono::steady_clock::time_point ts; };
    std::unordered_map<void*, Record> allocs_;
    std::mutex mtx_;
    std::atomic<size_t> total_{0}, peak_{0};

public:
    void record_allocation(void* ptr, size_t size) {
        std::lock_guard<std::mutex> lock(mtx_);
        allocs_[ptr] = {size, std::chrono::steady_clock::now()};
        size_t t = total_.fetch_add(size, std::memory_order_relaxed) + size;
        size_t old_peak = peak_.load(std::memory_order_relaxed);
        while (t > old_peak &&
               !peak_.compare_exchange_weak(old_peak, t,
                   std::memory_order_relaxed)) {}
    }
    void record_deallocation(void* ptr) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = allocs_.find(ptr);
        if (it != allocs_.end()) {
            total_.fetch_sub(it->second.size, std::memory_order_relaxed);
            allocs_.erase(it);
        }
    }
    void report_leaks() const {
        // const_cast because we take lock on mutable internal state
        auto& self = const_cast<MemoryLeakDetector&>(*this);
        std::lock_guard<std::mutex> lock(self.mtx_);
        if (allocs_.empty())
            std::cout << "[MEMLEAK] No leaks detected.\n";
        else
            std::cerr << "[MEMLEAK] " << allocs_.size()
                      << " allocation(s) outstanding.\n";
        std::cout << "[MEMLEAK] Peak memory: "
                  << peak_.load() / (1024.0 * 1024.0) << " MB\n";
    }
};

} // namespace lora_kernel
