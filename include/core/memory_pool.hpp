#pragma once
#include <vector>
#include <memory>
#include <iostream>
#include <sys/mman.h>
#include <mutex>
#include <cstring>
#include "include/core/memory_leak_detector.hpp"

namespace lora_kernel {

class MemoryPool {
private:
    uint8_t*  pool_;
    size_t    capacity_;
    size_t    offset_;
    std::mutex mtx_;
    MemoryLeakDetector* tracker_; 

public:
    explicit MemoryPool(size_t sz, MemoryLeakDetector* tracker = nullptr)
        : capacity_(sz), offset_(0), tracker_(tracker) {
        pool_ = static_cast<uint8_t*>(
            mmap(nullptr, sz, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        if (pool_ == MAP_FAILED)
            pool_ = static_cast<uint8_t*>(aligned_alloc(64, sz));
        if (!pool_) throw std::bad_alloc();
        std::cout << "[POOL] Allocated " << sz / (1024*1024) << " MB\n";
    }

    void* allocate(size_t sz) {
        std::lock_guard<std::mutex> lock(mtx_);
        size_t aligned_sz = (sz + 63) & ~size_t(63);
        if (offset_ + aligned_sz > capacity_) {
            std::cerr << "[POOL] Out of pool memory\n";
            return nullptr;
        }
        void* ptr = pool_ + offset_;
        offset_  += aligned_sz;
        if (tracker_) tracker_->record_allocation(ptr, sz);
        return ptr;
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mtx_);
        offset_ = 0;
    }

    ~MemoryPool() {
        if (pool_) munmap(pool_, capacity_);
    }
};

} // namespace lora_kernel
