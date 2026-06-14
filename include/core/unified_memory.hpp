#pragma once
#include <vector>
#include <map>
#include <mutex>
#include <cstring>
#include <iostream>
#include <atomic>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace lora_kernel {

enum class MemoryLocation { CPU, GPU, UNIFIED };

// Production unified memory manager with automatic CPU/GPU migration

class UnifiedMemoryManager {
private:
    struct UMBlock {
        void* ptr;
        size_t size;
        MemoryLocation location;
        MemoryLocation preferred_location;
        bool migrated_{false};
    };

    std::map<void*, UMBlock> blocks_;
    std::mutex mtx_;
    std::atomic<size_t> cpu_usage_{0};
    std::atomic<size_t> gpu_usage_{0};
    size_t migration_threshold_{512 * 1024 * 1024}; // 512 MB trigger

public:
    UnifiedMemoryManager() {
#ifdef __CUDACC__
        std::cout << "[UM] CUDA unified memory available\n";
#endif
    }

    // Allocate in preferred location
    void* allocate(size_t size, MemoryLocation preferred = MemoryLocation::GPU) {
        std::lock_guard<std::mutex> lock(mtx_);
        void* ptr = nullptr;

#ifdef __CUDACC__
        if (preferred == MemoryLocation::GPU) {
            cudaMalloc(&ptr, size);
            if (ptr) {
                gpu_usage_ += size;
                blocks_[ptr] = {ptr, size, MemoryLocation::GPU, MemoryLocation::GPU};
                return ptr;
            }
            // Fall back to unified
        }

        if (preferred == MemoryLocation::UNIFIED || ptr == nullptr) {
            cudaMallocManaged(&ptr, size);
            if (ptr) {
                // Prefer GPU residency
                cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
                blocks_[ptr] = {ptr, size, MemoryLocation::UNIFIED, MemoryLocation::GPU};
                return ptr;
            }
        }
#endif

        // CPU fallback
        ptr = std::malloc(size);
        if (ptr) {
            std::memset(ptr, 0, size);
            cpu_usage_ += size;
            blocks_[ptr] = {ptr, size, MemoryLocation::CPU, MemoryLocation::CPU};
        }
        return ptr;
    }

    // Migrate a block to a new location
    bool migrate(void* ptr, MemoryLocation target) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = blocks_.find(ptr);
        if (it == blocks_.end()) return false;

        auto& block = it->second;
        if (block.location == target) return true;

#ifdef __CUDACC__
        if (target == MemoryLocation::GPU && block.location != MemoryLocation::GPU) {
            void* gpu_ptr;
            cudaMalloc(&gpu_ptr, block.size);
            cudaMemcpy(gpu_ptr, block.ptr, block.size, cudaMemcpyHostToDevice);
            block.ptr = gpu_ptr;
            block.migrated_ = true;
            // Don't free CPU pointer (managed by this manager)
        } else if (target == MemoryLocation::CPU && block.location != MemoryLocation::CPU) {
            void* cpu_ptr = std::malloc(block.size);
            cudaMemcpy(cpu_ptr, block.ptr, block.size, cudaMemcpyDeviceToHost);
            cudaFree(block.ptr);
            block.ptr = cpu_ptr;
            block.migrated_ = true;
        }
#else
        (void)target;
#endif

        block.location = target;
        return true;
    }

    // Prefetch to GPU for compute
    void prefetch_to_gpu(void* ptr, size_t size) {
#ifdef __CUDACC__
        cudaMemPrefetchAsync(ptr, size, 0); // prefetch to GPU 0
#endif
    }

    // Prefetch to CPU for serialization
    void prefetch_to_cpu(void* ptr, size_t size) {
#ifdef __CUDACC__
        cudaMemPrefetchAsync(ptr, size, cudaCpuDeviceId);
#endif
    }

    void deallocate(void* ptr) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = blocks_.find(ptr);
        if (it == blocks_.end()) return;

        auto& block = it->second;
#ifdef __CUDACC__
        if (block.location == MemoryLocation::GPU || block.location == MemoryLocation::UNIFIED) {
            cudaFree(block.ptr);
        } else {
            std::free(block.ptr);
        }
#else
        std::free(block.ptr);
#endif
        cpu_usage_ -= (block.location == MemoryLocation::CPU ? block.size : 0);
        gpu_usage_ -= (block.location == MemoryLocation::GPU ? block.size : 0);
        blocks_.erase(it);
    }

    void print_stats() {
        std::cout << "=== Unified Memory Stats ===\n";
        std::cout << "CPU: " << cpu_usage_.load() / 1024 / 1024 << " MB\n";
        std::cout << "GPU: " << gpu_usage_.load() / 1024 / 1024 << " MB\n";
        std::cout << "Blocks: " << blocks_.size() << "\n";
    }

    size_t total_usage() const { return cpu_usage_ + gpu_usage_; }

    ~UnifiedMemoryManager() {
        std::lock_guard<std::mutex> lock(mtx_);
        for (auto& [ptr, block] : blocks_) {
#ifdef __CUDACC__
            cudaFree(block.ptr);
#else
            std::free(block.ptr);
#endif
        }
        blocks_.clear();
    }
};

} // namespace lora_kernel
