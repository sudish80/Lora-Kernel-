#pragma once
#include <string>
#include <vector>
#include <map>
#include <cstring>
#include <iostream>
#include <list>
#include <mutex>
#include <atomic>

#ifdef __linux__
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace lora_kernel {

// Production memory-mapped model loader with guard pages and LRU caching

class MmapModelLoader {
private:
    int fd_{-1};
    void* mapped_base_{nullptr};
    size_t file_size_{0};

    // Guard page support
    void* alloc_with_guard(size_t size) {
#ifdef __linux__
        // Allocate with guard page at end for overflow detection
        size_t page = sysconf(_SC_PAGESIZE);
        size_t aligned = (size + page - 1) & ~(page - 1);
        void* ptr = mmap(nullptr, aligned + page, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ptr == MAP_FAILED) return nullptr;
        // Guard page: last page is inaccessible
        mprotect((char*)ptr + aligned, page, PROT_NONE);
        return ptr;
#else
        return std::malloc(size);
#endif
    }

    void free_with_guard(void* ptr, size_t size) {
#ifdef __linux__
        size_t page = sysconf(_SC_PAGESIZE);
        size_t aligned = (size + page - 1) & ~(page - 1);
        munmap(ptr, aligned + page);
#else
        std::free(ptr);
#endif
    }

public:
    // Memory-map a weight file with lazy loading
    bool mmap_open(const std::string& path) {
#ifdef __linux__
        fd_ = open(path.c_str(), O_RDONLY);
        if (fd_ < 0) return false;

        struct stat st;
        fstat(fd_, &st);
        file_size_ = st.st_size;

        mapped_base_ = mmap(nullptr, file_size_, PROT_READ,
                           MAP_PRIVATE | MAP_POPULATE, fd_, 0);
        if (mapped_base_ == MAP_FAILED) {
            close(fd_);
            return false;
        }

        std::cout << "[MMAP] Mapped " << file_size_ / 1024 / 1024
                  << " MB from " << path << "\n";
        return true;
#else
        (void)path;
        return false;
#endif
    }

    // Lazy load: only fault in pages as needed (MAP_POPULATE helps)
    void prefetch_range(size_t offset, size_t size) {
#ifdef __linux__
        madvise((char*)mapped_base_ + offset, size, MADV_WILLNEED);
#endif
    }

    // Read from mapped file
    bool read(void* dst, size_t offset, size_t size) {
        if (!mapped_base_) return false;
        if (offset + size > file_size_) return false;
        std::memcpy(dst, (char*)mapped_base_ + offset, size);
        return true;
    }

    // Allocate a weight tensor with guard page
    void* allocate_weights(size_t size) {
        return alloc_with_guard(size);
    }

    // Load a tensor from mmap file into guarded buffer
    bool load_tensor(void* dst, size_t file_offset, size_t size) {
        return read(dst, file_offset, size);
    }

    void close() {
#ifdef __linux__
        if (mapped_base_) munmap(mapped_base_, file_size_);
        if (fd_ >= 0) close(fd_);
        mapped_base_ = nullptr;
        fd_ = -1;
        file_size_ = 0;
#endif
    }

    ~MmapModelLoader() { close(); }
};

// LRU cache for model weight streaming
class WeightCache {
private:
    struct CacheEntry {
        std::string name;
        std::vector<float> data;
        size_t size;
    };

    size_t max_bytes_;
    size_t current_bytes_{0};
    std::map<std::string, std::list<std::pair<std::string, CacheEntry>>::iterator> cache_map_;
    std::list<std::pair<std::string, CacheEntry>> lru_list_;
    std::mutex mtx_;

public:
    WeightCache(size_t max_mb = 1024) : max_bytes_(max_mb * 1024 * 1024) {}

    bool contains(const std::string& name) {
        std::lock_guard<std::mutex> lock(mtx_);
        return cache_map_.find(name) != cache_map_.end();
    }

    // Get weights, moving to front (MRU position)
    float* get(const std::string& name) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = cache_map_.find(name);
        if (it == cache_map_.end()) return nullptr;

        // Move to front (most recently used)
        lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
        return it->second->second.data.data();
    }

    // Insert weights, evicting LRU entries if needed
    void put(const std::string& name, const float* data, size_t n) {
        std::lock_guard<std::mutex> lock(mtx_);
        size_t bytes = n * sizeof(float);

        // Evict until enough space
        while (current_bytes_ + bytes > max_bytes_ && !lru_list_.empty()) {
            auto& lru = lru_list_.back();
            current_bytes_ -= lru.second.size * sizeof(float);
            cache_map_.erase(lru.first);
            lru_list_.pop_back();
        }

        CacheEntry entry{name, std::vector<float>(data, data + n), n};
        lru_list_.push_front({name, std::move(entry)});
        cache_map_[name] = lru_list_.begin();
        current_bytes_ += bytes;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mtx_);
        cache_map_.clear();
        lru_list_.clear();
        current_bytes_ = 0;
    }

    size_t current_mb() const { return current_bytes_ / 1024 / 1024; }
    size_t max_mb() const { return max_bytes_ / 1024 / 1024; }

    void print_stats() {
        std::lock_guard<std::mutex> lock(mtx_);
        std::cout << "[CACHE] " << cache_map_.size() << " entries, "
                  << current_mb() << "/" << max_mb() << " MB\n";
    }
};

} // namespace lora_kernel
