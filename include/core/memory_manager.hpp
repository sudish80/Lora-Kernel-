#pragma once
#include <vector>
#include <map>
#include <unordered_map>
#include <mutex>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <atomic>
#include <string>
#include <sstream>
#include <fstream>
#include <thread>
#include <list>

#ifdef __linux__
#include <sys/mman.h>
#include <numa.h>
#include <numaif.h>
#include <unistd.h>
#include <sched.h>
#endif

#ifdef _WIN32
#include <windows.h>
#include <sysinfoapi.h>
#endif

namespace lora_kernel {

// ============================================================
// WeightCache - O(1) LRU eviction using unordered_map + doubly-linked list
// ============================================================
template<typename Key, typename Value>
class WeightCache {
private:
    struct CacheNode {
        Key key;
        Value value;
        CacheNode* prev;
        CacheNode* next;
    };

    std::unordered_map<Key, CacheNode*> map_;
    CacheNode* head_;
    CacheNode* tail_;
    size_t capacity_;
    std::mutex mtx_;

    void detach(CacheNode* node) {
        if (node->prev) node->prev->next = node->next;
        if (node->next) node->next->prev = node->prev;
        if (node == head_) head_ = node->next;
        if (node == tail_) tail_ = node->prev;
    }

    void push_front(CacheNode* node) {
        node->next = head_;
        node->prev = nullptr;
        if (head_) head_->prev = node;
        head_ = node;
        if (!tail_) tail_ = node;
    }

    CacheNode* pop_back() {
        if (!tail_) return nullptr;
        CacheNode* node = tail_;
        detach(node);
        return node;
    }

public:
    WeightCache(size_t capacity = 1024) : head_(nullptr), tail_(nullptr), capacity_(capacity) {}

    ~WeightCache() {
        for (auto& [k, v] : map_) delete v;
    }

    void insert(const Key& key, const Value& value) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = map_.find(key);
        if (it != map_.end()) {
            detach(it->second);
            it->second->value = value;
            push_front(it->second);
            return;
        }
        if (map_.size() >= capacity_) {
            CacheNode* victim = pop_back();
            if (victim) {
                map_.erase(victim->key);
                delete victim;
            }
        }
        CacheNode* node = new CacheNode{key, value, nullptr, nullptr};
        map_[key] = node;
        push_front(node);
    }

    bool get(const Key& key, Value& out) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = map_.find(key);
        if (it == map_.end()) return false;
        detach(it->second);
        push_front(it->second);
        out = it->second->value;
        return true;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mtx_);
        for (auto& [k, v] : map_) delete v;
        map_.clear();
        head_ = tail_ = nullptr;
    }

    size_t size() const { return map_.size(); }
    size_t capacity() const { return capacity_; }
    void set_capacity(size_t cap) { capacity_ = cap; }
};

// ============================================================
// NUMATopology - auto-detection and binding
// ============================================================
struct NUMATopology {
    int num_nodes;
    std::vector<std::vector<int>> cpu_ids_per_node;

    NUMATopology() : num_nodes(0) {
        detect();
    }

    void detect() {
#ifdef __linux__
        std::ifstream online("/sys/devices/system/node/online");
        if (online.is_open()) {
            std::string content;
            std::getline(online, content);
            online.close();
            if (!content.empty()) {
                size_t dash = content.find('-');
                if (dash != std::string::npos) {
                    int lo = std::stoi(content.substr(0, dash));
                    int hi = std::stoi(content.substr(dash + 1));
                    num_nodes = hi - lo + 1;
                } else {
                    num_nodes = 1;
                }
            }
        }
        if (num_nodes <= 0) num_nodes = 1;

        cpu_ids_per_node.resize(num_nodes);
        for (int n = 0; n < num_nodes; ++n) {
            std::string path = "/sys/devices/system/node/node" + std::to_string(n) + "/cpulist";
            std::ifstream cpulist(path);
            if (cpulist.is_open()) {
                std::string line;
                std::getline(cpulist, line);
                cpulist.close();
                std::stringstream ss(line);
                std::string token;
                while (std::getline(ss, token, ',')) {
                    size_t dash = token.find('-');
                    if (dash != std::string::npos) {
                        int lo = std::stoi(token.substr(0, dash));
                        int hi = std::stoi(token.substr(dash + 1));
                        for (int c = lo; c <= hi; ++c)
                            cpu_ids_per_node[n].push_back(c);
                    } else {
                        cpu_ids_per_node[n].push_back(std::stoi(token));
                    }
                }
            }
        }
#elif defined(_WIN32)
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX buffer = nullptr;
        DWORD len = 0;
        GetLogicalProcessorInformationEx(RelationNumaNode, buffer, &len);
        if (GetLastError() == ERROR_INSUFFICIENT_BUFFER && len > 0) {
            buffer = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)malloc(len);
            if (GetLogicalProcessorInformationEx(RelationNumaNode, buffer, &len)) {
                DWORD offset = 0;
                while (offset < len) {
                    SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* info =
                        (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)((char*)buffer + offset);
                    if (info->Relationship == RelationNumaNode) {
                        num_nodes++;
                    }
                    offset += info->Size;
                }
            }
            free(buffer);
        }
        if (num_nodes <= 0) num_nodes = 1;
        cpu_ids_per_node.resize(num_nodes);
        // On Windows, approximate by distributing CPUs evenly
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        int total_cpus = sysInfo.dwNumberOfProcessors;
        int cpus_per_node = total_cpus / num_nodes;
        for (int n = 0; n < num_nodes; ++n) {
            for (int c = n * cpus_per_node; c < (n + 1) * cpus_per_node && c < total_cpus; ++c)
                cpu_ids_per_node[n].push_back(c);
        }
#else
        num_nodes = 1;
        cpu_ids_per_node.resize(1);
#endif
    }

    void bind_to_node(int node) {
        if (node < 0 || node >= num_nodes) return;
#ifdef __linux__
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        for (int cpu : cpu_ids_per_node[node])
            CPU_SET(cpu, &cpuset);
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
#elif defined(_WIN32)
        DWORD_PTR mask = 0;
        for (int cpu : cpu_ids_per_node[node])
            mask |= (DWORD_PTR)1 << cpu;
        SetThreadAffinityMask(GetCurrentThread(), mask);
#endif
    }

    void* alloc_on_node(size_t size, int node) {
        size_t aligned = (size + 4095) & ~4095;
#ifdef __linux__
        void* ptr = mmap(nullptr, aligned, PROT_READ | PROT_WRITE,
                         MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
        if (ptr == MAP_FAILED) return nullptr;
        if (node >= 0 && node < num_nodes) {
            struct bitmask* mask = numa_allocate_nodemask();
            if (mask) {
                numa_bitmask_setbit(mask, node);
                mbind(ptr, aligned, MPOL_BIND, mask->maskp, mask->size, MPOL_MF_MOVE);
                numa_free_nodemask(mask);
            }
        }
        return ptr;
#elif defined(_WIN32)
        return VirtualAllocExNuma(GetCurrentProcess(), NULL, aligned,
                                  MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE, (DWORD)node);
#else
        (void)node;
        return _aligned_malloc(aligned, 4096);
#endif
    }
};

// Production memory manager with fragmentation tracking, NUMA, huge pages

struct MemoryBlock {
    void* ptr;
    size_t size;
    bool free;
    int numa_node;
    bool huge_page;
    std::string label;
};

class MemoryManager {
private:
    std::map<void*, MemoryBlock> blocks_;
    std::mutex mtx_;
    std::atomic<size_t> total_allocated_{0};
    std::atomic<size_t> peak_allocated_{0};
    size_t page_size_;
    bool huge_page_available_{false};

    // Fragmentation tracking
    size_t largest_free_block_{0};
    size_t total_free_memory_{0};

public:
    MemoryManager() {
#ifdef __linux__
        page_size_ = sysconf(_SC_PAGESIZE);
        huge_page_available_ = (access("/sys/kernel/mm/hugepages", F_OK) == 0);
#else
        page_size_ = 4096;
#endif
    }

    void* allocate(size_t size, const std::string& label = "",
                   int numa_node = -1, bool use_huge = false) {
        std::lock_guard<std::mutex> lock(mtx_);

        size_t aligned = (size + page_size_ - 1) & ~(page_size_ - 1);
        void* ptr = nullptr;

#ifdef __linux__
        int flags = MAP_PRIVATE | MAP_ANONYMOUS;
        if (use_huge && huge_page_available_)
            flags |= MAP_HUGETLB;

        ptr = mmap(nullptr, aligned, PROT_READ | PROT_WRITE, flags, -1, 0);
        if (ptr == MAP_FAILED) {
            ptr = mmap(nullptr, aligned, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (ptr == MAP_FAILED) {
                ptr = aligned_alloc(page_size_, aligned);
            }
        }

        if (numa_node >= 0 && ptr) {
            struct bitmask* mask = numa_allocate_nodemask();
            numa_bitmask_setbit(mask, numa_node);
            mbind(ptr, aligned, MPOL_BIND, mask->maskp, mask->size, MPOL_MF_MOVE);
            numa_free_nodemask(mask);
        }
#else
        ptr = _aligned_malloc(aligned, page_size_);
#endif

        if (!ptr) throw std::bad_alloc();

        blocks_[ptr] = {ptr, aligned, false, numa_node, use_huge, label};
        total_allocated_ += aligned;
        size_t ta = total_allocated_.load();
        size_t pa = peak_allocated_.load();
        if (ta > pa)
            peak_allocated_.store(ta);

        std::memset(ptr, 0, aligned);
        return ptr;
    }

    void deallocate(void* ptr) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = blocks_.find(ptr);
        if (it == blocks_.end()) {
            std::cerr << "[MEM] Double free or invalid pointer\n";
            return;
        }

        it->second.free = true;
        total_allocated_ -= it->second.size;

#ifdef __linux__
        munmap(ptr, it->second.size);
#else
        _aligned_free(ptr);
#endif
        blocks_.erase(it);
    }

    void defragment() {
        std::lock_guard<std::mutex> lock(mtx_);
        std::vector<MemoryBlock> sorted;
        for (auto& [ptr, block] : blocks_)
            sorted.push_back(block);

        std::sort(sorted.begin(), sorted.end(),
                  [](auto& a, auto& b) { return a.ptr < b.ptr; });

        size_t largest_free = 0;
        size_t total_free = 0;

        for (size_t i = 0; i + 1 < sorted.size(); ++i) {
            if (sorted[i].free && sorted[i + 1].free) {
                size_t merged_size = sorted[i].size + sorted[i + 1].size;
                blocks_[sorted[i].ptr].size = merged_size;
                blocks_.erase(sorted[i + 1].ptr);
                if (merged_size > largest_free) largest_free = merged_size;
                total_free += merged_size;
            }
        }

        largest_free_block_ = largest_free;
        total_free_memory_ = total_free;

        std::cout << "[MEM] Defrag: largest_free=" << largest_free / 1024 / 1024
                  << " MB\n";
    }

    void print_stats() {
        std::lock_guard<std::mutex> lock(mtx_);
        std::cout << "=== Memory Manager Stats ===\n";
        std::cout << "Allocated: " << total_allocated_.load() / 1024 / 1024 << " MB\n";
        std::cout << "Peak: " << peak_allocated_.load() / 1024 / 1024 << " MB\n";
        std::cout << "Blocks: " << blocks_.size() << "\n";
        std::cout << "Page size: " << page_size_ << " bytes\n";
        std::cout << "Huge pages: " << (huge_page_available_ ? "yes" : "no") << "\n";
    }

    size_t total_allocated() const { return total_allocated_; }
    size_t peak_allocated() const { return peak_allocated_; }
};

// ============================================================
// Guard page support (Windows: VirtualAlloc + PAGE_GUARD)
// ============================================================
#ifdef _WIN32
inline void* alloc_guard_page(size_t size) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    size_t page_size = si.dwPageSize;
    size_t total = size + page_size;
    void* ptr = VirtualAlloc(NULL, total, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!ptr) return nullptr;
    void* guard = (char*)ptr + ((total / page_size) - 1) * page_size;
    DWORD old;
    VirtualProtect(guard, page_size, PAGE_GUARD | PAGE_NOACCESS, &old);
    return ptr;
}

inline void free_guard_page(void* ptr, size_t size) {
    (void)size;
    VirtualFree(ptr, 0, MEM_RELEASE);
}
#endif

} // namespace lora_kernel
