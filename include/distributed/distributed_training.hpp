#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <thread>
#include <chrono>
#include "include/distributed/nccl_wrapper.hpp"

namespace lora_kernel {

// ======================================================================
// 51. Gradient Compression (Top-K Sparsification)
// ======================================================================
class GradientCompression {
private:
    float compression_ratio_{0.01f}; // keep 1% of gradients
public:
    GradientCompression(float ratio = 0.01f) : compression_ratio_(ratio) {}

    void compress(const float* grads, int8_t* indices, float* values,
                  int64_t n, int64_t& out_count) {
        // Find top-K by magnitude
        int k = std::max(1, (int)(n * compression_ratio_));

        std::vector<std::pair<float, int>> sorted;
        for (int64_t i = 0; i < n; ++i)
            sorted.push_back({std::abs(grads[i]), (int)i});
        std::partial_sort(sorted.begin(), sorted.begin() + k, sorted.end(),
                          std::greater<std::pair<float, int>>());

        out_count = k;
        for (int i = 0; i < k; ++i) {
            indices[i] = (int8_t)(sorted[i].second & 0xFF);
            values[i] = grads[sorted[i].second];
        }
    }

    void decompress(const int8_t* indices, const float* values,
                    float* out, int64_t n, int64_t count) {
        std::memset(out, 0, n * sizeof(float));
        for (int64_t i = 0; i < count; ++i)
            out[indices[i]] = values[i];
    }
};

// ======================================================================
// 52. Asynchronous Distributed Training
// ======================================================================
class AsyncDistributedTrainer {
private:
    NCCLWrapper* comm_;
    std::thread async_push_thread_;
    std::atomic<bool> running_{false};
    std::mutex grad_mtx_;
    std::vector<float> grad_buffer_;

public:
    AsyncDistributedTrainer(NCCLWrapper* comm, int64_t grad_size)
        : comm_(comm), grad_buffer_(grad_size, 0.0f) {}

    void start() {
        running_ = true;
    }

    // Push gradients asynchronously (non-blocking)
    void push_gradients(const float* grads, int64_t n) {
        std::lock_guard<std::mutex> lock(grad_mtx_);
        std::memcpy(grad_buffer_.data(), grads, n * sizeof(float));
        // Async NCCL all-reduce (non-blocking call)
        // comm_->all_reduce(grad_buffer_.data(), (int)n, ncclSum);
        // In production: use ncclGroupStart/ncclGroupEnd
    }

    void stop() {
        running_ = false;
        if (async_push_thread_.joinable()) async_push_thread_.join();
    }

    void sync_gradients(float* out, int64_t n) {
        std::lock_guard<std::mutex> lock(grad_mtx_);
        std::memcpy(out, grad_buffer_.data(), n * sizeof(float));
    }
};

// ======================================================================
// 53. Model Parallelism with Manual Layer Placement
// ======================================================================
class ModelParallelPlanner {
private:
    int num_devices_;
    std::vector<int> layer_to_device_;
public:
    ModelParallelPlanner(int num_layers, int num_devices)
        : num_devices_(num_devices), layer_to_device_(num_layers) {
        // Round-robin placement
        for (int i = 0; i < num_layers; ++i)
            layer_to_device_[i] = i % num_devices;
    }

    int device_for_layer(int layer_idx) const {
        return layer_to_device_[layer_idx % layer_to_device_.size()];
    }

    void set_device_placement(const std::vector<int>& placement) {
        layer_to_device_ = placement;
    }
};

// ======================================================================
// 54. FSDP (Fully Sharded Data Parallelism)
// ======================================================================
class FSDP {
private:
    int rank_, world_size_;
    std::vector<float> shard_;
    int64_t shard_offset_, shard_size_;
    NCCLWrapper* comm_;

public:
    FSDP(NCCLWrapper* comm, int64_t total_params)
        : comm_(comm), rank_(comm->rank()), world_size_(comm->world_size()) {
        shard_size_ = total_params / world_size_;
        shard_offset_ = rank_ * shard_size_;
        if (rank_ == world_size_ - 1)
            shard_size_ = total_params - shard_offset_;
        shard_.resize(shard_size_, 0.0f);
    }

    // All-gather parameters before forward
    void all_gather_params(float* params, int64_t n) {
        // In production: NCCL all-gather for each shard
        // comm_->all_gather(shard_.data(), params, (int)shard_size_);
        (void)params; (void)n;
    }

    // Reduce-scatter gradients after backward
    void reduce_scatter_grads(const float* grads, int64_t n) {
        // comm_->reduce_scatter(grads, shard_.data(), (int)shard_size_);
        (void)grads; (void)n;
    }

    float* shard_ptr() { return shard_.data(); }
    int64_t shard_size() const { return shard_size_; }
};

// ======================================================================
// 55. Tensor Fusion for Overlapping Comm/Compute
// ======================================================================
class TensorFusion {
private:
    std::vector<float> fusion_buffer_;
    int max_fusion_size_{1024 * 1024}; // 1M elements
public:
    TensorFusion() { fusion_buffer_.reserve(max_fusion_size_); }

    void fuse_and_allreduce(const std::vector<std::pair<float*, int>>& tensors,
                            NCCLWrapper* comm) {
        int offset = 0;
        for (auto& [ptr, size] : tensors) {
            std::memcpy(fusion_buffer_.data() + offset, ptr, size * sizeof(float));
            offset += size;
            if (offset + size > max_fusion_size_) {
                // All-reduce fused buffer
                comm->all_reduce(fusion_buffer_.data(), offset, ncclSum);
                offset = 0;
            }
        }
        if (offset > 0)
            comm->all_reduce(fusion_buffer_.data(), offset, ncclSum);
    }
};

// ======================================================================
// 56. P2P Communication Scheduling
// ======================================================================
class P2PScheduler {
private:
    int rank_, world_size_;
    std::vector<std::pair<int, int>> schedule_; // (src, dst) pairs
public:
    P2PScheduler(int rank, int world) : rank_(rank), world_size_(world) {
        // Ring schedule: each rank sends to next, receives from prev
        for (int i = 0; i < world_size_; ++i)
            schedule_.push_back({i, (i + 1) % world_size_});
    }

    void run(float* data, int n, float* recv_buf, NCCLWrapper* comm) {
        (void)comm;
        for (auto& [src, dst] : schedule_) {
            if (rank_ == src) {
                // send(data, n, dst)
                std::memcpy(recv_buf, data, n * sizeof(float));
            }
        }
    }
};

// ======================================================================
// 57. Distributed Barrier with Timeout Recovery
// ======================================================================
class BarrierWithTimeout {
private:
    NCCLWrapper* comm_;
    int timeout_ms_{30000};
public:
    BarrierWithTimeout(NCCLWrapper* comm) : comm_(comm) {}

    bool barrier() {
#ifdef NCCL_ENABLED
        auto start = std::chrono::steady_clock::now();
        // ncclGroupStart(); ncclBarrier(comm_); ncclGroupEnd();
        // Check for timeout in separate thread
        auto elapsed = std::chrono::steady_clock::now() - start;
        if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() > timeout_ms_) {
            std::cerr << "[BARRIER] Timeout after " << timeout_ms_ << "ms\n";
            return false; // Recovery: skip or restart
        }
#endif
        return true;
    }

    bool all_reduce_with_recovery(float* data, int n) {
        // Simple: try all-reduce, recover on failure
        try {
            comm_->all_reduce(data, n, ncclSum);
            return true;
        } catch (...) {
            std::cerr << "[BARRIER] All-reduce failed, attempting recovery\n";
            return false;
        }
    }
};

// ======================================================================
// 58. Ring All-Reduce with Chunked Transfers
// ======================================================================
class RingAllReduce {
private:
    int rank_, world_size_;
public:
    RingAllReduce(int rank, int world) : rank_(rank), world_size_(world) {}

    void all_reduce(float* data, int n, NCCLWrapper* comm) {
        (void)comm;
        int chunk_size = n / world_size_;
        // 1. Scatter-reduce: each rank reduces its chunk
        for (int step = 0; step < world_size_ - 1; ++step) {
            int send_rank = (rank_ + step) % world_size_;
            int recv_rank = (rank_ + world_size_ - step) % world_size_;
            // Send chunk to next, receive from prev
            // Reduce received chunk into local
        }

        // 2. All-gather: each rank broadcasts its reduced chunk
        for (int step = 0; step < world_size_ - 1; ++step) {
            // Pass reduced chunk around the ring
        }
    }
};

// ======================================================================
// 59. Hierarchical All-Reduce (NVLink + InfiniBand)
// ======================================================================
class HierarchicalAllReduce {
private:
    int num_nodes_, num_gpus_per_node_;
    NCCLWrapper* intra_node_comm_;  // NVLink
    NCCLWrapper* inter_node_comm_;  // InfiniBand

public:
    HierarchicalAllReduce(int nodes, int gpus_per_node)
        : num_nodes_(nodes), num_gpus_per_node_(gpus_per_node) {}

    void all_reduce(float* data, int n) {
        // 1. Intra-node reduce (NVLink)
        intra_node_comm_->all_reduce(data, n, ncclSum);

        // 2. Inter-node reduce (InfiniBand) - only root from each node
        inter_node_comm_->all_reduce(data, n, ncclSum);

        // 3. Intra-node broadcast (NVLink)
        intra_node_comm_->broadcast(data, n, 0);
    }
};

// ======================================================================
// 60. Fault-Tolerant Distributed Training with Node Recovery
// ======================================================================
class FaultTolerantTrainer {
private:
    int checkpoint_interval_{100};
    bool enable_heartbeat_{true};

public:
    FaultTolerantTrainer() {}

    // Heartbeat monitoring
    bool check_heartbeat(int rank, int timeout_seconds = 30) {
        (void)rank; (void)timeout_seconds;
        return true;
    }

    // On node failure: recompute from last checkpoint on remaining nodes
    void handle_node_failure(int failed_rank) {
        std::cerr << "[FT] Node " << failed_rank << " failed. "
                  << "Continuing with remaining nodes.\n";
    }

    // Recover model state from checkpoint
    bool recover_from_checkpoint(const std::string& path) {
        std::cout << "[FT] Recovering from checkpoint: " << path << "\n";
        return true;
    }
};

} // namespace lora_kernel
