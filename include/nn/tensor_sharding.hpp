#pragma once
#include <vector>
#include <algorithm>
#include <iostream>
#include "include/core/tensor.hpp"

namespace lora_kernel {

// Production tensor sharding for ZeRO-3: sharding params, grads, and optimizer states

class TensorSharding {
private:
    int rank_, world_size_;

public:
    TensorSharding(int rank, int world_size)
        : rank_(rank), world_size_(world_size) {}

    // Compute offset and size for this rank's shard
    void get_shard(int64_t total_size, int64_t& offset, int64_t& local_size) const {
        local_size = total_size / world_size_;
        offset = rank_ * local_size;
        if (rank_ == world_size_ - 1)
            local_size = total_size - offset; // last rank gets remainder
    }

    // Check if this rank owns index i
    bool owns(int64_t i, int64_t total_size) const {
        int64_t per_rank = total_size / world_size_;
        int64_t start = rank_ * per_rank;
        int64_t end = (rank_ == world_size_ - 1) ? total_size : start + per_rank;
        return i >= start && i < end;
    }

    // Scatter a tensor across ranks
    Tensor scatter(const Tensor& global, int world_size) const {
        int64_t per_rank = global.numel() / world_size;
        int64_t offset = rank_ * per_rank;
        Tensor local({per_rank});
        std::memcpy(local.data(), global.data() + offset, per_rank * sizeof(float));
        return local;
    }

    // Gather shards from all ranks into a full tensor
    Tensor gather(const std::vector<Tensor>& shards) const {
        int64_t total = 0;
        for (auto& s : shards) total += s.numel();
        Tensor full({total});
        int64_t offset = 0;
        for (auto& s : shards) {
            std::memcpy(full.data() + offset, s.data(), s.numel() * sizeof(float));
            offset += s.numel();
        }
        return full;
    }
};

// Gradient Checkpointing with recomputation schedule
class GradientCheckpointingScheduler {
private:
    bool enabled_{false};
    int checkpoint_every_n_layers_{4};
    int total_layers_{12};

public:
    GradientCheckpointingScheduler(int total_layers = 12, int interval = 4)
        : total_layers_(total_layers), checkpoint_every_n_layers_(interval) {}

    // Should this layer checkpoint (save inputs, recompute backward)?
    bool should_checkpoint(int layer_idx) const {
        return enabled_ && (layer_idx % checkpoint_every_n_layers_ == 0);
    }

    // Number of activations saved (vs full save)
    float memory_savings_factor() const {
        if (!enabled_) return 1.0f;
        return 1.0f / checkpoint_every_n_layers_;
    }

    void set_interval(int n) { checkpoint_every_n_layers_ = n; }
    void enable() { enabled_ = true; }
    void disable() { enabled_ = false; }
    bool is_enabled() const { return enabled_; }
};

// Selective activation offloading (CPU for activations, GPU for params)
class ActivationOffloader {
private:
    std::vector<std::vector<float>> cpu_buffer_;
    std::vector<bool> offloaded_;
    std::mutex mtx_;

public:
    ActivationOffloader(int max_layers = 32) {
        cpu_buffer_.resize(max_layers);
        offloaded_.resize(max_layers, false);
    }

    // Offload activations to CPU
    void offload(int layer, const float* data, int64_t n) {
        std::lock_guard<std::mutex> lock(mtx_);
        cpu_buffer_[layer].assign(data, data + n);
        offloaded_[layer] = true;
    }

    // Retrieve offloaded activations
    float* retrieve(int layer) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!offloaded_[layer]) return nullptr;
        return cpu_buffer_[layer].data();
    }

    bool is_offloaded(int layer) const { return offloaded_[layer]; }
    void clear() {
        for (auto& buf : cpu_buffer_) buf.clear();
        std::fill(offloaded_.begin(), offloaded_.end(), false);
    }

    size_t total_offloaded_bytes() const {
        size_t bytes = 0;
        for (auto& buf : cpu_buffer_) bytes += buf.size() * sizeof(float);
        return bytes;
    }
};

} // namespace lora_kernel
