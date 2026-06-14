#pragma once
#include <vector>
#include <map>
#include <cstring>
#include <iostream>
#include <string>

namespace lora_kernel {

enum class CheckpointPolicy {
    Always,
    EveryNLayers,
    MemoryBudget,
    Never
};

class GradientCheckpointing {
private:
    struct ActivationEntry {
        std::vector<float> data;
        size_t byte_size;
        std::vector<int64_t> shape;
    };

    std::map<std::string, ActivationEntry> saved_activations_;
    size_t total_memory_bytes_{0};
    size_t memory_budget_{1024 * 1024 * 1024}; // 1 GB default
    int checkpoint_interval_{2};

public:
    GradientCheckpointing() = default;

    explicit GradientCheckpointing(size_t budget_bytes, int interval = 2)
        : memory_budget_(budget_bytes), checkpoint_interval_(interval) {}

    void save_activations(const std::string& layer_id, const float* tensor_data, size_t num_elements) {
        size_t byte_size = num_elements * sizeof(float);

        if (total_memory_bytes_ + byte_size > memory_budget_) {
            evict_oldest();
        }

        ActivationEntry entry;
        entry.data.assign(tensor_data, tensor_data + num_elements);
        entry.byte_size = byte_size;
        saved_activations_[layer_id] = std::move(entry);
        total_memory_bytes_ += byte_size;
    }

    void save_activations(const std::string& layer_id, const Tensor& tensor) {
        save_activations(layer_id, tensor.data(), static_cast<size_t>(tensor.numel()));
    }

    bool restore_activations(const std::string& layer_id, float* output_buffer, size_t num_elements) {
        auto it = saved_activations_.find(layer_id);
        if (it == saved_activations_.end()) return false;

        size_t copy_bytes = std::min(it->second.byte_size, num_elements * sizeof(float));
        std::memcpy(output_buffer, it->second.data.data(), copy_bytes);
        return true;
    }

    bool restore_activations(const std::string& layer_id, Tensor& output) {
        return restore_activations(layer_id, output.data(), static_cast<size_t>(output.numel()));
    }

    bool should_checkpoint(const std::string& layer_id, int layer_idx, CheckpointPolicy policy) const {
        switch (policy) {
            case CheckpointPolicy::Always:
                return true;
            case CheckpointPolicy::EveryNLayers:
                return (layer_idx % checkpoint_interval_ == 0);
            case CheckpointPolicy::MemoryBudget:
                return total_memory_bytes_ < memory_budget_;
            case CheckpointPolicy::Never:
                return false;
            default:
                return false;
        }
    }

    void free_activations(const std::string& layer_id) {
        auto it = saved_activations_.find(layer_id);
        if (it != saved_activations_.end()) {
            total_memory_bytes_ -= it->second.byte_size;
            saved_activations_.erase(it);
        }
    }

    void clear() {
        saved_activations_.clear();
        total_memory_bytes_ = 0;
    }

    size_t total_memory_bytes() const { return total_memory_bytes_; }
    size_t total_memory_mb() const { return total_memory_bytes_ / (1024 * 1024); }
    int num_saved_activations() const { return static_cast<int>(saved_activations_.size()); }

    void set_memory_budget(size_t budget_bytes) { memory_budget_ = budget_bytes; }
    void set_checkpoint_interval(int interval) { checkpoint_interval_ = interval; }

private:
    void evict_oldest() {
        if (saved_activations_.empty()) return;
        auto oldest = saved_activations_.begin();
        total_memory_bytes_ -= oldest->second.byte_size;
        saved_activations_.erase(oldest);
    }
};

} // namespace lora_kernel
