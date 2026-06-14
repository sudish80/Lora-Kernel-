#pragma once
#include <vector>
#include <cstring>
#include <map>
#include <set>
#include <list>
#include <iostream>
#include "include/core/tensor.hpp"

namespace lora_kernel {

class KVCache {
private:
    std::vector<float> k_cache_;
    std::vector<float> v_cache_;
    int max_seq_len_, num_heads_, head_dim_, num_layers_;
    int current_len_{0};

    // Backward tracking: for each (layer, position), store whether it was written
    std::set<std::pair<int,int>> written_positions_;
    // Saved key/value slices for backward (keyed by layer, then position)
    std::map<std::pair<int,int>, std::vector<float>> saved_k_;
    std::map<std::pair<int,int>, std::vector<float>> saved_v_;

public:
    KVCache(int max_seq, int layers, int heads, int dim)
        : max_seq_len_(max_seq), num_heads_(heads),
          head_dim_(dim), num_layers_(layers) {
        int64_t size = (int64_t)max_seq * layers * heads * dim;
        k_cache_.resize(size, 0.0f);
        v_cache_.resize(size, 0.0f);
        std::cout << "[KV] Cache allocated: " << size * 2 * sizeof(float) / 1024 / 1024
                  << " MB\n";
    }

    void append(int layer, const float* k, const float* v) {
        int64_t offset = (int64_t)layer * max_seq_len_ * num_heads_ * head_dim_
                       + (int64_t)current_len_ * num_heads_ * head_dim_;
        std::memcpy(k_cache_.data() + offset, k, num_heads_ * head_dim_ * sizeof(float));
        std::memcpy(v_cache_.data() + offset, v, num_heads_ * head_dim_ * sizeof(float));

        // Save for backward
        auto key = std::make_pair(layer, current_len_);
        written_positions_.insert(key);
        int elems = num_heads_ * head_dim_;
        saved_k_[key].assign(k, k + elems);
        saved_v_[key].assign(v, v + elems);
    }

    void retrieve(int layer, float* k_out, float* v_out) {
        int64_t offset = (int64_t)layer * max_seq_len_ * num_heads_ * head_dim_;
        int64_t copy_size = (int64_t)current_len_ * num_heads_ * head_dim_;
        std::memcpy(k_out, k_cache_.data() + offset, copy_size * sizeof(float));
        std::memcpy(v_out, v_cache_.data() + offset, copy_size * sizeof(float));
    }

    void advance() { current_len_++; }
    int current_length() const { return current_len_; }
    void reset() {
        current_len_ = 0;
        written_positions_.clear();
        saved_k_.clear();
        saved_v_.clear();
    }

    void prefill(int layer, const float* k, const float* v, int pos, int count) {
        int64_t offset = (int64_t)layer * max_seq_len_ * num_heads_ * head_dim_
                       + (int64_t)pos * num_heads_ * head_dim_;
        int64_t copy_size = (int64_t)count * num_heads_ * head_dim_;
        std::memcpy(k_cache_.data() + offset, k, copy_size * sizeof(float));
        std::memcpy(v_cache_.data() + offset, v, copy_size * sizeof(float));
        current_len_ = pos + count;

        int elems = num_heads_ * head_dim_;
        for (int p = 0; p < count; ++p) {
            auto key = std::make_pair(layer, pos + p);
            written_positions_.insert(key);
            saved_k_[key].assign(k + p * elems, k + (p + 1) * elems);
            saved_v_[key].assign(v + p * elems, v + (p + 1) * elems);
        }
    }

    void backward(Tensor& grad_key, Tensor& grad_val, int layer, int start_pos, int end_pos) {
        int elems = num_heads_ * head_dim_;
        for (int pos = start_pos; pos < end_pos; ++pos) {
            auto key = std::make_pair(layer, pos);
            auto it_k = saved_k_.find(key);
            auto it_v = saved_v_.find(key);
            if (it_k != saved_k_.end() && it_v != saved_v_.end()) {
                // Gradient flows through: retrieve stored KV and accumulate into grad
                float* gk = grad_key.data() + (pos - start_pos) * elems;
                float* gv = grad_val.data() + (pos - start_pos) * elems;
                for (int i = 0; i < elems; ++i) {
                    gk[i] += it_k->second[i];
                    gv[i] += it_v->second[i];
                }
            }
            // If evicted (not in saved map), gradient is implicitly zero
        }
    }
};

class PagedKVCache {
private:
    struct Page {
        std::vector<float> k_data;
        std::vector<float> v_data;
        bool used{false};
    };

    int block_size_;
    int page_size_;
    std::vector<Page> physical_blocks_;
    std::map<int, int> page_table_;
    std::list<int> free_blocks_;

    // Backward tracking
    std::set<std::pair<int,int>> written_positions_;
    std::map<std::pair<int,int>, std::vector<float>> saved_k_;
    std::map<std::pair<int,int>, std::vector<float>> saved_v_;

public:
    PagedKVCache(int num_blocks, int block_size, int heads, int dim)
        : block_size_(block_size), page_size_(block_size * heads * dim) {
        physical_blocks_.resize(num_blocks);
        for (int i = 0; i < num_blocks; ++i) {
            physical_blocks_[i].k_data.resize(page_size_, 0.0f);
            physical_blocks_[i].v_data.resize(page_size_, 0.0f);
            free_blocks_.push_back(i);
        }
        std::cout << "[PagedKV] " << num_blocks << " blocks, "
                  << block_size << " tokens/block\n";
    }

    int allocate_block(int logical_idx) {
        if (free_blocks_.empty()) {
            std::cerr << "[PagedKV] Out of blocks! Evicting...\n";
            return -1;
        }
        int phys = free_blocks_.front();
        free_blocks_.pop_front();
        physical_blocks_[phys].used = true;
        page_table_[logical_idx] = phys;
        return phys;
    }

    void store(int layer, int token_pos, const float* k, const float* v) {
        int logical_block = token_pos / block_size_;
        int offset_in_block = (token_pos % block_size_) * (page_size_ / block_size_);

        auto it = page_table_.find(logical_block);
        if (it == page_table_.end()) {
            int phys = allocate_block(logical_block);
            if (phys < 0) return;
            it = page_table_.find(logical_block);
        }
        int phys = page_table_[logical_block];
        int elems = page_size_ / block_size_;
        std::memcpy(physical_blocks_[phys].k_data.data() + offset_in_block,
                    k, elems * sizeof(float));
        std::memcpy(physical_blocks_[phys].v_data.data() + offset_in_block,
                    v, elems * sizeof(float));

        auto key = std::make_pair(layer, token_pos);
        written_positions_.insert(key);
        saved_k_[key].assign(k, k + elems);
        saved_v_[key].assign(v, v + elems);
    }

    void load(int layer, int token_pos, float* k_out, float* v_out) {
        int logical_block = token_pos / block_size_;
        int offset_in_block = (token_pos % block_size_) * (page_size_ / block_size_);

        auto it = page_table_.find(logical_block);
        if (it == page_table_.end()) return;
        int phys = it->second;
        int elems = page_size_ / block_size_;
        std::memcpy(k_out, physical_blocks_[phys].k_data.data() + offset_in_block,
                    elems * sizeof(float));
        std::memcpy(v_out, physical_blocks_[phys].v_data.data() + offset_in_block,
                    elems * sizeof(float));
    }

    void backward(Tensor& grad_key, Tensor& grad_val, int layer, int start_pos, int end_pos) {
        int elems = page_size_ / block_size_;
        for (int pos = start_pos; pos < end_pos; ++pos) {
            auto key = std::make_pair(layer, pos);
            auto it_k = saved_k_.find(key);
            auto it_v = saved_v_.find(key);
            if (it_k != saved_k_.end() && it_v != saved_v_.end()) {
                float* gk = grad_key.data() + (pos - start_pos) * elems;
                float* gv = grad_val.data() + (pos - start_pos) * elems;
                for (int i = 0; i < elems; ++i) {
                    gk[i] += it_k->second[i];
                    gv[i] += it_v->second[i];
                }
            }
        }
    }

    void free_block(int logical_idx) {
        auto it = page_table_.find(logical_idx);
        if (it == page_table_.end()) return;
        physical_blocks_[it->second].used = false;
        free_blocks_.push_back(it->second);
        page_table_.erase(it);
    }

    void reset() {
        for (auto& [logical, phys] : page_table_) {
            physical_blocks_[phys].used = false;
            free_blocks_.push_back(phys);
        }
        page_table_.clear();
        written_positions_.clear();
        saved_k_.clear();
        saved_v_.clear();
    }
};

} // namespace lora_kernel
