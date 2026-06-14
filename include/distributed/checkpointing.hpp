#pragma once
#include <string>
#include <fstream>
#include <thread>
#include <vector>
#include <iostream>

namespace lora_kernel {

class DistributedCheckpointer {
private:
    int rank_, world_size_;
    std::string base_path_;
    std::thread async_thread_;
    bool save_in_progress_{false};

public:
    DistributedCheckpointer(int rank, int world_size, const std::string& path)
        : rank_(rank), world_size_(world_size), base_path_(path) {}

    // Each rank saves its own shard
    void save_shard(const float* data, int64_t size) {
        std::string path = base_path_ + ".rank" + std::to_string(rank_);
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(&size), sizeof(size));
        f.write(reinterpret_cast<const char*>(data), size * sizeof(float));
        std::cout << "[CKPT] Rank " << rank_ << " saved shard " << path << "\n";
    }

    // Async save in background thread
    void save_async(const float* data, int64_t size) {
        if (save_in_progress_) return;
        save_in_progress_ = true;
        float* data_copy = new float[size];
        std::memcpy(data_copy, data, size * sizeof(float));

        async_thread_ = std::thread([this, data_copy, size]() {
            save_shard(data_copy, size);
            delete[] data_copy;
            save_in_progress_ = false;
        });
        async_thread_.detach();
    }

    // Load shard
    bool load_shard(float* data, int64_t max_size) {
        std::string path = base_path_ + ".rank" + std::to_string(rank_);
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        int64_t stored_size = 0;
        f.read(reinterpret_cast<char*>(&stored_size), sizeof(stored_size));
        if (stored_size > max_size) return false;
        f.read(reinterpret_cast<char*>(data), stored_size * sizeof(float));
        return true;
    }

    // Metadata file with shard map (rank 0 only)
    void save_metadata(int64_t total_params) {
        if (rank_ != 0) return;
        std::string path = base_path_ + ".meta";
        std::ofstream f(path);
        f << "total_params: " << total_params << "\n";
        f << "world_size: " << world_size_ << "\n";
        f << "shards:\n";
        for (int i = 0; i < world_size_; ++i) {
            int64_t shard_size = total_params / world_size_;
            if (i < total_params % world_size_) shard_size++;
            f << "  - rank: " << i << " size: " << shard_size << "\n";
        }
        std::cout << "[CKPT] Metadata saved\n";
    }
};

} // namespace lora_kernel
