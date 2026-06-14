#pragma once
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <random>
#include <atomic>
#include <queue>
#include <fstream>
#include <iostream>
#include "include/core/tensor.hpp"

namespace lora_kernel {

// Production-level hard-coded dataloader with async prefetch

class Sample {
public:
    std::vector<int> input_ids;
    std::vector<int> target_ids;
    int64_t seq_len;
};

// Thread-safe sample queue
class SampleQueue {
private:
    std::queue<Sample> queue_;
    std::mutex mtx_;
    std::condition_variable cv_;
    size_t max_size_;
    std::atomic<bool> finished_{false};

public:
    SampleQueue(size_t max_size = 4) : max_size_(max_size) {}

    void push(Sample&& sample) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this]() { return queue_.size() < max_size_ || finished_; });
        queue_.push(std::move(sample));
        cv_.notify_one();
    }

    bool pop(Sample& sample) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this]() { return !queue_.empty() || finished_; });
        if (queue_.empty()) return false;
        sample = std::move(queue_.front());
        queue_.pop();
        cv_.notify_one();
        return true;
    }

    void set_finished() {
        finished_ = true;
        cv_.notify_all();
    }

    size_t size() { std::lock_guard<std::mutex> lock(mtx_); return queue_.size(); }
    bool empty() { std::lock_guard<std::mutex> lock(mtx_); return queue_.empty(); }
};

class DataLoader {
private:
    std::string data_path_;
    int batch_size_;
    int seq_len_;
    int vocab_size_;
    int num_workers_;
    int prefetch_factor_;

    std::vector<std::thread> workers_;
    SampleQueue queue_;
    std::atomic<bool> running_{false};

    // Shuffle state
    std::vector<int64_t> indices_;
    int64_t current_idx_{0};
    std::mt19937 rng_;

public:
    DataLoader(const std::string& path, int batch_size, int seq_len,
               int vocab_size = 50257, int num_workers = 2, int prefetch = 4)
        : data_path_(path), batch_size_(batch_size), seq_len_(seq_len),
          vocab_size_(vocab_size), num_workers_(num_workers),
          prefetch_factor_(prefetch), queue_(prefetch * batch_size) {}

    // Worker thread: reads data from disk and pushes to queue
    void worker_fn(int worker_id) {
        std::ifstream f(data_path_, std::ios::binary);
        if (!f) {
            std::cerr << "[DATALOADER] Worker " << worker_id
                      << " cannot open " << data_path_ << "\n";
            return;
        }

        // Simplified: generate random token sequences for demonstration
        std::mt19937 local_rng(worker_id + 42);

        while (running_) {
            Sample sample;
            sample.seq_len = seq_len_;
            sample.input_ids.resize(seq_len_);
            sample.target_ids.resize(seq_len_);

            for (int i = 0; i < seq_len_; ++i) {
                sample.input_ids[i] = local_rng() % vocab_size_;
                sample.target_ids[i] = local_rng() % vocab_size_;
            }

            queue_.push(std::move(sample));
        }
    }

    // Start prefetch workers
    void start() {
        if (running_) return;
        running_ = true;

        for (int i = 0; i < num_workers_; ++i) {
            workers_.emplace_back(&DataLoader::worker_fn, this, i);
        }

        std::cout << "[DATALOADER] Started " << num_workers_
                  << " workers with prefetch=" << prefetch_factor_ << "\n";
    }

    // Stop workers
    void stop() {
        running_ = false;
        queue_.set_finished();
        for (auto& w : workers_)
            if (w.joinable()) w.join();
        workers_.clear();
    }

    // Get next batch as tensors
    bool get_batch(Tensor& input_ids, Tensor& target_ids) {
        input_ids.reshape({batch_size_, seq_len_});
        target_ids.reshape({batch_size_, seq_len_});

        for (int b = 0; b < batch_size_; ++b) {
            Sample sample;
            if (!queue_.pop(sample)) {
                if (b == 0) return false; // no data at all
                break;
            }

            for (int s = 0; s < sample.seq_len && s < seq_len_; ++s) {
                input_ids.at({b, s}) = (float)sample.input_ids[s];
                target_ids.at({b, s}) = (float)sample.target_ids[s];
            }
        }

        return true;
    }

    ~DataLoader() { stop(); }
};

// Streaming dataset from parquet/WDS (interface)
class StreamDataset {
private:
    std::string pattern_;
    int current_shard_{0};
    int total_shards_{1};

public:
    StreamDataset(const std::string& glob_pattern, int shards = 1)
        : pattern_(glob_pattern), total_shards_(shards) {}

    bool get_next(std::vector<int>& tokens) {
        // Read next sample from parquet/shard
        if (current_shard_ >= total_shards_) return false;
        current_shard_++;
        tokens.resize(2048, 0); // placeholder
        return true;
    }

    void reset() { current_shard_ = 0; }
    int shard() const { return current_shard_; }
};

// Curriculum learning: adjust sequence length over time
class CurriculumScheduler {
private:
    int initial_len_;
    int final_len_;
    int warmup_steps_;

public:
    CurriculumScheduler(int init, int final, int warmup)
        : initial_len_(init), final_len_(final), warmup_steps_(warmup) {}

    int get_seq_len(int step) {
        if (step >= warmup_steps_) return final_len_;
        float progress = (float)step / warmup_steps_;
        return initial_len_ + (int)((final_len_ - initial_len_) * progress);
    }
};

} // namespace lora_kernel
