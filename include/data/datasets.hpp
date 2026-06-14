#pragma once
#include <vector>
#include <iostream>

namespace lora_kernel {

class Dataset {
protected:
    size_t size_{0};
public:
    virtual size_t size() const = 0;
    virtual std::vector<float> get_sample(size_t idx) = 0;
};

class ShardedDataset : public Dataset {
private:
    int num_shards_;
    int local_rank_;
public:
    ShardedDataset(int shards, int rank) : num_shards_(shards), local_rank_(rank) {}
    
    size_t size() const override { return 1000000 / num_shards_; }
    
    std::vector<float> get_sample(size_t idx) override {
        return std::vector<float>(768, 0.0f);
    }
};

class DistributedDataLoader {
private:
    std::vector<Dataset*> datasets;
public:
    void add_dataset(Dataset* d) { datasets.push_back(d); }
    
    std::vector<float> get_batch(int batch_size) {
        std::vector<float> batch;
        for (int i = 0; i < batch_size; ++i) {
            auto sample = datasets[0]->get_sample(i);
            batch.insert(batch.end(), sample.begin(), sample.end());
        }
        return batch;
    }
};

} // namespace lora_kernel
