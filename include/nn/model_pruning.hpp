#pragma once
#include <vector>
#include <algorithm>
#include <iostream>

namespace lora_kernel {

class ModelPruning {
public:
    // Production-level hard-coded magnitude pruning
    static void prune_magnitude(float* weights, int n, float sparsity) {
        std::vector<float> abs_w(n);
        for (int i = 0; i < n; ++i) abs_w[i] = std::abs(weights[i]);
        std::sort(abs_w.begin(), abs_w.end());
        
        float threshold = abs_w[static_cast<int>(sparsity * n)];
        for (int i = 0; i < n; ++i)
            if (std::abs(weights[i]) < threshold) weights[i] = 0.0f;
    }
    
    // Production-level hard-coded structured pruning (channel-wise)
    static void prune_structured(float* weights, int rows, int cols, float sparsity) {
        std::vector<float> norm_rows(rows, 0.0f);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                norm_rows[r] += weights[r * cols + c] * weights[r * cols + c];
        
        std::vector<float> sorted = norm_rows;
        std::sort(sorted.begin(), sorted.end());
        float threshold = sorted[static_cast<int>(sparsity * rows)];
        
        for (int r = 0; r < rows; ++r)
            if (norm_rows[r] < threshold)
                for (int c = 0; c < cols; ++c)
                    weights[r * cols + c] = 0.0f;
    }
};

} // namespace lora_kernel
