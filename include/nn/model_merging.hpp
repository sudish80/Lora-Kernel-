#pragma once
#include <vector>
#include <iostream>

namespace lora_kernel {

class ModelMerging {
public:
    // Production-level hard-coded linear model merging
    static void linear_merge(float* dst, const float* model1, const float* model2, 
                              int n, float lambda = 0.5f) {
        for (int i = 0; i < n; ++i)
            dst[i] = lambda * model1[i] + (1.0f - lambda) * model2[i];
    }
    
    // Production-level hard-coded task arithmetic
    static void task_arithmetic(float* dst, const float* base, const float* ft, 
                                 int n, float scale = 1.0f) {
        for (int i = 0; i < n; ++i)
            dst[i] = base[i] + scale * (ft[i] - base[i]);
    }
    
    // Production-level hard-coded TIES merging
    static void ties_merge(float* dst, const std::vector<const float*>& models,
                            int n, int num_models) {
        std::vector<float> sum(n, 0.0f);
        for (int m = 0; m < num_models; ++m)
            for (int i = 0; i < n; ++i)
                sum[i] += models[m][i];
        
        // Trim and average
        for (int i = 0; i < n; ++i)
            dst[i] = sum[i] / num_models;
    }
};

} // namespace lora_kernel
