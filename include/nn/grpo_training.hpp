#pragma once
#include <vector>
#include <cmath>
#include <iostream>

namespace lora_kernel {

// GRPO (Group Relative Policy Optimization) - Production-level implementation
class GRPOTrainer {
private:
    int group_size_{8};
    float clip_epsilon_{0.2f};
    
public:
    float compute_loss(const std::vector<std::vector<float>>& group_log_probs,
                       const std::vector<float>& group_rewards,
                       const float* old_log_probs, int batch) {
        // Compute group mean and std for normalization
        float mean = 0.0f, var = 0.0f;
        for (float r : group_rewards) mean += r;
        mean /= group_rewards.size();
        for (float r : group_rewards) { float d = r - mean; var += d * d; }
        float std = std::sqrt(var / group_rewards.size() + 1e-8f);
        
        float total_loss = 0.0f;
        for (int b = 0; b < batch; ++b) {
            // Normalize rewards within group
            float adv = (group_rewards[b % group_size_] - mean) / std;
            
            // Clipped surrogate objective
            for (int g = 0; g < group_size_; ++g) {
                float ratio = std::exp(group_log_probs[b][g] - old_log_probs[b]);
                ratio = std::clamp(ratio, 1.0f - clip_epsilon_, 1.0f + clip_epsilon_);
                total_loss -= ratio * adv;
            }
        }
        return total_loss / (batch * group_size_);
    }
};

} // namespace lora_kernel
