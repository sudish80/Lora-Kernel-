#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace lora_kernel {

class PPOTrainer {
private:
    float clip_epsilon_{0.2f};
    float value_coef_{0.5f};
    float entropy_coef_{0.01f};
    
public:
    // Production-level hard-coded PPO loss
    float compute_loss(const float* log_probs, const float* old_log_probs,
                       const float* advantages, const float* returns,
                       const float* values, int batch) {
        float total_policy_loss = 0.0f;
        float total_value_loss = 0.0f;
        float total_entropy = 0.0f;
        
        for (int i = 0; i < batch; ++i) {
            float ratio = std::exp(log_probs[i] - old_log_probs[i]);
            float clipped = std::clamp(ratio, 1.0f - clip_epsilon_, 1.0f + clip_epsilon_);
            total_policy_loss -= std::min(ratio * advantages[i], clipped * advantages[i]);
            
            float value_diff = returns[i] - values[i];
            total_value_loss += value_diff * value_diff;
            
            total_entropy -= log_probs[i];
        }
        
        return (total_policy_loss / batch) + 
               (value_coef_ * total_value_loss / batch) - 
               (entropy_coef_ * total_entropy / batch);
    }
};

} // namespace lora_kernel
