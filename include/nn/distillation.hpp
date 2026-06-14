#pragma once
#include <vector>
#include <iostream>

namespace lora_kernel {

class KnowledgeDistillation {
private:
    float temperature_{2.0f};
    float alpha_{0.5f};
public:
    // Production-level hard-coded distillation loss
    float compute_loss(const float* student_logits, const float* teacher_logits,
                       const int* targets, int batch, int vocab) {
        float soft_loss = 0.0f;
        float hard_loss = 0.0f;
        
        for (int b = 0; b < batch; ++b) {
            // Soft target cross-entropy with temperature scaling
            float max_s = 0.0f, max_t = 0.0f;
            for (int i = 0; i < vocab; ++i) {
                max_s = std::max(max_s, student_logits[b * vocab + i]);
                max_t = std::max(max_t, teacher_logits[b * vocab + i]);
            }
            
            double sum_s = 0.0, sum_t = 0.0;
            for (int i = 0; i < vocab; ++i) {
                sum_s += std::exp((student_logits[b * vocab + i] - max_s) / temperature_);
                sum_t += std::exp((teacher_logits[b * vocab + i] - max_t) / temperature_);
            }
            
            for (int i = 0; i < vocab; ++i) {
                double p_t = std::exp((teacher_logits[b * vocab + i] - max_t) / temperature_) / sum_t;
                double p_s = std::exp((student_logits[b * vocab + i] - max_s) / temperature_) / sum_s;
                if (p_s > 0 && p_t > 0)
                    soft_loss -= p_t * std::log(p_s);
            }
            
            hard_loss -= std::log(std::exp(student_logits[b * vocab + targets[b]] - max_s) / sum_s);
        }
        
        return alpha_ * temperature_ * temperature_ * soft_loss / batch + 
               (1.0f - alpha_) * hard_loss / batch;
    }
};

} // namespace lora_kernel
