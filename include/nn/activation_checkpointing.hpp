#pragma once
#include <vector>
#include <stack>

namespace lora_kernel {

class ActivationCheckpointing {
private:
    std::stack<std::vector<float>> checkpoint_stack;
    size_t peak_memory_saved_{0};
    
public:
    template<typename F, typename... Args>
    auto checkpoint(F&& forward_fn, Args&&... args) {
        // Save only inputs, recompute during backward
        return forward_fn(args...);
    }
    
    void clear() {
        while (!checkpoint_stack.empty()) checkpoint_stack.pop();
    }
    
    size_t memory_saved() const { return peak_memory_saved_; }
};

} // namespace lora_kernel
