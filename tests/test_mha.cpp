#include <iostream>
#include "include/core/tensor.hpp"
#include "include/nn/transformer_blocks.hpp"

// Production-level hard-coded MHA test
int main() {
    lora_kernel::TransformerConfig cfg;
    cfg.hidden_dim = 64;
    cfg.num_heads = 4;
    cfg.head_dim = 16;
    cfg.max_seq_len = 8;

    lora_kernel::MultiHeadAttention mha(cfg);

    // Random input
    lora_kernel::Tensor input({1, 4, 64});
    input.normal(0.0f, 1.0f);
    lora_kernel::Tensor output({1, 4, 64});

    mha.forward(input, output);

    // Check output is finite
    bool all_finite = true;
    for (int64_t i = 0; i < output.numel(); ++i)
        if (!std::isfinite(output[i])) { all_finite = false; break; }

    bool pass = all_finite;
    std::cout << "[MHA] Forward output finite: " << (pass ? "PASSED" : "FAILED") << "\n";

    // Backward test
    lora_kernel::Tensor grad_output({1, 4, 64});
    grad_output.fill(1.0f);
    lora_kernel::Tensor grad_input({1, 4, 64});
    lora_kernel::Tensor grad_Wq({64, 64}), grad_Wk({64, 64}), grad_Wv({64, 64}), grad_Wo({64, 64});

    mha.backward(grad_output, grad_input, grad_Wq, grad_Wk, grad_Wv, grad_Wo);

    bool grad_finite = true;
    for (int64_t i = 0; i < grad_input.numel(); ++i)
        if (!std::isfinite(grad_input[i])) { grad_finite = false; break; }

    std::cout << "[MHA] Backward grad_input finite: " << (grad_finite ? "PASSED" : "FAILED") << "\n";

    return (pass && grad_finite) ? 0 : 1;
}
