#include <iostream>
#include <vector>
#include "include/nn/transformer_blocks.hpp"
#include "include/nn/optimizer_and_loss.hpp"

bool test_backprop() {
    lora_kernel::TransformerConfig cfg;
    cfg.hidden_dim = 128;
    cfg.num_heads = 4;
    cfg.head_dim = 32;
    cfg.vocab_size = 1024;
    cfg.max_seq_len = 64;
    cfg.num_layers = 1;
    cfg.dropout = 0.0f;
    cfg.ff_dim = 512;

    std::cerr << "creating block...\n";
    lora_kernel::TransformerBlock block(cfg);

    std::cerr << "creating tensors...\n";
    lora_kernel::Tensor input({1, 1, cfg.hidden_dim}, 1.0f);
    lora_kernel::Tensor output({1, 1, cfg.hidden_dim}, 0.0f);
    lora_kernel::Tensor grad_output({1, 1, cfg.hidden_dim}, 0.1f);
    lora_kernel::Tensor grad_input({1, 1, cfg.hidden_dim}, 0.0f);

    std::cerr << "forward...\n";
    block.forward(input, output);
    std::cerr << "backward...\n";
    block.backward(grad_output, grad_input);
    std::cerr << "backward done\n";

    float grad_norm = 0.0f;
    for (int i = 0; i < (int)grad_input.numel(); ++i)
        grad_norm += grad_input.data()[i] * grad_input.data()[i];

    bool pass = grad_norm > 0.0f;
    std::cout << "[TEST] Backpropagation: " << (pass ? "PASSED" : "FAILED")
              << " (grad_norm=" << grad_norm << ")\n";
    return pass;
}

int main() {
    return test_backprop() ? 0 : 1;
}
