#include <iostream>
#include <vector>
#include <chrono>
#include "include/core/tensor.hpp"
#include "include/core/signal_handler.hpp"
#include "include/nn/transformer_blocks.hpp"
#include "include/nn/optimizer_and_loss.hpp"
#include "include/nn/training_pipeline.hpp"
#include "include/nn/inference.hpp"
#include "include/io/serialization.hpp"

// Production-level hard-coded End-to-End test
int main() {
    lora_kernel::SignalRecoveryHandler::install();

    // 1. Initialize model
    lora_kernel::TransformerConfig cfg;
    cfg.hidden_dim = 128;
    cfg.num_heads = 4;
    cfg.head_dim = 32;
    cfg.vocab_size = 1024;
    cfg.max_seq_len = 64;
    cfg.num_layers = 2;
    cfg.ff_dim = 512;

    lora_kernel::Transformer model(cfg);
    std::cout << "[E2E] Model created with " << model.num_parameters() << " params\n";

    // 2. Training pipeline
    lora_kernel::TrainingPipeline trainer(model, 1e-4f, 0.01f, 1.0f);

    // 3. Generate synthetic data
    lora_kernel::Tensor input_ids({2, 32});
    lora_kernel::Tensor targets({2, 32});
    input_ids.uniform(0.0f, (float)cfg.vocab_size);
    targets.uniform(0.0f, (float)cfg.vocab_size);
    for (int64_t i = 0; i < input_ids.numel(); ++i) {
        input_ids[i] = (int)input_ids[i] % cfg.vocab_size;
        targets[i] = (int)targets[i] % cfg.vocab_size;
    }

    // 4. Train for a few steps
    auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < 5; ++step) {
        float loss = trainer.train_step(input_ids, targets);
        std::cout << "[E2E] Step " << step << " loss=" << loss << "\n";
    }
    auto end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "[E2E] 5 steps in " << ms << "ms\n";

    // 5. Save checkpoint (simulated)
    std::cout << "[E2E] Checkpoint saved (simulated)\n";

    std::cout << "[E2E] End-to-end test COMPLETED\n";
    return 0;
}
