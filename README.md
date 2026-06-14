# LoRA Kernel

Production-grade C++ LLM training and inference engine with full transformer backpropagation, distributed training, quantization, serving, and monitoring.

## Features

- **Transformer** — Multi-head attention (MHA/GQA/MQA), FeedForward with SwiGLU, RoPE, RMSNorm, FlashAttention v2
- **Full backpropagation** — Numerical gradient verified backward pass through all layers
- **Training pipeline** — AdamW, gradient accumulation, dynamic loss scaling, gradient clipping, SWA/EMA
- **Quantization** — INT4 group-wise, FP8 E4M3/E5M2, dynamic/per-channel, static calibration, QAT, GPTQ, AWQ, SmoothQuant, KV cache quant
- **Distributed** — NCCL/MPI backends, ring all-reduce, FSDP, tensor parallelism, ZeRO-1/2/3, gradient compression
- **Serving** — REST API (OpenAI-compatible `/v1/completions`, `/v1/chat/completions`), SSE streaming, dynamic batching, prefix caching
- **Monitoring** — Prometheus exporter, latency histograms (P50/95/99), throughput monitor, GPU/PCIe utilization, distributed tracing
- **Security** — JWT auth (HMAC-SHA256), token bucket rate limiting, RBAC, TLS, encrypted weights, audit log
- **Hardware** — CUDA kernels (tiled GEMM, warp-level softmax, LayerNorm), NUMA-aware allocation, huge pages, CPU feature detection (AVX512/AVX2/NEON)
- **Python bindings** — pybind11 module for Tensor, Transformer, TrainingPipeline

## Quick Start

```bash
# Dependencies
sudo apt install -y cmake g++ libomp-dev libssl-dev

# Optional (GPU acceleration)
sudo apt install -y nvidia-cuda-toolkit libnccl-dev

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run tests
cd build && ctest --output-on-failure

# Training demo
./lora_kernel train

# Benchmark
./lora_kernel bench

# HTTP server
./lora_kernel serve
```

## Project Structure

```
├── CMakeLists.txt              # Build system (CUDA/NCCL/MKL/OpenMP)
├── Dockerfile                  # Production container
├── ci.sh                       # CI verification script
├── .github/workflows/ci.yml    # GitHub Actions CI
├── main.cpp                    # Entry point: train/bench/serve
├── bindings/python_module.cpp  # Python bindings (pybind11)
├── include/
│   ├── core/                   # Tensor, GEMM, memory, logging, profiler, error handling
│   ├── nn/                     # Transformer, attention, FFN, training, quantization, KV cache
│   ├── distributed/            # NCCL/MPI, FSDP, ZeRO, gradient compression
│   ├── api/                    # REST server, JWT auth, rate limiting
│   ├── monitoring/             # Prometheus, metrics, tracing, alerts, dashboards
│   ├── serving/                # ONNX Runtime backend
│   ├── hw/                     # CUDA kernels, memory manager, topology detection
│   ├── data/                   # Dataloader, tokenizer, datasets
│   ├── io/                     # Serialization, mmap loader
│   ├── crypto/                 # Encryption engine
│   └── tensor/                 # safetensors format
├── src/                        # Compiled .cpp files
└── tests/                      # CTest targets
    ├── test_gradient_checker   # Numerical gradient verification
    ├── test_mha                # Multi-head attention forward
    ├── test_kv_cache           # KV cache store/load
    ├── test_backprop           # Full backward pass verification
    ├── test_convergence        # Training convergence test
    ├── test_e2e                # End-to-end pipeline
    ├── test_nn                 # RMSNorm and layer tests
    ├── test_distributed        # Distributed backend init
    └── benchmark               # Tok/sec, latency P50/95/99, bandwidth
```

## Configuration

```cpp
TransformerConfig cfg;
cfg.hidden_dim = 768;       // Model dimension
cfg.num_heads = 12;         // Attention heads
cfg.head_dim = 64;          // Head dimension
cfg.vocab_size = 50257;     // Vocabulary size
cfg.max_seq_len = 2048;     // Maximum sequence length
cfg.num_layers = 12;        // Transformer layers
cfg.dropout = 0.1f;         // Dropout rate
cfg.ff_dim = 3072;          // Feed-forward dimension
```

## Training

```cpp
Transformer model(cfg);
TrainingPipeline pipeline(model, 3e-4f, 0.01f, 1.0f);
pipeline.set_gradient_accumulation_steps(4);
pipeline.init_ema(0.995f);

Tensor input_ids({B, S}), targets({B, S});
float loss = pipeline.train_step(input_ids, targets);
```

## Serving

```bash
./lora_kernel serve
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 50}'
```

## Tests

| Test | What it verifies |
|---|---|
| `gradient_checker` | Numerical gradient via central differences (rel error < 1e-3) |
| `backprop` | Full backward pass produces non-zero gradients |
| `convergence` | Model converges on a simple quadratic |
| `mha` | Multi-head attention forward pass produces output |
| `kv_cache` | KV cache store/load/append works correctly |
| `nn` | RMSNorm forward/backward produces correct shapes |
| `distributed` | NCCL/MPI/ZeRO init succeeds |
| `e2e` | Signal handler + config + forward pipeline |
| `benchmark` | Tok/sec, latency distribution, memory bandwidth |

## License

MIT
