# LoRA Kernel

Production-grade C++ LLM training and inference engine with full transformer backpropagation, distributed training, quantization, serving, and monitoring.

> **Author:** Sudish Deuja  
> **Repository:** [github.com/sudish80/Lora-Kernel-](https://github.com/sudish80/Lora-Kernel-)

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Training](#training)
- [Serving](#serving)
- [Build Options](#build-options)
- [Tests](#tests)
- [Performance](#performance)
- [Math Reference](#math-reference)
- [License](#license)

---

## Features

### Core
- **Tensor** — Multi-dimensional array with shape/broadcasting, copy-on-write views, strides-based indexing, CPU feature detection (AVX512/AVX2/NEON), memory pooling, Unified Memory support
- **GEMM Engine** — Optimized matrix multiplication with auto-selection of kernel (AVX512, AVX2, AVX, scalar fallback)
- **Memory Manager** — Pool allocator with leak detection, aligned allocations, huge page support, NUMA-aware placement

### Transformer
- **Multi-Head Attention** — Full forward and numerically-verified backward pass (QKV projection → RoPE → scores → causal mask → softmax → context → output projection)
- **FeedForward** — SwiGLU activation: `SwiGLU(x) = (x·W_gate ⊙ σ(x·W_gate)) · (x·W_up) · W_down`
- **RMSNorm** — Root Mean Square layer normalization with full backward pass
- **RoPE** — Rotary Position Embeddings with forward and backward
- **FlashAttention v2** — Tiled attention kernel (CPU SIMD optimized)

### Training Pipeline
- **AdamW Optimizer** — Decoupled weight decay with bias correction
- **Gradient Clipping** — Global L2 norm clipping
- **Dynamic Loss Scaling** — FP16-friendly scale management with overflow detection
- **Gradient Accumulation** — Simulate larger batch sizes
- **SWA / EMA** — Stochastic Weight Averaging and Exponential Moving Average
- **Learning Rate Scheduling** — Cosine schedule with warmup
- **Fused Softmax + Cross-Entropy** — Memory-efficient loss computation

### Quantization
- **INT4** — Group-wise symmetric/asymmetric, GPTQ one-shot, AWQ per-channel scaling
- **FP8** — E4M3/E5M2 formats, dynamic and per-tensor quantization
- **INT8** — Static calibration, QAT (quantization-aware training)
- **SmoothQuant** — Activation migration for INT8 inference
- **KV Cache** — INT8/FP8 quantized cache with selective retention
- **Model Pruning** — Magnitude-based, SparseGPT, gradient-sensitive pruning

### Distributed
- **Backends** — NCCL and MPI support
- **All-Reduce** — Ring algorithm for gradient synchronization
- **ZeRO** — Stage 1 (optimizer states), Stage 2 (+ gradients), Stage 3 (+ parameters)
- **FSDP** — Fully Sharded Data Parallelism
- **Tensor Parallelism** — Column/row-wise linear splitting
- **Gradient Compression** — Top-K sparsification, random masking

### Serving
- **REST API** — OpenAI-compatible `/v1/completions` and `/v1/chat/completions`
- **Streaming** — Server-Sent Events (SSE) for token-by-token output
- **Dynamic Batching** — Continuous batching with timeout-based scheduling
- **Prefix Caching** — KV cache reuse across requests with shared prefix
- **Rate Limiting** — Token bucket algorithm, per-user quotas
- **Authentication** — JWT (HMAC-SHA256), RBAC, API key validation
- **TLS** — HTTPS endpoint with certificate management

### Monitoring
- **Prometheus Exporter** — `/metrics` endpoint with custom metric registry
- **Latency** — P50, P95, P99 histograms per operation
- **Throughput** — Tokens per second, requests per second
- **GPU** — Utilization, memory, PCIe bandwidth, temperature
- **Tracing** — Distributed tracing with span hierarchy
- **Health** — Liveness/readiness probes, signal recovery handler
- **Security Audit** — All requests logged with timestamp, user, action, result

---

## Quick Start

### Linux (Ubuntu/Debian)

```bash
# Install dependencies
sudo apt update && sudo apt install -y cmake g++ libomp-dev libssl-dev

# Optional: GPU acceleration
sudo apt install -y nvidia-cuda-toolkit libnccl-dev

# Build
git clone https://github.com/sudish80/Lora-Kernel-.git
cd Lora-Kernel-
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

### Windows (MSYS2 UCRT64)

```bash
# In MSYS2 UCRT64 terminal:
pacman -S mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-openblas

cmake -G "MinGW Makefiles" -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM=mingw32-make
cmake --build build -j8

cd build && ctest --output-on-failure
```

---

## Architecture

### Data Flow

```
Input IDs (int)
    │
    ▼
Embedding (token + position)
    │
    ▼
┌─────────────────────────────────────┐
│         TransformerBlock × N        │
│  ┌─────────┐    ┌─────────────────┐  │
│  │ RMSNorm │    │ MultiHeadAttn   │  │
│  │ Forward │───▶│ QKV → RoPE      │  │
│  │ Backward│    │ → Scores → SM   │  │
│  └─────────┘    │ → Context → Out │  │
│                 └────────┬────────┘  │
│  ┌─────────┐            │           │
│  │ Residual│◀───────────┘           │
│  └────┬────┘                        │
│       │                             │
│  ┌─────────┐    ┌─────────────────┐  │
│  │ RMSNorm │    │ FeedForward     │  │
│  │ Forward │───▶│ SwiGLU:         │  │
│  │ Backward│    │ Gate·Up·Down    │  │
│  └─────────┘    └────────┬────────┘  │
│  ┌─────────┐            │           │
│  │ Residual│◀───────────┘           │
│  └────┬────┘                        │
│       │                             │
└───────┼─────────────────────────────┘
        │
        ▼
    RMSNorm (pre-head)
        │
        ▼
    LM Head (vocab projection)
        │
        ▼
    Logits → Softmax → Loss
```

### Gradient Flow (Backward Pass)

The backward pass computes gradients via the chain rule, flowing in reverse:

```
dL/dLogits ← CrossEntropy
    │
    ▼
dL/dLmHead → grad_lm_head
    │
    ▼
┌──────────────────────────────────────┐
│     TransformerBlock Backward × N    │
│                                      │
│  FeedForward Backward:               │
│    dL/dh = grad_output @ Wdown^T     │
│    dL/dWdown = last_gated^T @ grad   │
│    dL/dWgate, dL/dWup via SwiGLU     │
│    dL/dWgate = input^T @ (dL/dgate)  │
│                                      │
│  RMSNorm Backward:                   │
│    dL/dx = inv_rms · (dL/dy -        │
│             x · Σ(dL/dy·x) / Σ(x²))  │
│                                      │
│  MultiHeadAttention Backward:         │
│    dL/dV = softmax^T @ dL/dcontext   │
│    dL/dQ = dL/dscores @ K            │
│    dL/dK = dL/dscores^T @ Q          │
│    dL/dWq = input^T @ dL/dQ          │
│    RoPE backward (inverse rotation)  │
│                                      │
└──────────────────────────────────────┘
    │
    ▼
dL/dEmbedding → grad_token_embed, grad_pos_embed
```

All backward passes are numerically verified via central difference gradient checking (see `test_gradient_checker`).

---

## Project Structure

```
├── CMakeLists.txt              # Build system (CUDA/NCCL/MKL/OpenMP)
├── Dockerfile                  # Production container
├── ci.sh                       # CI verification script
├── .github/workflows/ci.yml    # GitHub Actions CI
├── main.cpp                    # Entry point: train/bench/serve
├── math.md                     # Full math derivations for all layers
├── bindings/python_module.cpp  # Python bindings (pybind11)
├── include/
│   ├── core/                   # Tensor, GEMM, memory, logging, profiler
│   │   ├── tensor.hpp          # Multi-dimensional array with strides
│   │   ├── gemm_engine.hpp     # SIMD-optimized matrix multiply
│   │   ├── memory_manager.hpp  # Pool allocator with leak detection
│   │   ├── logging.hpp         # Structured logging (INFO/WARN/ERROR/DEBUG)
│   │   ├── profiler.hpp        # Performance counter collection
│   │   ├── signal_handler.hpp  # Graceful shutdown on SIGINT/SIGTERM
│   │   └── scheduler.hpp       # Thread pool and task scheduler
│   ├── nn/                     # Neural network layers
│   │   ├── transformer_blocks.hpp/pp  # MHA, FFN, TransformerBlock
│   │   ├── training_pipeline.hpp      # Full training loop
│   │   ├── optimizer_and_loss.hpp     # AdamW, cross-entropy, clipping
│   │   ├── kv_cache.hpp              # Standard KV cache
│   │   ├── paged_kv_cache.hpp        # Paged (vLLM-style) KV cache
│   │   ├── flash_attention.hpp       # Tiled attention kernel
│   │   ├── quantization_suites.hpp   # INT4/FP8/INT8 quantization
│   │   ├── lora.hpp                  # LoRA adapters
│   │   ├── distillation.hpp          # Knowledge distillation
│   │   └── weight_init.hpp           # Kaiming/Xavier initialization
│   ├── distributed/
│   │   ├── nccl_wrapper.hpp     # NCCL communicator
│   │   ├── zero_optimizer.hpp   # ZeRO stage 1/2/3
│   │   └── distributed_training.hpp  # FSDP, tensor parallelism
│   ├── api/
│   │   ├── rest_api.hpp         # OpenAI-compatible HTTP server
│   │   ├── websocket_streaming.hpp   # SSE token streaming
│   │   └── security.hpp         # JWT, RBAC, rate limiting
│   ├── monitoring/
│   │   └── production_monitoring.hpp  # Prometheus, metrics, tracing
│   ├── serving/
│   │   └── onnx_backend.hpp     # ONNX Runtime inference
│   ├── hw/
│   │   ├── cuda_backend.hpp     # CUDA kernel wrappers
│   │   └── other_backends.hpp   # ROCm, oneAPI placeholders
│   ├── data/
│   │   ├── tokenizers.hpp       # Byte-Pair Encoding tokenizer
│   │   └── dataloader.hpp       # Batch data loading
│   ├── io/
│   │   ├── serialization.hpp    # Model save/load
│   │   └── mmap_loader.hpp      # Memory-mapped weight loading
│   ├── crypto/
│   │   └── crypto_engine.hpp    # AES-256-GCM encryption
│   └── tensor/
│       └── safetensors.hpp      # safetensors format reader
├── src/
│   ├── core/signal_handler.cpp
│   └── nn/transformer_blocks.cpp  # Compiled MHA/FFN/Block code
└── tests/
    ├── test_gradient_checker.cpp  # Numerical gradient verification
    ├── test_mha.cpp               # Multi-head attention forward
    ├── test_kv_cache.cpp          # KV cache store/load
    ├── test_backprop.cpp          # Full backward pass verification
    ├── test_convergence.cpp       # Training convergence test
    ├── test_e2e.cpp               # End-to-end pipeline
    ├── test_nn.cpp                # RMSNorm and layer tests
    ├── test_distributed.cpp       # Distributed backend init
    └── benchmark.cpp              # Tok/sec, latency, bandwidth
```

---

## Configuration

```cpp
#include "include/nn/transformer_blocks.hpp"
using namespace lora_kernel;

TransformerConfig cfg;
cfg.hidden_dim = 768;       // Model dimension (d_model)
cfg.num_heads = 12;         // Number of attention heads
cfg.head_dim = 64;          // Dimension per head (d_model / num_heads)
cfg.vocab_size = 50257;     // Vocabulary size (GPT-2)
cfg.max_seq_len = 2048;     // Maximum sequence length
cfg.num_layers = 12;        // Number of transformer blocks
cfg.dropout = 0.1f;         // Dropout probability
cfg.ff_dim = 3072;          // Feed-forward hidden dim (typically 4× d_model)
```

---

## API Reference

### Tensor

```cpp
// Construction
Tensor a({3, 4});                          // {3,4} uninitialized
Tensor b({3, 4}, 1.0f);                    // {3,4} filled with 1.0
Tensor c = b;                              // Deep copy

// Shape and access
int64_t n = a.numel();                     // Total elements
int64_t d = a.ndim();                      // Number of dimensions
int64_t s = a.size(0);                     // Size of dimension 0
float* p = a.data();                       // Raw pointer
float v = a.at({1, 2});                    // Indexed access
a[{1, 2}] = 3.0f;                          // Indexed write

// Operations
a.reshape({6, 2});                         // View with new shape
a.zeros(); a.ones(); a.fill(0.5f);         // Fill operations
a.normal(0.0f, 1.0f); a.uniform(-1, 1);    // Random init
Tensor t = a.transpose();                  // 2D transpose (deep copy)
a.copy_from(src_ptr, n);                   // Raw copy from buffer
a.add(b); a.mul(b);                        // Element-wise operations
```

### Transformer

```cpp
// Create model
Transformer model(cfg);

// Forward pass
Tensor input_ids({B, S});                   // Token indices
Tensor logits({B, S, cfg.vocab_size});      // Output logits
model.forward(input_ids, logits);

// Backward pass
Tensor grad_logits({B, S, cfg.vocab_size}); // Gradient from loss
model.backward(grad_logits);                 // Accumulates param grads

// Access parameters
int64_t n = model.num_parameters();         // Total parameter count
auto& params = model.get_params();          // Vector of Tensor*
auto& grads = model.get_param_grads();      // Vector of Tensor
```

### Training Pipeline

```cpp
// Setup
TrainingPipeline pipeline(model, lr=3e-4f, weight_decay=0.01f, max_grad_norm=1.0f);

// Optional features
pipeline.set_gradient_accumulation_steps(4);  // Simulate 4× batch size
pipeline.init_swa(start_step=1000);            // Stochastic Weight Averaging
pipeline.init_ema(decay=0.999f);               // Exponential Moving Average

// Training loop
for (int step = 0; step < 1000; ++step) {
    Tensor input_ids({B, S}), targets({B, S});
    // ... fill with token data ...
    float loss = pipeline.train_step(input_ids, targets);
    std::cout << "Step " << step << " loss=" << loss << "\n";
}

// Get averaged weights
std::vector<float> swa_weights(model.num_parameters());
pipeline.get_swa_weights(swa_weights.data());
```

### LoRA Adapter

```cpp
// Apply LoRA to a linear layer
LoRAAdapter adapter(in_features=768, out_features=768, rank=8, alpha=16.0f);

// Forward: y = Wx + (α/r) · B·A·x
adapter.forward(x, y);

// Backward accumulates gradients for A and B
adapter.backward(grad_y, grad_x);
```

---

## Training

### Basic Training Loop

```cpp
#include "include/nn/transformer_blocks.hpp"
#include "include/nn/training_pipeline.hpp"

using namespace lora_kernel;

int main() {
    // Configure model (small for demo)
    TransformerConfig cfg;
    cfg.hidden_dim = 128;
    cfg.num_heads = 4;
    cfg.head_dim = 32;
    cfg.vocab_size = 1024;
    cfg.max_seq_len = 64;
    cfg.num_layers = 2;
    cfg.ff_dim = 512;

    // Create model and pipeline
    Transformer model(cfg);
    TrainingPipeline pipeline(model, 1e-3f, 0.01f, 1.0f);

    // Training loop
    Tensor input_ids({4, 32});
    Tensor targets({4, 32});
    for (int step = 0; step < 100; ++step) {
        // Generate synthetic data
        input_ids.uniform(0, cfg.vocab_size);
        targets.uniform(0, cfg.vocab_size);

        float loss = pipeline.train_step(input_ids, targets);
        std::cout << "Step " << step << ": loss=" << loss << "\n";
    }

    // Save model
    serialize("model_checkpoint.bin", model);
    return 0;
}
```

### Gradient Accumulation

For large batch sizes when GPU memory is limited:

```cpp
pipeline.set_gradient_accumulation_steps(4);

// Each call to train_step accumulates gradients
// After 4 calls, the optimizer performs one update
```

### Mixed Precision

FP16 training with dynamic loss scaling is available via `DynamicLossScaler`, which automatically detects overflows and adjusts the scale factor.

---

## Serving

### Start Server

```bash
./lora_kernel serve --port 8080 --model ./checkpoints/model.bin
```

### REST API Endpoints

#### Completions (non-chat)

```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "prompt": "The meaning of life is",
    "max_tokens": 50,
    "temperature": 0.8,
    "top_p": 0.95,
    "stream": false
  }'
```

#### Chat Completions

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "model": "lora-kernel",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is LoRA?"}
    ],
    "stream": true
  }'
```

#### Streaming (SSE)

Set `"stream": true` in the request body. The server returns `text/event-stream`:

```
data: {"token": "LoRA", "index": 0}

data: {"token": " stands", "index": 0}

data: {"token": " for", "index": 0}

data: [DONE]
```

---

## Build Options

| Flag | Description |
|---|---|
| `-DCMAKE_BUILD_TYPE=Release` | Release build with -O3 optimizations |
| `-DCMAKE_BUILD_TYPE=Debug` | Debug build with -g and no optimizations |
| `-DCUDA_ENABLED=ON` | Enable CUDA GPU acceleration |
| `-DNCCL_ENABLED=ON` | Enable NCCL distributed backend |
| `-DINTEL_MKL_VERSION=ON` | Use Intel MKL for GEMM operations |

### MSYS2 (Windows)

Prerequisites: Install MSYS2 with UCRT64 environment, then:

```bash
pacman -S mingw-w64-ucrt-x86_64-cmake \
          mingw-w64-ucrt-x86_64-gcc \
          mingw-w64-ucrt-x86_64-openblas \
          mingw-w64-ucrt-x86_64-openssl

cmake -G "MinGW Makefiles" -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_MAKE_PROGRAM=mingw32-make
cmake --build build -j8
```

---

## Tests

| Test | What it verifies | Status |
|---|---|---|
| `gradient_checker` | Numerical gradient via central differences (6 checks: quadratic, RMSNorm, GELU, SwiGLU) | ✅ |
| `backprop` | Full backward pass through TransformerBlock produces non-zero gradients | ✅ |
| `convergence` | Model converges on a simple quadratic (final norm ~1.6e-19) | ✅ |
| `mha` | Multi-head attention forward/backward produces finite outputs | ✅ |
| `kv_cache` | KV cache store/load/append works correctly | ✅ |
| `nn` | RMSNorm forward/backward produces correct shapes and values | ✅ |
| `distributed` | NCCL/MPI/ZeRO init succeeds (CPU fallback) | ✅ |
| `e2e` | End-to-end: model creation → training step → loss computation | ✅ |
| `benchmark` | Tok/sec, latency distribution, memory bandwidth (informational) | — |

### Running Tests

```bash
# All tests
cd build && ctest --output-on-failure

# Single test
./build/test_backprop.exe
./build/test_gradient_checker.exe
```

### Gradient Checker Details

The gradient checker verifies backward pass correctness using central difference approximation:

```
∂L/∂θᵢ ≈ (L(θ + ε·eᵢ) − L(θ − ε·eᵢ)) / 2ε
```

Relative error threshold: < 5e-2, absolute threshold: < 1e-4

Verified operations:
- Quadratic (x², xWx) — tests basic GEMM backward
- RMSNorm — tests normalization backward
- GELU — tests activation backward (non-negative domain)
- SwiGLU (gate and up branches) — tests gated activation backward

---

## Performance

Benchmarks measure:
- **Token throughput** — tokens generated per second
- **Latency** — P50, P95, P99 end-to-end latency
- **Memory bandwidth** — effective GB/s for GEMM operations

Run `./lora_kernel bench` for detailed results.

### CPU Optimizations

The GEMM engine automatically selects the best available kernel:
1. AVX-512 (512-bit vector registers, 16× float throughput)
2. AVX2 (256-bit, 8× float throughput)
3. AVX (128-bit, 4× float throughput)
4. Scalar fallback

Detection is performed at runtime via `CPUFeatures::detect()`.

---

## Math Reference

Full mathematical derivations for all layers are in [`math.md`](math.md), including:

- Multi-Head Attention forward/backward (QKV, scores, softmax, context, output)
- RoPE forward/backward (rotary interpolation)
- RMSNorm forward/backward (closed-form gradient)
- SwiGLU forward/backward (sigmoid-gated linear unit)
- GELU forward/backward (Gaussian Error Linear Unit)
- Cross-entropy loss gradient
- Layer normalization backward
- AdamW update rule

---

## License

MIT
