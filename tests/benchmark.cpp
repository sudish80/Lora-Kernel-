#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <numeric>
#include "include/core/tensor.hpp"
#include "include/core/gemm_engine.hpp"
#include "include/nn/transformer_blocks.hpp"

using namespace lora_kernel;

struct BenchmarkReport {
    struct ForwardPass {
        int seq_len;
        double tok_per_sec;
        double ms_per_step;
    };
    std::vector<ForwardPass> forward_results;

    struct LatencyStats {
        double p50_ms;
        double p95_ms;
        double p99_ms;
        double mean_ms;
    };
    LatencyStats latency;

    struct MemBandwidth {
        int M, N, K;
        double bandwidth_gb_s;
    };
    std::vector<MemBandwidth> bandwidth_results;

    void save(const std::string& path) {
        std::ofstream f(path);
        f << "{\n  \"forward_pass\": [\n";
        for (size_t i = 0; i < forward_results.size(); ++i) {
            auto& r = forward_results[i];
            f << "    {\"seq_len\": " << r.seq_len
              << ", \"tok_per_sec\": " << r.tok_per_sec
              << ", \"ms_per_step\": " << r.ms_per_step << "}";
            if (i + 1 < forward_results.size()) f << ",";
            f << "\n";
        }
        f << "  ],\n  \"latency\": {\n";
        f << "    \"p50_ms\": " << latency.p50_ms << ",\n";
        f << "    \"p95_ms\": " << latency.p95_ms << ",\n";
        f << "    \"p99_ms\": " << latency.p99_ms << ",\n";
        f << "    \"mean_ms\": " << latency.mean_ms << "\n";
        f << "  },\n  \"memory_bandwidth\": [\n";
        for (size_t i = 0; i < bandwidth_results.size(); ++i) {
            auto& r = bandwidth_results[i];
            f << "    {\"M\": " << r.M << ", \"N\": " << r.N << ", \"K\": " << r.K
              << ", \"bandwidth_gb_s\": " << r.bandwidth_gb_s << "}";
            if (i + 1 < bandwidth_results.size()) f << ",";
            f << "\n";
        }
        f << "  ]\n}\n";
        f.close();
        std::cout << "[BENCH] Report saved to " << path << "\n";
    }
};

double measure_forward_pass(Transformer& model, int seq_len, int batch_size = 1) {
    Tensor input({batch_size, seq_len});
    input.uniform(0.0f, (float)model.vocab_size());
    Tensor logits({batch_size, seq_len, model.vocab_size()});

    auto start = std::chrono::high_resolution_clock::now();
    int warmup = 3;
    int iters = 10;
    for (int i = 0; i < warmup; ++i)
        model.forward(input, logits);
    for (int i = 0; i < iters; ++i)
        model.forward(input, logits);
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / iters;
    double total_tokens = batch_size * seq_len * iters;
    double tok_per_sec = total_tokens / (total_ms / 1000.0);
    return tok_per_sec;
}

void measure_latency(Transformer& model, int seq_len, int iterations,
                     double& p50, double& p95, double& p99, double& mean) {
    Tensor input({1, seq_len});
    input.uniform(0.0f, (float)model.vocab_size());
    Tensor logits({1, seq_len, model.vocab_size()});

    std::vector<double> latencies;
    latencies.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        model.forward(input, logits);
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        latencies.push_back(ms);
    }

    std::sort(latencies.begin(), latencies.end());
    mean = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    p50 = latencies[(size_t)(latencies.size() * 0.50)];
    p95 = latencies[(size_t)(latencies.size() * 0.95)];
    p99 = latencies[(size_t)(latencies.size() * 0.99)];
}

double measure_matmul_bandwidth(int M, int N, int K, int iters = 20) {
    Tensor A({M, K});
    Tensor B({K, N});
    Tensor C({M, N});
    A.uniform(-1.0f, 1.0f);
    B.uniform(-1.0f, 1.0f);

    int64_t total_bytes = (int64_t)(M * K + K * N + M * N) * sizeof(float);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i)
        matmul(A, B, C);
    auto end = std::chrono::high_resolution_clock::now();

    double total_sec = std::chrono::duration<double>(end - start).count();
    double avg_sec = total_sec / iters;
    double bandwidth = (total_bytes / avg_sec) / (1024.0 * 1024.0 * 1024.0);
    return bandwidth;
}

int main() {
    std::cout << "=== LoRA Kernel Benchmark ===\n\n";

    TransformerConfig cfg;
    cfg.hidden_dim = 512;
    cfg.num_heads = 8;
    cfg.head_dim = 64;
    cfg.vocab_size = 50257;
    cfg.max_seq_len = 2048;
    cfg.num_layers = 4;
    cfg.dropout = 0.0f;
    cfg.ff_dim = 2048;

    Transformer model(cfg);

    BenchmarkReport report;

    // 1. Forward pass throughput at varying sequence lengths
    std::vector<int> seq_lengths = {128, 512, 2048};
    std::cout << "--- Forward Pass Throughput ---\n";
    for (int sl : seq_lengths) {
        double tokps = measure_forward_pass(model, sl);
        double ms = (1.0 / tokps) * 1000.0 * sl;
        std::cout << "  seq_len=" << sl << ": " << tokps << " tok/s, "
                  << ms << " ms/step\n";
        report.forward_results.push_back({sl, tokps, ms});
    }

    // 2. Latency P50/P95/P99
    std::cout << "\n--- Latency (seq_len=512, 1000 iters) ---\n";
    double p50, p95, p99, mean;
    measure_latency(model, 512, 1000, p50, p95, p99, mean);
    std::cout << "  P50:  " << p50 << " ms\n";
    std::cout << "  P95:  " << p95 << " ms\n";
    std::cout << "  P99:  " << p99 << " ms\n";
    std::cout << "  Mean: " << mean << " ms\n";
    report.latency = {p50, p95, p99, mean};

    // 3. Memory bandwidth of matmul operations
    std::cout << "\n--- Matmul Memory Bandwidth ---\n";
    struct MatmulShape { int M, N, K; };
    std::vector<MatmulShape> shapes = {
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048}
    };
    for (auto& s : shapes) {
        double bw = measure_matmul_bandwidth(s.M, s.N, s.K);
        std::cout << "  " << s.M << "x" << s.N << "x" << s.K
                  << ": " << bw << " GB/s\n";
        report.bandwidth_results.push_back({s.M, s.N, s.K, bw});
    }

    report.save("benchmark_results.json");
    std::cout << "\n=== Benchmark complete ===\n";
    return 0;
}
