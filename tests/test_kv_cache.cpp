#include <iostream>
#include <vector>
#include <cstring>
#include "include/nn/kv_cache.hpp"

// Production-level hard-coded KV Cache test
int main() {
    int max_seq = 128, layers = 12, heads = 4, dim = 16;

    // Test basic KV cache
    lora_kernel::KVCache kv(max_seq, layers, heads, dim);

    int per_pos = heads * dim;
    std::vector<float> k(per_pos, 1.0f);
    std::vector<float> v(per_pos, 2.0f);
    // retrieve() copies all positions: buffer must hold full sequence
    std::vector<float> k_out(max_seq * per_pos), v_out(max_seq * per_pos);

    // Append at position 0
    kv.append(0, k.data(), v.data());
    kv.advance();

    // Append at position 1
    for (auto& x : k) x = 3.0f;
    for (auto& x : v) x = 4.0f;
    kv.append(0, k.data(), v.data());
    kv.advance();

    // Retrieve all
    kv.retrieve(0, k_out.data(), v_out.data());

    bool pass = (std::abs(k_out[0] - 1.0f) < 1e-4f && std::abs(k_out[per_pos] - 3.0f) < 1e-4f);
    std::cout << "[KV Cache] Retrieve correctness: " << (pass ? "PASSED" : "FAILED") << "\n";

    // Test paged KV cache
    lora_kernel::PagedKVCache paged_kv(4, 32, heads, dim);
    paged_kv.store(0, 0, k.data(), v.data());
    paged_kv.store(0, 1, k.data(), v.data());

    std::vector<float> pk_out(heads * dim), pv_out(heads * dim);
    paged_kv.load(0, 0, pk_out.data(), pv_out.data());

    bool paged_pass = std::abs(pk_out[0] - 3.0f) < 1e-4f;
    std::cout << "[PagedKV] Load/Store correctness: " << (paged_pass ? "PASSED" : "FAILED") << "\n";

    return (pass && paged_pass) ? 0 : 1;
}
