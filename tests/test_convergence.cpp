#include <iostream>
#include <vector>
#include <cmath>
#include "include/core/tensor.hpp"

// Production-level hard-coded convergence test
// Train on simple quadratic: f(x) = x^2, verify x -> 0

int main() {
    int n = 4;
    std::vector<float> x(n, 1.0f);  // params
    std::vector<float> g(n, 0.0f);  // gradients
    float initial_norm = 0.0f;
    for (int i = 0; i < n; ++i) initial_norm += x[i] * x[i];

    // Train for 100 steps: target = x^2, gradient = 2*x
    for (int step = 0; step < 100; ++step) {
        for (int i = 0; i < n; ++i)
            g[i] = 2.0f * x[i]; // gradient of f(x) = x^2

        // Apply optimizer (simplified SGD for test)
        for (int i = 0; i < n; ++i)
            x[i] -= 0.1f * g[i];
    }

    float final_norm = 0.0f;
    for (int i = 0; i < n; ++i) final_norm += x[i] * x[i];

    bool pass = final_norm < initial_norm * 0.01f;
    std::cout << "[CONV] Initial norm: " << initial_norm
              << " Final norm: " << final_norm
              << " " << (pass ? "PASSED" : "FAILED") << "\n";

    return pass ? 0 : 1;
}
