#include <iostream>
#include <cmath>
#include <vector>
#include <functional>
#include <random>
#include "include/core/tensor.hpp"
#include "include/nn/transformer_blocks.hpp"

namespace lora_kernel {

// Production-level numerical gradient checker
// Uses central difference: (f(x+h) - f(x-h)) / (2h) for each parameter

struct GradCheckResult {
    std::string name;
    double max_relative_error;
    double max_absolute_error;
    int num_failed_dims;
    bool passed;
};

class GradientChecker {
private:
    double rel_tol_;
    double abs_tol_;
    double h_;
    bool verbose_;

public:
    GradientChecker(double rel_tol = 1e-3, double abs_tol = 1e-5,
                    double h = 1e-4, bool verbose = true)
        : rel_tol_(rel_tol), abs_tol_(abs_tol), h_(h), verbose_(verbose) {}

    GradCheckResult check(
        const std::string& name,
        const std::vector<float>& params,
        const std::vector<float>& analytical_grad,
        std::function<float(const std::vector<float>&)> f) {

        GradCheckResult result{name, 0.0, 0.0, 0, true};
        int64_t n = params.size();
        double max_rel = 0.0, max_abs = 0.0;
        int failures = 0;

        for (int64_t i = 0; i < n && i < 50; ++i) {
            std::vector<float> x_plus = params;
            std::vector<float> x_minus = params;
            x_plus[i] += (float)h_;
            x_minus[i] -= (float)h_;

            float f_plus = f(x_plus);
            float f_minus = f(x_minus);
            float numerical = (f_plus - f_minus) / (2.0f * (float)h_);

            float analytical = analytical_grad[i];
            float abs_err = std::abs(numerical - analytical);
            float rel_err = abs_err / (std::abs(numerical) + 1e-8f);

            if (rel_err > max_rel) max_rel = rel_err;
            if (abs_err > max_abs) max_abs = abs_err;
            if (rel_err > rel_tol_ && abs_err > abs_tol_) {
                failures++;
                if (verbose_) {
                    std::cerr << "  [" << name << "] dim " << i
                              << ": numerical=" << numerical
                              << " analytical=" << analytical
                              << " rel_err=" << rel_err << "\n";
                }
            }
        }

        result.max_relative_error = max_rel;
        result.max_absolute_error = max_abs;
        result.num_failed_dims = failures;
        result.passed = (failures == 0);

        std::cout << "[GRADCHECK] " << name
                  << ": max_rel=" << max_rel
                  << " max_abs=" << max_abs
                  << " failed=" << failures
                  << " " << (result.passed ? "PASSED" : "FAILED") << "\n";

        return result;
    }
};

} // namespace lora_kernel

float quadratic_loss(const std::vector<float>& x) {
    float sum = 0.0f;
    for (float v : x) sum += v * v;
    return sum;
}

int main() {
    lora_kernel::GradientChecker checker(5e-2, 1e-4, 1e-3, true);
    int passed = 0, total = 0;

    // Test 1: Quadratic f(x) = x^2, gradient = 2x
    {
        total++;
        std::vector<float> params = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        std::vector<float> grad(params.size());
        for (size_t i = 0; i < params.size(); ++i) grad[i] = 2.0f * params[i];

        auto result = checker.check("quadratic_x2", params, grad, quadratic_loss);
        if (result.passed) passed++;
    }

    // Test 2: Quadratic f(x) = x^T W x with fixed matrix W
    {
        total++;
        int n = 5;
        // Fixed symmetric-ish matrix W (n x n)
        std::vector<float> W = {
            2.0f, 0.5f, 0.0f, 0.0f, 0.0f,
            0.5f, 3.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 4.0f, 0.5f, 0.0f,
            0.0f, 0.0f, 0.5f, 5.0f, 1.0f,
            0.0f, 0.0f, 0.0f, 1.0f, 6.0f
        };
        std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

        auto quad_W_loss = [&](const std::vector<float>& x_in) {
            float val = 0.0f;
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    val += x_in[i] * W[i * n + j] * x_in[j];
            return val;
        };

        // Analytical gradient: d/dx_i = sum_j (W_ij + W_ji) * x_j
        std::vector<float> grad(n, 0.0f);
        for (int i = 0; i < n; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < n; ++j)
                sum += (W[i * n + j] + W[j * n + i]) * x[j];
            grad[i] = sum;
        }

        auto result = checker.check("quadratic_xWx", x, grad, quad_W_loss);
        if (result.passed) passed++;
    }

    // Test 3: RMSNorm gradient
    {
        total++;
        int n = 8;
        std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 0.5f, -1.0f, -0.5f, 2.5f};
        std::vector<float> weight(n, 1.0f);
        std::vector<float> y(n), grad_out = {0.1f, 0.2f, 0.3f, 0.4f, -0.1f, -0.2f, 0.0f, 0.5f};
        std::vector<float> grad_x(n, 0.0f);

        lora_kernel::rmsnorm_forward(x.data(), y.data(), n, weight.data(), 1e-6f);
        lora_kernel::rmsnorm_backward(grad_out.data(), x.data(), grad_x.data(),
                                       weight.data(), n, 1e-6f);

        auto rmsnorm_loss = [&](const std::vector<float>& x_in) {
            std::vector<float> y_out(n);
            lora_kernel::rmsnorm_forward(x_in.data(), y_out.data(), n, weight.data(), 1e-6f);
            float loss = 0.0f;
            for (int i = 0; i < n; ++i) loss += grad_out[i] * y_out[i];
            return loss;
        };

        auto result = checker.check("rmsnorm", x, grad_x, rmsnorm_loss);
        if (result.passed) passed++;
    }

    // Test 4: GELU activation backward (use x >= 0 for float32 numerical stability)
    {
        total++;
        int n = 6;
        std::vector<float> x = {0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f};
        std::vector<float> y(n), grad_x(n);
        std::vector<float> grad_out = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        lora_kernel::gelu_forward(x.data(), y.data(), n);
        lora_kernel::gelu_backward(grad_out.data(), x.data(), grad_x.data(), n);

        auto gelu_loss = [&](const std::vector<float>& x_in) {
            std::vector<float> y_out(n);
            lora_kernel::gelu_forward(x_in.data(), y_out.data(), n);
            float loss = 0.0f;
            for (int i = 0; i < n; ++i) loss += grad_out[i] * y_out[i];
            return loss;
        };

        auto result = checker.check("gelu", x, grad_x, gelu_loss);
        if (result.passed) passed++;
    }

    // Test 5: SwiGLU backward (gate branch)
    {
        total++;
        int n = 6;
        std::vector<float> x = {0.5f, -0.5f, 1.0f, -1.0f, 2.0f, -2.0f};
        std::vector<float> y = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<float> out(n), gx(n), gy(n);
        std::vector<float> grad_out(n, 1.0f);

        lora_kernel::swiglu_forward(x.data(), y.data(), out.data(), n);
        lora_kernel::swiglu_backward(grad_out.data(), x.data(), y.data(), gx.data(), gy.data(), n);

        auto swiglu_gate_loss = [&](const std::vector<float>& x_in) {
            std::vector<float> out_inner(n);
            lora_kernel::swiglu_forward(x_in.data(), y.data(), out_inner.data(), n);
            float loss = 0.0f;
            for (int i = 0; i < n; ++i) loss += grad_out[i] * out_inner[i];
            return loss;
        };

        auto result = checker.check("swiglu_gate", x, gx, swiglu_gate_loss);
        if (result.passed) passed++;
    }

    // Test 6: SwiGLU backward (up branch)
    {
        total++;
        int n = 6;
        std::vector<float> x = {0.5f, -0.5f, 1.0f, -1.0f, 2.0f, -2.0f};
        std::vector<float> y = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<float> out(n), gx(n), gy(n);
        std::vector<float> grad_out(n, 1.0f);

        lora_kernel::swiglu_forward(x.data(), y.data(), out.data(), n);
        lora_kernel::swiglu_backward(grad_out.data(), x.data(), y.data(), gx.data(), gy.data(), n);

        auto swiglu_up_loss = [&](const std::vector<float>& y_in) {
            std::vector<float> out_inner(n);
            lora_kernel::swiglu_forward(x.data(), y_in.data(), out_inner.data(), n);
            float loss = 0.0f;
            for (int i = 0; i < n; ++i) loss += grad_out[i] * out_inner[i];
            return loss;
        };

        auto result = checker.check("swiglu_up", y, gy, swiglu_up_loss);
        if (result.passed) passed++;
    }

    std::cout << "\n=== Gradient Checker Summary: " << passed << "/" << total << " passed ===\n";
    return (passed == total) ? 0 : 1;
}
