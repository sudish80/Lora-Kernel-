#include <iostream>
#include <vector>
#include <cmath>
#include "include/nn/rms_norm.hpp"

// Simple unit test for RMSNorm
bool test_rmsnorm() {
    lora_kernel::RMSNorm norm(4);
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> output(4);
    norm.forward(input.data(), output.data(), 4);
    
    // Basic verification: output should not be all zeros
    bool pass = true;
    for(float f : output) if (std::abs(f) < 1e-6f) pass = false;
    
    std::cout << "[TEST] RMSNorm: " << (pass ? "PASSED" : "FAILED") << "\n";
    return pass;
}

int main() {
    test_rmsnorm();
    return 0;
}
