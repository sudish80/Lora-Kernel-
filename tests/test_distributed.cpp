#include <iostream>
#include "include/distributed/nccl_wrapper.hpp"
#include "include/distributed/zero_optimizer.hpp"

bool test_distributed() {
    lora_kernel::NCCLWrapper nccl(0, 1);
    lora_kernel::MPIWrapper mpi;
    lora_kernel::ZeROOptimizer zero(2, &nccl, 100);

    std::cout << "[TEST] Distributed subsystems initialized\n";
    std::cout << "[TEST] ZeRO stage 2 initialized\n";
    return true;
}

int main() {
    test_distributed();
    return 0;
}
