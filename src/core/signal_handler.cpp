#include "include/core/signal_handler.hpp"

namespace lora_kernel {

std::atomic<bool> SignalRecoveryHandler::shutdown_flag{false};
std::atomic<int>  SignalRecoveryHandler::last_signal{0};

} // namespace lora_kernel
