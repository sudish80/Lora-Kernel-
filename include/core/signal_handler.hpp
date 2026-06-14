#pragma once
#include <atomic>
#include <csignal>
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#endif

namespace lora_kernel {

class SignalRecoveryHandler {
private:
    static std::atomic<bool> shutdown_flag;
    static std::atomic<int>  last_signal;

    static void handler(int sig) {
        last_signal.store(sig, std::memory_order_relaxed);
        shutdown_flag.store(true, std::memory_order_release);
    }

#ifdef _WIN32
    static BOOL WINAPI console_handler(DWORD dwCtrlType) {
        handler((int)dwCtrlType);
        return TRUE;
    }
#endif

public:
    static void install() {
#ifdef _WIN32
        SetConsoleCtrlHandler(console_handler, TRUE);
        signal(SIGINT, handler);
        signal(SIGTERM, handler);
        signal(SIGABRT, handler);
#else
        struct sigaction sa{};
        sa.sa_handler = handler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_RESETHAND;
        sigaction(SIGINT,  &sa, nullptr);
        sigaction(SIGTERM, &sa, nullptr);
        sigaction(SIGSEGV, &sa, nullptr);
#endif
    }

    static bool should_shutdown() {
        return shutdown_flag.load(std::memory_order_acquire);
    }
    static int  last_signal_received() {
        return last_signal.load(std::memory_order_relaxed);
    }
};

// Define static members here for simplicity in header-only, 
// or move to a .cpp if needed. Let's keep it header-only for now 
// if possible, but static members must be defined.
// Actually, header-only with static data members requires 
// them to be defined in one place (a .cpp file).

} // namespace lora_kernel
