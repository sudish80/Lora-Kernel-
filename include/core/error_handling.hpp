#pragma once
#include <stdexcept>
#include <string>
#include <sstream>
#include <iostream>
#include <atomic>
#include <functional>

#ifdef __linux__
#include <execinfo.h>
#include <cxxabi.h>
#endif

namespace lora_kernel {

// Production-level error handling with stack traces and recovery

// Exception hierarchy
class LoraError : public std::runtime_error {
public:
    LoraError(const std::string& msg) : std::runtime_error(msg) {}
};

class ShapeError : public LoraError {
public:
    ShapeError(const std::string& msg) : LoraError("ShapeError: " + msg) {}
};

class NanError : public LoraError {
public:
    NanError(const std::string& msg) : LoraError("NanError: " + msg) {}
};

class CudaError : public LoraError {
public:
    int cuda_code;
    CudaError(const std::string& msg, int code = 0)
        : LoraError("CudaError[" + std::to_string(code) + "]: " + msg), cuda_code(code) {}
};

class OutOfMemoryError : public LoraError {
public:
    size_t requested_bytes;
    OutOfMemoryError(size_t bytes)
        : LoraError("OOM: requested " + std::to_string(bytes / 1024 / 1024) + " MB"),
          requested_bytes(bytes) {}
};

// Stack trace capture
inline std::string capture_stacktrace(int skip = 0) {
#ifdef __linux__
    void* buffer[64];
    int n = backtrace(buffer, 64);
    char** symbols = backtrace_symbols(buffer, n);
    std::ostringstream oss;
    for (int i = skip; i < n; ++i) {
        std::string sym(symbols[i]);
        // Demangle if possible
        size_t start = sym.find('(');
        size_t end = sym.find('+', start);
        if (start != std::string::npos && end != std::string::npos) {
            std::string mangled = sym.substr(start + 1, end - start - 1);
            int status;
            char* demangled = abi::__cxa_demangle(mangled.c_str(), nullptr, nullptr, &status);
            if (demangled) {
                sym = sym.substr(0, start + 1) + demangled + sym.substr(end);
                std::free(demangled);
            }
        }
        oss << "  #" << (i - skip) << " " << sym << "\n";
    }
    std::free(symbols);
    return oss.str();
#else
    return "  (stack trace unavailable on this platform)\n";
#endif
}

// Recovery mechanism
class RecoveryManager {
private:
    static std::atomic<int> error_count_;
    static std::atomic<bool> fatal_state_;
    static std::function<void()> emergency_handler_;

public:
    // Register handler for catastrophic errors
    static void set_emergency_handler(std::function<void()> handler) {
        emergency_handler_ = handler;
    }

    // Called when an error is caught
    static void handle_error(const std::string& context, bool is_fatal = false) {
        error_count_++;
        std::cerr << "[ERROR] " << context << "\n";
        std::cerr << capture_stacktrace(1);

        if (is_fatal || error_count_ > 10) {
            fatal_state_ = true;
            std::cerr << "[FATAL] Too many errors. Entering safe mode.\n";
            if (emergency_handler_) emergency_handler_();
        }
    }

    // Check if system is in a recoverable state
    static bool is_recoverable() { return !fatal_state_; }
    static int error_count() { return error_count_; }
    static void reset() { error_count_ = 0; fatal_state_ = false; }
};

inline std::atomic<int> RecoveryManager::error_count_{0};
inline std::atomic<bool> RecoveryManager::fatal_state_{false};
inline std::function<void()> RecoveryManager::emergency_handler_ = nullptr;

// Safe tensor operations with validation
inline void validate_tensor_shape(const std::vector<int64_t>& shape,
                                   const std::string& name) {
    if (shape.empty())
        throw ShapeError(name + " has empty shape");
    for (auto s : shape)
        if (s <= 0)
            throw ShapeError(name + " has invalid dimension " + std::to_string(s));
}

// NaN/Inf detection
inline bool has_nan_or_inf(const float* data, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        if (std::isnan(data[i]) || std::isinf(data[i])) return true;
    }
    return false;
}

inline void check_nan_inf(const float* data, int64_t n, const std::string& name) {
    if (has_nan_or_inf(data, n)) {
        std::cerr << "[NAN] " << name << " contains NaN/Inf at " << n << " elements\n";
        throw NanError(name + " contains NaN/Inf");
    }
}

} // namespace lora_kernel
