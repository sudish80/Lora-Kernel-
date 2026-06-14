#pragma once
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <mutex>
#include <atomic>

// windows.h #define ERROR 0 conflicts with our enum value
#ifdef ERROR
#undef ERROR
#endif

namespace lora_kernel {

enum class LogLevel {
    TRACE = 0,
    DEBUG = 1,
    INFO = 2,
    WARN = 3,
    ERROR = 4,
    FATAL = 5
};

class Logger {
private:
    std::string name_;
    LogLevel min_level_{LogLevel::INFO};
    std::ofstream file_;
    std::mutex mtx_;
    static std::atomic<int> log_counter_;

    std::string level_to_string(LogLevel lvl) {
        switch (lvl) {
            case LogLevel::TRACE: return "TRACE";
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO:  return "INFO";
            case LogLevel::WARN:  return "WARN";
            case LogLevel::ERROR: return "ERROR";
            case LogLevel::FATAL: return "FATAL";
            default: return "UNKN";
        }
    }

    std::string current_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now.time_since_epoch()) % 1000;
        auto t = std::chrono::system_clock::to_time_t(now);
        std::tm tm;
#ifdef _WIN32
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S")
            << "." << std::setfill('0') << std::setw(3) << ms.count();
        return oss.str();
    }

public:
    Logger(const std::string& name) : name_(name) {}

    void set_level(LogLevel lvl) { min_level_ = lvl; }
    void set_log_file(const std::string& path) {
        file_.open(path, std::ios::app);
    }
    void set_min_level(LogLevel level) { min_level_ = level; }

    // Structured JSON output
    void log(LogLevel lvl, const std::string& file, int line,
             const std::string& message) {
        if (lvl < min_level_) return;

        std::lock_guard<std::mutex> lock(mtx_);
        int id = log_counter_.fetch_add(1, std::memory_order_relaxed);

        std::ostringstream oss;
        oss << "{\"ts\":\"" << current_timestamp()
            << "\",\"level\":\"" << level_to_string(lvl)
            << "\",\"logger\":\"" << name_
            << "\",\"file\":\"" << file << ":" << line
            << "\",\"msg\":\"" << message
            << "\",\"id\":" << id << "}";

        std::string output = oss.str();

        // Always print to stderr for ERROR and FATAL
        if (lvl >= LogLevel::ERROR) {
            std::cerr << output << std::endl;
        } else {
            std::cout << output << std::endl;
        }

        if (file_.is_open()) {
            file_ << output << std::endl;
        }
    }
};

std::atomic<int> Logger::log_counter_{0};

// Convenience macros
#define LOG_TRACE(logger, msg)  logger.log(LogLevel::TRACE, __FILE__, __LINE__, msg)
#define LOG_DEBUG(logger, msg)  logger.log(LogLevel::DEBUG, __FILE__, __LINE__, msg)
#define LOG_INFO(logger, msg)   logger.log(LogLevel::INFO,  __FILE__, __LINE__, msg)
#define LOG_WARN(logger, msg)   logger.log(LogLevel::WARN,  __FILE__, __LINE__, msg)
#define LOG_ERROR(logger, msg)  logger.log(LogLevel::ERROR, __FILE__, __LINE__, msg)
#define LOG_FATAL(logger, msg)  logger.log(LogLevel::FATAL, __FILE__, __LINE__, msg)

// Global logger instance
inline Logger& get_logger() {
    static Logger logger("lora_kernel");
    return logger;
}

} // namespace lora_kernel
