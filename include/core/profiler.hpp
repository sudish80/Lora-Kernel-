#pragma once
#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <mutex>
#include <atomic>
#include <sstream>

namespace lora_kernel {

// Production-level hard-coded profiling system with flamegraph output

struct ProfileEvent {
    std::string name;
    std::string category;
    int64_t start_ns;
    int64_t end_ns;
    int thread_id;
};

class Profiler {
private:
    std::vector<ProfileEvent> events_;
    std::mutex mtx_;
    std::atomic<bool> enabled_{false};
    std::string output_path_;

    int64_t now_ns() {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   now.time_since_epoch()).count();
    }

public:
    Profiler(const std::string& path = "profile.json") : output_path_(path) {}

    void enable() { enabled_ = true; }
    void disable() { enabled_ = false; }

    struct ScopedTimer {
        Profiler* profiler;
        std::string name;
        std::string category;
        int64_t start;
        int thread_id;

        ScopedTimer(Profiler* p, const std::string& n, const std::string& cat)
            : profiler(p), name(n), category(cat) {
            if (p && p->enabled_) {
                start = p->now_ns();
                thread_id = 0;
            }
        }

        ~ScopedTimer() {
            if (profiler && profiler->enabled_) {
                int64_t end = profiler->now_ns();
                std::lock_guard<std::mutex> lock(profiler->mtx_);
                profiler->events_.push_back({name, category, start, end, thread_id});
            }
        }
    };

    ScopedTimer scope(const std::string& name, const std::string& cat = "default") {
        return ScopedTimer(this, name, cat);
    }

    // Flush to Chrome trace format (flamegraph-compatible)
    void flush() {
        std::lock_guard<std::mutex> lock(mtx_);

        std::ofstream f(output_path_);
        f << "{\"traceEvents\":[\n";

        for (size_t i = 0; i < events_.size(); ++i) {
            auto& e = events_[i];
            f << "  {\"cat\":\"" << e.category
              << "\",\"dur\":" << (e.end_ns - e.start_ns)
              << ",\"name\":\"" << e.name
              << "\",\"ph\":\"X\",\"pid\":1,\"tid\":" << e.thread_id
              << ",\"ts\":" << e.start_ns << "}";
            if (i < events_.size() - 1) f << ",";
            f << "\n";
        }

        f << "]}\n";
        std::cout << "[PROFILE] Wrote " << events_.size()
                  << " events to " << output_path_ << "\n";
        events_.clear();
    }

    // Report top 10 slowest operations
    void report() {
        std::lock_guard<std::mutex> lock(mtx_);

        // Aggregate by name
        std::unordered_map<std::string, int64_t> total_time;
        std::unordered_map<std::string, int> count;

        for (auto& e : events_) {
            total_time[e.name] += (e.end_ns - e.start_ns);
            count[e.name]++;
        }

        // Sort by total time
        std::vector<std::pair<std::string, int64_t>> sorted;
        for (auto& [name, time] : total_time)
            sorted.push_back({name, time});
        std::sort(sorted.begin(), sorted.end(),
                  [](auto& a, auto& b) { return a.second > b.second; });

        std::cout << "\n=== Profile Report (Top 10) ===\n";
        std::cout << std::left << std::setw(30) << "Operation"
                  << std::setw(12) << "Calls"
                  << std::setw(12) << "Total(ms)"
                  << std::setw(12) << "Avg(ms)" << "\n";
        std::cout << std::string(66, '-') << "\n";

        for (int i = 0; i < std::min(10, (int)sorted.size()); ++i) {
            auto& [name, total] = sorted[i];
            int c = count[name];
            double total_ms = total / 1e6;
            double avg_ms = total_ms / c;
            std::cout << std::left << std::setw(30) << name
                      << std::right << std::setw(12) << c
                      << std::setw(12) << std::fixed << std::setprecision(2) << total_ms
                      << std::setw(12) << std::fixed << std::setprecision(2) << avg_ms << "\n";
        }
        std::cout << "\n";
    }

    void set_output(const std::string& path) { output_path_ = path; }
};

// Global profiler instance
inline Profiler& get_profiler() {
    static Profiler profiler;
    return profiler;
}

// Convenience macros
#define PROFILE_SCOPE(name) \
    auto CONCAT_VAR(prof_scope_, __LINE__) = get_profiler().scope(name)

#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__)

} // namespace lora_kernel
