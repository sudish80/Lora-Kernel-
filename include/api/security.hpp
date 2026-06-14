#pragma once
#include <string>
#include <iostream>
#include <unordered_map>
#include <chrono>

namespace lora_kernel {

class Authentication {
private:
    std::unordered_map<std::string, std::string> api_keys;
public:
    void add_key(const std::string& user, const std::string& key) {
        api_keys[user] = key;
    }
    bool validate(const std::string& key) {
        for (auto& [u, k] : api_keys) if (k == key) return true;
        return false;
    }
};

class RateLimiter {
private:
    std::unordered_map<std::string, int> request_counts;
    int max_requests_;
    std::chrono::seconds window_;
public:
    RateLimiter(int max_rps = 100) : max_requests_(max_rps), window_(1) {}
    
    bool allow(const std::string& client_id) {
        int& count = request_counts[client_id];
        if (count >= max_requests_) return false;
        count++;
        return true;
    }
    
    void reset(const std::string& client_id) {
        request_counts[client_id] = 0;
    }
};

} // namespace lora_kernel
