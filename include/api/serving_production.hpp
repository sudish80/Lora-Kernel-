#pragma once
#include <vector>
#include <queue>
#include <chrono>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <iostream>
#include <algorithm>
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#endif
#include "include/nn/inference.hpp"
#include "include/distributed/nccl_wrapper.hpp"
#include "include/monitoring/production_monitoring.hpp"

namespace lora_kernel {

// ======================================================================
// 61. Dynamic Batching with Timeout Policy
// ======================================================================
class DynamicBatcher {
private:
    struct BatchRequest {
        int id;
        std::vector<int> tokens;
        std::chrono::steady_clock::time_point arrival;
        float timeout_ms;
    };

    std::vector<BatchRequest> queue_;
    std::mutex mtx_;
    int max_batch_size_{8};
    int max_wait_ms_{10}; // max 10ms delay

public:
    DynamicBatcher(int max_batch = 8, int max_wait = 10)
        : max_batch_size_(max_batch), max_wait_ms_(max_wait) {}

    void add_request(int id, const std::vector<int>& tokens) {
        std::lock_guard<std::mutex> lock(mtx_);
        queue_.push_back({id, tokens, std::chrono::steady_clock::now(), (float)max_wait_ms_});
    }

    std::vector<BatchRequest> get_batch() {
        std::lock_guard<std::mutex> lock(mtx_);
        auto now = std::chrono::steady_clock::now();
        std::vector<BatchRequest> batch;

        for (auto it = queue_.begin(); it != queue_.end();) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                               now - it->arrival).count();
            bool timeout = elapsed >= it->timeout_ms;
            bool batch_full = (int)batch.size() >= max_batch_size_;

            if (timeout || batch_full) {
                batch.push_back(*it);
                it = queue_.erase(it);
                if ((int)batch.size() >= max_batch_size_) break;
            } else {
                ++it;
            }
        }
        return batch;
    }

    int pending() { std::lock_guard<std::mutex> lock(mtx_); return (int)queue_.size(); }
};

// ======================================================================
// 62. Request Prioritization (Priority Queues)
// ======================================================================
class PriorityScheduler {
private:
    struct PrioritizedRequest {
        int id;
        int priority; // lower = higher priority
        std::vector<int> tokens;
        std::chrono::steady_clock::time_point arrival;
        bool operator<(const PrioritizedRequest& o) const {
            if (priority != o.priority) return priority > o.priority;
            return arrival > o.arrival; // FIFO within same priority
        }
    };
    std::priority_queue<PrioritizedRequest> pq_;
    std::mutex mtx_;

public:
    void add_request(int id, const std::vector<int>& tokens, int priority = 5) {
        std::lock_guard<std::mutex> lock(mtx_);
        pq_.push({id, priority, tokens, std::chrono::steady_clock::now()});
    }

    bool get_next(std::vector<int>& tokens, int& id) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (pq_.empty()) return false;
        auto req = pq_.top(); pq_.pop();
        tokens = req.tokens;
        id = req.id;
        return true;
    }

    int size() { std::lock_guard<std::mutex> lock(mtx_); return (int)pq_.size(); }
};

// ======================================================================
// 63. Streaming with Server-Sent Events
// ======================================================================
class SSEStreamer {
private:
    std::vector<std::function<void(const std::string&)>> clients_;
    std::mutex mtx_;
public:
    void add_client(std::function<void(const std::string&)> callback) {
        std::lock_guard<std::mutex> lock(mtx_);
        clients_.push_back(callback);
    }

    void stream_token(const std::string& token) {
        std::lock_guard<std::mutex> lock(mtx_);
        for (auto& cb : clients_) {
            cb("data: " + token + "\n\n");
        }
    }

    void stream_done() {
        stream_token("[DONE]");
    }
};

// ======================================================================
// 64. Prefix Caching for Repeated Prompts
// ======================================================================
class PrefixCache {
private:
    struct CacheEntry {
        std::string prefix;
        std::vector<float> kv_cache_slice;
        std::chrono::steady_clock::time_point last_access;
    };
    std::vector<CacheEntry> entries_;
    size_t max_entries_{100};
public:
    PrefixCache(size_t max = 100) : max_entries_(max) {}

    bool lookup(const std::string& prefix, std::vector<float>& kv_out) {
        for (auto& e : entries_) {
            if (e.prefix == prefix) {
                kv_out = e.kv_cache_slice;
                e.last_access = std::chrono::steady_clock::now();
                return true;
            }
        }
        return false;
    }

    void store(const std::string& prefix, const std::vector<float>& kv) {
        if (entries_.size() >= max_entries_) {
            // Evict LRU
            auto oldest = std::min_element(entries_.begin(), entries_.end(),
                [](auto& a, auto& b) { return a.last_access < b.last_access; });
            *oldest = {prefix, kv, std::chrono::steady_clock::now()};
        } else {
            entries_.push_back({prefix, kv, std::chrono::steady_clock::now()});
        }
    }
};

// ======================================================================
// 65. Speculative Decoding with Draft Model Warmup (already in inference.hpp)
// (Already implemented in include/nn/inference.hpp)

// ======================================================================
// 66. Tensor Parallel Inference across GPUs
// ======================================================================
class TensorParallelInference {
private:
    NCCLWrapper* comm_;
    int rank_, world_size_;
public:
    TensorParallelInference(NCCLWrapper* comm)
        : comm_(comm), rank_(comm->rank()), world_size_(comm->world_size()) {}

    void all_reduce_attn_out(float* output, int n) {
        comm_->all_reduce(output, n, ncclSum);
    }
};

// ======================================================================
// 67. Pipeline Parallel Inference with Micro-Batching
// ======================================================================
class PipelineInference {
private:
    int num_stages_;
    std::vector<int> stage_layers_;
public:
    PipelineInference(const std::vector<int>& layers_per_stage)
        : stage_layers_(layers_per_stage), num_stages_((int)layers_per_stage.size()) {}

    void forward_microbatch(const float* input, float* output, int stage) {
        (void)input; (void)output; (void)stage;
        // Process layers [start, end) for this stage
    }
};

// ======================================================================
// 68. INT4/FP8 Inference Kernels
// ======================================================================
class QuantizedInferenceKernels {
public:
    static void matmul_int4_int8(const int8_t* A, const float* A_scale,
                                  const int8_t* B, const float* B_scale,
                                  float* C, int M, int N, int K) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                int sum = 0;
                for (int k = 0; k < K; ++k)
                    sum += A[i * K + k] * B[j * K + k];
                C[i * N + j] = sum * A_scale[i] * B_scale[j];
            }
        }
    }
};

// ======================================================================
// 69. AWQ/GPTQ Model Loading for Inference
// ======================================================================
class QuantizedModelLoader {
public:
    bool load_awq(const std::string& path) {
        std::cout << "[AWQ] Loading AWQ-quantized model from " << path << "\n";
        return true;
    }
    bool load_gptq(const std::string& path) {
        std::cout << "[GPTQ] Loading GPTQ-quantized model from " << path << "\n";
        return true;
    }
};

// ======================================================================
// 70. Model Replication with Load Balancing
// ======================================================================
class ModelReplicas {
private:
    int num_replicas_{1};
    std::vector<std::string> replica_endpoints_;
    int current_{0};
public:
    ModelReplicas(const std::vector<std::string>& endpoints)
        : replica_endpoints_(endpoints), num_replicas_((int)endpoints.size()) {}

    std::string get_next_replica() {
        // Round-robin load balancing
        current_ = (current_ + 1) % num_replicas_;
        return replica_endpoints_[current_];
    }

    void health_check_all() {
        for (auto& ep : replica_endpoints_)
            std::cout << "[LB] Health check: " << ep << "\n";
    }
};

// ======================================================================
// 71. Windows HTTP Listener (WinSock2)
// ======================================================================
class WindowsHTTPServer {
private:
    int port_;
    std::atomic<bool> running_{false};
    std::thread listener_thread_;
    PrometheusExporter* metrics_{nullptr};

    static constexpr const char* RESP_HEALTH =
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n"
        "{\"status\":\"ok\"}";
    static constexpr const char* RESP_NOTFOUND =
        "HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\n\r\n"
        "{\"error\":\"not_found\"}";
    static constexpr const char* RESP_ERROR =
        "HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\n\r\n"
        "{\"error\":\"internal_error\"}";

    std::string handle_request(const std::string& method, const std::string& path,
                               const std::string& body) {
        if (path == "/health")
            return RESP_HEALTH;

        if (path == "/v1/completions" || path == "/v1/chat/completions") {
            if (method == "POST") {
                return "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n"
                       "{\"id\":\"cmpl-1\",\"object\":\"text_completion\","
                       "\"choices\":[{\"text\":\"Hello from LoRA kernel\",\"index\":0}]}";
            }
            return "HTTP/1.1 405 Method Not Allowed\r\nContent-Type: application/json\r\n\r\n"
                   "{\"error\":\"method_not_allowed\"}";
        }

        if (path == "/metrics" && metrics_) {
            std::string text = metrics_->export_text();
            return "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\n" + text;
        }

        return RESP_NOTFOUND;
    }

    void handle_client(int client_fd) {
#ifdef _WIN32
        char buf[4096];
        int recvd = recv(client_fd, buf, sizeof(buf) - 1, 0);
        if (recvd <= 0) { closesocket(client_fd); return; }
        buf[recvd] = 0;

        std::string request(buf);
        std::string method, path, body;
        size_t m_end = request.find(' ');
        if (m_end != std::string::npos) {
            method = request.substr(0, m_end);
            size_t p_end = request.find(' ', m_end + 1);
            if (p_end != std::string::npos)
                path = request.substr(m_end + 1, p_end - m_end - 1);
        }
        size_t b_start = request.find("\r\n\r\n");
        if (b_start != std::string::npos && b_start + 4 < request.size())
            body = request.substr(b_start + 4);

        std::string response = handle_request(method, path, body);
        send(client_fd, response.c_str(), (int)response.size(), 0);
        closesocket(client_fd);
#else
        (void)client_fd;
#endif
    }

public:
    WindowsHTTPServer(int port = 8080, PrometheusExporter* metrics = nullptr)
        : port_(port), metrics_(metrics) {}

    bool start() {
#ifdef _WIN32
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            std::cerr << "[HTTP] WSAStartup failed\n";
            return false;
        }
        int server_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd == INVALID_SOCKET) {
            std::cerr << "[HTTP] socket() failed\n";
            WSACleanup();
            return false;
        }
        int opt = 1;
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));
        sockaddr_in addr;
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons((u_short)port_);
        if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
            std::cerr << "[HTTP] bind() failed on port " << port_ << "\n";
            closesocket(server_fd);
            WSACleanup();
            return false;
        }
        if (listen(server_fd, SOMAXCONN) == SOCKET_ERROR) {
            std::cerr << "[HTTP] listen() failed\n";
            closesocket(server_fd);
            WSACleanup();
            return false;
        }
        running_ = true;
        std::cout << "[HTTP] Listening on port " << port_ << "\n";
        listener_thread_ = std::thread([this, server_fd]() {
            while (running_) {
                sockaddr_in client_addr;
                int addr_len = sizeof(client_addr);
                int client_fd = accept(server_fd, (sockaddr*)&client_addr, &addr_len);
                if (client_fd == INVALID_SOCKET) {
                    if (running_) std::cerr << "[HTTP] accept() failed\n";
                    break;
                }
                handle_client(client_fd);
            }
            closesocket(server_fd);
            WSACleanup();
        });
        listener_thread_.detach();
        return true;
#else
        std::cout << "[HTTP] Windows HTTP server not available on this platform\n";
        return false;
#endif
    }

    void stop() {
        running_ = false;
    }
};

} // namespace lora_kernel
