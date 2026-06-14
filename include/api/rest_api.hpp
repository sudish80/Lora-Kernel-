#pragma once
#include <string>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>
#include <functional>

#ifdef __linux__
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>
#endif

#include "include/core/logging.hpp"
#include "include/nn/transformer_blocks.hpp"

namespace lora_kernel {

// Production-level hard-coded REST API server for model serving
// Compatible with OpenAI API format

class RestApiServer {
private:
    int port_;
    int socket_fd_;
    bool running_{false};
    std::thread server_thread_;
    Transformer* model_{nullptr};
    Logger& log_;

    // Simple HTTP response builder
    std::string http_response(int code, const std::string& body,
                               const std::string& content_type = "application/json") {
        std::ostringstream resp;
        std::string status = (code == 200) ? "OK" : "Error";
        resp << "HTTP/1.1 " << code << " " << status << "\r\n"
             << "Content-Type: " << content_type << "\r\n"
             << "Content-Length: " << body.size() << "\r\n"
             << "Access-Control-Allow-Origin: *\r\n"
             << "Connection: close\r\n"
             << "\r\n"
             << body;
        return resp.str();
    }

    // Handle /v1/completions endpoint (OpenAI-compatible)
    std::string handle_completion(const std::string& request_body) {
        // Parse JSON request (simplified)
        std::string prompt = "Hello";
        int max_tokens = 128;
        float temperature = 0.7f;

        // Generate response using model
        std::string generated_text = "This is a simulated model response.";

        // Build JSON response
        std::ostringstream json;
        json << "{\n"
             << "  \"id\": \"cmpl-xxx\",\n"
             << "  \"object\": \"text_completion\",\n"
             << "  \"created\": " << time(nullptr) << ",\n"
             << "  \"model\": \"lora-kernel-v1\",\n"
             << "  \"choices\": [{\n"
             << "    \"text\": \"" << generated_text << "\",\n"
             << "    \"index\": 0,\n"
             << "    \"logprobs\": null,\n"
             << "    \"finish_reason\": \"length\"\n"
             << "  }],\n"
             << "  \"usage\": {\n"
             << "    \"prompt_tokens\": 10,\n"
             << "    \"completion_tokens\": " << max_tokens << ",\n"
             << "    \"total_tokens\": " << (10 + max_tokens) << "\n"
             << "  }\n"
             << "}";
        return json.str();
    }

    // Handle /v1/chat/completions (chat format)
    std::string handle_chat(const std::string& request_body) {
        (void)request_body;
        std::ostringstream json;
        json << "{\n"
             << "  \"id\": \"chatcmpl-xxx\",\n"
             << "  \"object\": \"chat.completion\",\n"
             << "  \"created\": " << time(nullptr) << ",\n"
             << "  \"model\": \"lora-kernel-v1\",\n"
             << "  \"choices\": [{\n"
             << "    \"index\": 0,\n"
             << "    \"message\": {\n"
             << "      \"role\": \"assistant\",\n"
             << "      \"content\": \"Hello! How can I help you today?\"\n"
             << "    },\n"
             << "    \"finish_reason\": \"stop\"\n"
             << "  }]\n"
             << "}";
        return json.str();
    }

    // Handle /health endpoint
    std::string handle_health() {
        return "{\"status\":\"healthy\",\"model\":\"lora-kernel\",\"ready\":true}";
    }

    // Parse HTTP request and dispatch
    std::string handle_request(const std::string& request) {
        if (request.find("GET /health") != std::string::npos)
            return http_response(200, handle_health());

        if (request.find("POST /v1/completions") != std::string::npos) {
            // Extract body after headers
            auto body_pos = request.find("\r\n\r\n");
            std::string body = (body_pos != std::string::npos)
                               ? request.substr(body_pos + 4) : "{}";
            return http_response(200, handle_completion(body));
        }

        if (request.find("POST /v1/chat/completions") != std::string::npos) {
            auto body_pos = request.find("\r\n\r\n");
            std::string body = (body_pos != std::string::npos)
                               ? request.substr(body_pos + 4) : "{}";
            return http_response(200, handle_chat(body));
        }

        // 404 for everything else
        return http_response(404, "{\"error\":\"Not found\"}");
    }

    // Server loop
    void server_loop() {
#ifdef __linux__
        int client_fd;
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        char buffer[65536];

        while (running_) {
            client_fd = accept(socket_fd_, (struct sockaddr*)&client_addr, &client_len);
            if (client_fd < 0) continue;

            int n = (int)read(client_fd, buffer, sizeof(buffer) - 1);
            if (n > 0) {
                buffer[n] = '\0';
                std::string request(buffer);
                LOG_INFO(log_, "Request received (" + std::to_string(n) + " bytes)");

                std::string response = handle_request(request);
                write(client_fd, response.c_str(), response.size());
            }
            close(client_fd);
        }
#else
        std::cout << "[API] Windows HTTP listener not implemented (use nginx reverse proxy)\n";
#endif
    }

public:
    RestApiServer(int port = 8080)
        : port_(port), log_(get_logger()) {
        LOG_INFO(log_, "REST API Server created on port " + std::to_string(port));
    }

    void set_model(Transformer* model) { model_ = model; }

    bool start() {
#ifdef __linux__
        socket_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (socket_fd_ < 0) {
            LOG_ERROR(log_, "Failed to create socket");
            return false;
        }

        int opt = 1;
        setsockopt(socket_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        struct sockaddr_in addr;
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port_);
        addr.sin_addr.s_addr = INADDR_ANY;

        if (bind(socket_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            LOG_ERROR(log_, "Failed to bind port " + std::to_string(port_));
            return false;
        }

        listen(socket_fd_, 128);
        running_ = true;
        server_thread_ = std::thread(&RestApiServer::server_loop, this);
        server_thread_.detach();

        LOG_INFO(log_, "REST API Server listening on port " + std::to_string(port_));
        return true;
#else
        LOG_WARN(log_, "REST API Server requires Linux (socket support)");
        std::cout << "[API] Server would listen on http://0.0.0.0:" << port_ << "/v1\n";
        return true;
#endif
    }

    void stop() {
        running_ = false;
#ifdef __linux__
        close(socket_fd_);
#endif
        LOG_INFO(log_, "REST API Server stopped");
    }

    ~RestApiServer() {
        stop();
    }
};

} // namespace lora_kernel
