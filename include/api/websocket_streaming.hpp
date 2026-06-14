#pragma once
#include <string>
#include <vector>
#include <queue>
#include <iostream>

namespace lora_kernel {

class WebSocketStreamer {
private:
    std::queue<std::string> message_queue;
    bool connected_{false};
public:
    void connect(const std::string& endpoint) {
        connected_ = true;
        std::cout << "[WS] Connected to " << endpoint << "\n";
    }
    
    void send_chunk(const std::string& token) {
        message_queue.push(token);
    }
    
    std::string receive() {
        if (message_queue.empty()) return "";
        auto msg = message_queue.front();
        message_queue.pop();
        return msg;
    }
    
    void broadcast(const std::string& message) {
        std::cout << "[WS] Broadcasting: " << message.substr(0, 50) << "...\n";
    }
};

} // namespace lora_kernel
