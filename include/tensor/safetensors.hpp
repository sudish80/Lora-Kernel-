#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>

namespace lora_kernel {

class SafetensorsWriter {
private:
    struct TensorEntry {
        std::string          name;
        std::vector<int64_t> shape;
        std::vector<uint8_t> data;   // raw float bytes
    };
    std::vector<TensorEntry> tensors_;

public:
    void add(const std::string& name, const float* data,
             const std::vector<int64_t>& shape) {
        TensorEntry e;
        e.name  = name;
        e.shape = shape;
        size_t n = 1;
        for (auto d : shape) n *= static_cast<size_t>(d);
        e.data.resize(n * sizeof(float));
        std::memcpy(e.data.data(), data, e.data.size());
        tensors_.push_back(std::move(e));
    }

    bool write(const std::string& filename) {
        std::ostringstream json;
        json << "{";

        size_t data_offset = 0;
        for (size_t t = 0; t < tensors_.size(); ++t) {
            const auto& e = tensors_[t];
            size_t byte_len = e.data.size();
            size_t aligned_start = (data_offset + 63) & ~size_t(63);
            size_t end           = aligned_start + byte_len;

            if (t > 0) json << ",";
            json << "\"" << e.name << "\":{\"dtype\":\"F32\",\"shape\":[";
            for (size_t i = 0; i < e.shape.size(); ++i) {
                if (i > 0) json << ",";
                json << e.shape[i];
            }
            json << "],\"data_offsets\":[" << aligned_start << "," << end << "]}";

            data_offset = end;
        }
        json << "}";

        std::string header_str = json.str();
        while (header_str.size() % 8 != 0) header_str += ' ';

        uint64_t header_len = static_cast<uint64_t>(header_str.size());

        std::ofstream f(filename, std::ios::binary);
        if (!f) { std::cerr << "[ST] Cannot open " << filename << "\n"; return false; }

        f.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
        f.write(header_str.data(), header_str.size());

        size_t written = 0;
        for (const auto& e : tensors_) {
            size_t aligned = (written + 63) & ~size_t(63);
            if (aligned > written) {
                static const uint8_t zeros[64] = {};
                f.write(reinterpret_cast<const char*>(zeros), aligned - written);
            }
            f.write(reinterpret_cast<const char*>(e.data.data()), e.data.size());
            written = aligned + e.data.size();
        }

        std::cout << "[ST] Written: " << filename
                  << " | tensors=" << tensors_.size()
                  << " | header=" << header_len << " B\n";
        return true;
    }
};

class SafetensorsValidator {
public:
    bool validate_safetensors_file(const std::string& filename) const {
        std::ifstream f(filename, std::ios::binary);
        if (!f) { std::cerr << "[ST] Cannot open: " << filename << "\n"; return false; }

        uint64_t header_len = 0;
        f.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
        if (header_len == 0 || header_len > 100 * 1024 * 1024) {
            std::cerr << "[ST] Invalid header length: " << header_len << "\n";
            return false;
        }

        std::string header(header_len, '\0');
        f.read(header.data(), header_len);
        if (!f) { std::cerr << "[ST] Header read failed\n"; return false; }

        size_t s = header.find_first_not_of(' ');
        size_t e = header.find_last_not_of(' ');
        if (s == std::string::npos || header[s] != '{' || header[e] != '}') {
            std::cerr << "[ST] Header is not valid JSON object\n"; return false;
        }

        std::cout << "[ST] Validation passed: " << filename
                  << " (header=" << header_len << " B)\n";
        return true;
    }
};

} // namespace lora_kernel
