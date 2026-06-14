#pragma once
#include <string>
#include <iostream>

namespace lora_kernel {

class ExportSuite {
public:
    void export_gguf(const std::string& path) { std::cout << "[EXPORT] GGUF: " << path << "\n"; }
    void export_onnx(const std::string& path) { std::cout << "[EXPORT] ONNX: " << path << "\n"; }
    void export_trt(const std::string& path)  { std::cout << "[EXPORT] TensorRT: " << path << "\n"; }
    void export_ov(const std::string& path)   { std::cout << "[EXPORT] OpenVINO: " << path << "\n"; }
};

} // namespace lora_kernel
