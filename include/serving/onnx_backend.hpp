#pragma once
#include <string>
#include <vector>
#include <iostream>
#include "include/core/tensor.hpp"

#ifdef ONNX_RUNTIME_FOUND
#include <onnxruntime_cxx_api.h>
#endif

namespace lora_kernel {

class OnnxBackend {
private:
#ifdef ONNX_RUNTIME_FOUND
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "onnx-backend"};
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_{nullptr};
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
#else
    bool initialized_{false};
#endif
    bool loaded_{false};

public:
    OnnxBackend() {
#ifndef ONNX_RUNTIME_FOUND
        std::cerr << "[ONNX] ONNX Runtime not available at compile time\n";
#endif
    }

    bool load_model(const std::string& path) {
#ifdef ONNX_RUNTIME_FOUND
        try {
            session_ = std::make_unique<Ort::Session>(env_, path.c_str(), session_options_);
            memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            Ort::AllocatorWithDefaultOptions allocator;
            size_t num_inputs = session_->GetInputCount();
            size_t num_outputs = session_->GetOutputCount();
            input_names_.resize(num_inputs);
            output_names_.resize(num_outputs);
            for (size_t i = 0; i < num_inputs; ++i)
                input_names_[i] = session_->GetInputNameAllocated(i, allocator).release();
            for (size_t i = 0; i < num_outputs; ++i)
                output_names_[i] = session_->GetOutputNameAllocated(i, allocator).release();

            loaded_ = true;
            std::cout << "[ONNX] Loaded model: " << path << "\n";
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[ONNX] Failed to load model: " << e.what() << "\n";
            return false;
        }
#else
        (void)path;
        std::cerr << "[ONNX] ONNX Runtime not available\n";
        return false;
#endif
    }

    bool run_inference(const Tensor& input, Tensor& output) {
#ifdef ONNX_RUNTIME_FOUND
        if (!loaded_ || !session_) return false;

        std::vector<int64_t> input_shape = input.shape();
        size_t total_elements = 1;
        for (auto s : input_shape) total_elements *= s;

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, const_cast<float*>(input.data()),
            total_elements, input_shape.data(), input_shape.size());

        std::vector<Ort::Value> ort_outputs = session_->Run(
            Ort::RunOptions{},
            input_names_.data(), &input_tensor, input_names_.size(),
            output_names_.data(), output_names_.size());

        if (ort_outputs.empty()) return false;

        float* output_data = ort_outputs[0].GetTensorMutableData<float>();
        auto type_info = ort_outputs[0].GetTensorTypeAndShapeInfo();
        std::vector<int64_t> out_shape = type_info.GetShape();

        if (output.numel() == 0) {
            // output tensor not pre-allocated; this path expects it to be
            return false;
        }
        size_t out_count = type_info.GetElementCount();
        size_t copy_size = std::min(out_count, (size_t)output.numel());
        std::memcpy(output.data(), output_data, copy_size * sizeof(float));
        return true;
#else
        (void)input;
        (void)output;
        std::cerr << "[ONNX] ONNX Runtime not available\n";
        return false;
#endif
    }

    bool is_loaded() const { return loaded_; }
};

} // namespace lora_kernel
