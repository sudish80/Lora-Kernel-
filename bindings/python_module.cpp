#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "include/core/tensor.hpp"
#include "include/nn/transformer_blocks.hpp"
#include "include/nn/training_pipeline.hpp"

namespace py = pybind11;
using namespace lora_kernel;

PYBIND11_MODULE(lora_kernel, m) {
    m.doc() = "LoRA Kernel - LLM Training and Inference Engine";

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const std::vector<int64_t>&>())
        .def("shape", &Tensor::shape)
        .def("numel", &Tensor::numel)
        .def("__getitem__", [](Tensor& t, int i) { return t[i]; })
        .def("__setitem__", [](Tensor& t, int i, float v) { t[i] = v; });

    py::class_<TransformerConfig>(m, "TransformerConfig")
        .def(py::init<>())
        .def_readwrite("hidden_dim", &TransformerConfig::hidden_dim)
        .def_readwrite("num_heads", &TransformerConfig::num_heads)
        .def_readwrite("head_dim", &TransformerConfig::head_dim)
        .def_readwrite("vocab_size", &TransformerConfig::vocab_size)
        .def_readwrite("max_seq_len", &TransformerConfig::max_seq_len)
        .def_readwrite("num_layers", &TransformerConfig::num_layers)
        .def_readwrite("dropout", &TransformerConfig::dropout)
        .def_readwrite("ff_dim", &TransformerConfig::ff_dim);

    py::class_<Transformer>(m, "Transformer")
        .def(py::init<const TransformerConfig&>())
        .def("forward", [](Transformer& m, py::array_t<int> ids) {
            auto buf = ids.request();
            Tensor input({buf.shape[0], buf.shape[1]});
            std::memcpy(input.data(), buf.ptr, input.numel() * sizeof(float));
            Tensor logits({buf.shape[0], buf.shape[1], 50257});
            m.forward(input, logits);
            return logits;
        })
        .def("num_parameters", &Transformer::num_parameters);

    py::class_<TrainingPipeline>(m, "TrainingPipeline")
        .def(py::init<Transformer&, float, float, float>())
        .def("train_step", [](TrainingPipeline& p, py::array_t<int> ids, py::array_t<int> tgt) {
            auto buf_ids = ids.request();
            auto buf_tgt = tgt.request();
            Tensor inp({buf_ids.shape[0], buf_ids.shape[1]});
            Tensor tar({buf_tgt.shape[0], buf_tgt.shape[1]});
            std::memcpy(inp.data(), buf_ids.ptr, inp.numel() * sizeof(float));
            std::memcpy(tar.data(), buf_tgt.ptr, tar.numel() * sizeof(float));
            return p.train_step(inp, tar);
        })
        .def("set_learning_rate", &TrainingPipeline::set_learning_rate)
        .def("current_step", &TrainingPipeline::current_step);
}
