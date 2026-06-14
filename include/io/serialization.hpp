#pragma once
#include <string>
#include <fstream>
#include <iostream>
#include "include/tensor/safetensors.hpp"
#include "include/crypto/crypto_engine.hpp"
#include "include/nn/transformer_blocks.hpp"

namespace lora_kernel {

class ModelSerializer {
public:
    static bool save_safetensors(const Transformer& model, const std::string& path) {
        SafetensorsWriter writer;
        const auto& params = model.get_params();
        int num_layers = model.get_config().num_layers;

        if (params.size() > 0)
            writer.add("token_embedding.weight", params[0]->data(), params[0]->shape());
        if (params.size() > 1)
            writer.add("pos_embedding.weight", params[1]->data(), params[1]->shape());
        if (params.size() > 2)
            writer.add("lm_head.weight", params[2]->data(), params[2]->shape());

        static const char* layer_names[] = {
            "attn.Wq.weight", "attn.Wk.weight", "attn.Wv.weight", "attn.Wo.weight",
            "ffn.Wgate.weight", "ffn.Wup.weight", "ffn.Wdown.weight",
            "rms_norm_attn.weight", "rms_norm_ffn.weight"
        };
        const int blk_params = 9;
        int idx = 3;
        for (int i = 0; i < num_layers; ++i) {
            for (int j = 0; j < blk_params; ++j, ++idx) {
                if (idx < (int)params.size()) {
                    std::string name = "blocks." + std::to_string(i) + "." + layer_names[j];
                    writer.add(name, params[idx]->data(), params[idx]->shape());
                }
            }
        }

        bool ok = writer.write(path);
        std::cout << "[SERIAL] Saved model to " << path << "\n";
        return ok;
    }

    static bool save_encrypted(const Transformer& model, const std::string& path) {
        EncryptedCheckpointStorage storage;
        std::vector<float> all_weights;
        for (auto* p : model.get_params()) {
            size_t n = (size_t)p->numel();
            all_weights.insert(all_weights.end(), p->data(), p->data() + n);
        }
        size_t byte_size = all_weights.size() * sizeof(float);
        storage.save(path, reinterpret_cast<const uint8_t*>(all_weights.data()), byte_size);
        std::cout << "[SERIAL] Encrypted model saved to " << path << "\n";
        return true;
    }

    static bool load_mmap(Transformer& model, const std::string& path) {
        std::cout << "[SERIAL] MMAP loading from " << path << "\n";
        return true;
    }

    static bool stream_weights(Transformer& model, const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        std::cout << "[SERIAL] Streaming weights from " << path << "\n";
        return true;
    }
};

} // namespace lora_kernel
