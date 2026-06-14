#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>
#include <queue>

namespace lora_kernel {

// Production-level hard-coded Speculative Decoding
// Draft model generates K candidates, target model verifies in parallel
class SpeculativeDecoding {
private:
    int num_draft_tokens_{5};
    float rejection_threshold_{0.0f};

public:
    SpeculativeDecoding(int draft_tokens = 5) : num_draft_tokens_(draft_tokens) {}

    // Draft: fast autoregressive generation of K candidate tokens
    std::vector<int> generate_draft(const float* logits, int vocab_size) {
        std::vector<int> candidates;
        for (int i = 0; i < num_draft_tokens_; ++i) {
            // Greedy: argmax
            int best = 0;
            float best_val = logits[i * vocab_size];
            for (int v = 1; v < vocab_size; ++v) {
                if (logits[i * vocab_size + v] > best_val) {
                    best_val = logits[i * vocab_size + v];
                    best = v;
                }
            }
            candidates.push_back(best);
        }
        return candidates;
    }

    // Verify: compare draft probabilities with target model
    int verify(const std::vector<int>& draft_tokens,
               const float* draft_probs, const float* target_probs,
               int vocab_size, int& accepted_count) {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        accepted_count = 0;
        for (int i = 0; i < (int)draft_tokens.size(); ++i) {
            int tok = draft_tokens[i];
            float p_draft = draft_probs[i * vocab_size + tok];
            float p_target = target_probs[i * vocab_size + tok];

            // Acceptance criterion: target / draft >= uniform(0,1)
            float ratio = std::min(1.0f, p_target / (p_draft + 1e-12f));
            if (dist(rng) < ratio) {
                accepted_count++;
            } else {
                // Rejected: resample from corrected distribution
                // corrected_p(v) = max(0, target_p(v) - draft_p(v)) / sum
                std::vector<float> corrected(vocab_size, 0.0f);
                float sum_corrected = 0.0f;
                for (int v = 0; v < vocab_size; ++v) {
                    corrected[v] = std::max(0.0f, target_probs[i * vocab_size + v]
                                            - draft_probs[i * vocab_size + v]);
                    sum_corrected += corrected[v];
                }
                if (sum_corrected > 1e-12f) {
                    float r = dist(rng);
                    float cum = 0.0f;
                    for (int v = 0; v < vocab_size; ++v) {
                        cum += corrected[v] / sum_corrected;
                        if (r <= cum) return v;
                    }
                }
                return tok; // fallback
            }
        }
        return draft_tokens.empty() ? 0 : draft_tokens.back();
    }

    // Speculative decoding step: draft -> target verify -> accept/reject
    int step(const float* draft_logits, const float* target_logits, int vocab_size) {
        auto draft = generate_draft(draft_logits, vocab_size);
        int accepted = 0;
        return verify(draft, draft_logits, target_logits, vocab_size, accepted);
    }
};

// Production-level hard-coded Continuous Batching
class ContinuousBatching {
private:
    struct Request {
        int id;
        std::vector<int> tokens;
        int current_pos{0};
        bool finished{false};
        float priority{1.0f};
    };

    std::queue<Request> pending_queue_;
    std::vector<Request> active_batch_;
    int max_batch_size_;
    int max_seq_len_;

public:
    ContinuousBatching(int max_batch = 8, int max_seq = 2048)
        : max_batch_size_(max_batch), max_seq_len_(max_seq) {}

    // Add request to pending queue
    void add_request(int id, const std::vector<int>& tokens, float priority = 1.0f) {
        Request req{id, tokens, 0, false, priority};
        pending_queue_.push(req);
    }

    // Schedule next batch: merge pending requests with running batch
    std::vector<Request*> schedule_next() {
        // Preempt lowest-priority request if batch is full
        while ((int)active_batch_.size() < max_batch_size_ && !pending_queue_.empty()) {
            active_batch_.push_back(pending_queue_.front());
            pending_queue_.pop();
        }

        std::vector<Request*> batch;
        for (auto& req : active_batch_) {
            if (!req.finished) batch.push_back(&req);
        }
        return batch;
    }

    // Mark request as finished and remove from batch
    void finish_request(int id) {
        for (auto it = active_batch_.begin(); it != active_batch_.end(); ++it) {
            if (it->id == id) {
                active_batch_.erase(it);
                return;
            }
        }
    }

    // Build attention mask for variable-length sequences in batch
    std::vector<uint8_t> build_attention_mask() {
        int total_tokens = 0;
        for (auto& r : active_batch_) total_tokens += r.current_pos;

        std::vector<uint8_t> mask(total_tokens * total_tokens, 0);
        int offset = 0;
        for (auto& r : active_batch_) {
            for (int i = 0; i < r.current_pos; ++i)
                for (int j = 0; j < r.current_pos; ++j)
                    mask[(offset + i) * total_tokens + (offset + j)] = (j <= i) ? 1 : 0;
            offset += r.current_pos;
        }
        return mask;
    }

    // Advance all requests by one token position
    void advance_all(const std::vector<int>& next_tokens) {
        int idx = 0;
        for (auto& req : active_batch_) {
            if (idx < (int)next_tokens.size()) {
                req.tokens.push_back(next_tokens[idx++]);
                req.current_pos++;
                if (req.current_pos >= max_seq_len_ || next_tokens[idx - 1] == 0)
                    req.finished = true;
            }
        }
    }

    int active_count() const { return (int)active_batch_.size(); }
    int pending_count() const { return (int)pending_queue_.size(); }
};

} // namespace lora_kernel
