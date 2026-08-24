#include "model.h"
#include "config.h"
#include <stdexcept>

PhiTinyMoEModel::PhiTinyMoEModel(const std::string& model_file)
    : loader_(model_file),
      embeddings_(loader_.load("model.embed_tokens.weight")),
      final_norm_weight_(loader_.load("model.norm.weight")),
      final_norm_bias_(loader_.load("model.norm.bias")),
      lm_head_(loader_, "lm_head.weight", "lm_head.bias") {
    layers_.reserve(apss26::NUM_LAYERS);
    for (std::size_t i = 0; i < apss26::NUM_LAYERS; ++i) layers_.emplace_back(loader_, i);
}

void PhiTinyMoEModel::forward(const std::vector<int>& input_ids, Tensor& logits) const {
    if (input_ids.empty()) throw std::invalid_argument("empty input");
    const std::size_t s = input_ids.size();
    if (s > apss26::MAX_POSITION_EMBEDDINGS) throw std::invalid_argument("sequence is too long");
    Tensor hidden({s, apss26::HIDDEN_SIZE});
    for (std::size_t si = 0; si < s; ++si) {
        const int token = input_ids[si];
        if (token < 0 || static_cast<std::size_t>(token) >= apss26::VOCAB_SIZE) throw std::invalid_argument("token out of vocabulary");
        for (std::size_t h = 0; h < apss26::HIDDEN_SIZE; ++h) hidden.at(si, h) = embeddings_.at(static_cast<std::size_t>(token), h);
    }
    for (const auto& layer : layers_) { Tensor next; layer.forward(hidden, next); hidden = std::move(next); }
    Tensor normed(hidden.shape());
    tensor_ops::layer_norm(hidden, final_norm_weight_, final_norm_bias_, apss26::NORM_EPS, normed);
    Tensor last({1, apss26::HIDDEN_SIZE});
    for (std::size_t h = 0; h < apss26::HIDDEN_SIZE; ++h) last.at(0, h) = normed.at(s - 1, h);
    lm_head_.forward(last, logits);
    logits.reshape({1, apss26::VOCAB_SIZE});
}

void PhiTinyMoEModel::generate(
    const std::vector<std::vector<int>>& input_ids,
    Tensor& logits) const {
    if (input_ids.empty()) {
        throw std::runtime_error("generate received an empty input batch");
    }

    const std::size_t batch = input_ids.size();
    logits = Tensor({batch, apss26::VOCAB_SIZE});

    for (std::size_t b = 0; b < batch; ++b) {
        Tensor one_logits;
        forward(input_ids[b], one_logits);
        for (std::size_t v = 0; v < apss26::VOCAB_SIZE; ++v) {
            logits.at(b, v) = one_logits.at(0, v);
        }
    }
}

void PhiTinyMoEModel::generate_decode(
    const std::vector<std::vector<int>>& input_ids,
    std::size_t max_new_tokens,
    Tensor& logits,
    std::vector<std::vector<int>>& generated_ids) const {
    if (input_ids.empty()) {
        throw std::runtime_error("generate_decode received an empty input batch");
    }
    if (max_new_tokens == 0) {
        throw std::invalid_argument("max_new_tokens must be positive");
    }

    const std::size_t batch = input_ids.size();
    logits = Tensor({batch, max_new_tokens, apss26::VOCAB_SIZE});
    generated_ids.assign(batch, {});

    for (std::size_t b = 0; b < batch; ++b) {
        std::vector<int> prefix = input_ids[b];
        generated_ids[b].reserve(max_new_tokens);

        for (std::size_t step = 0; step < max_new_tokens; ++step) {
            Tensor one_logits;
            forward(prefix, one_logits);

            int next_token = 0;
            float best_logit = one_logits.at(0, 0);
            for (std::size_t v = 1; v < apss26::VOCAB_SIZE; ++v) {
                const float value = one_logits.at(0, v);
                if (value > best_logit) {
                    best_logit = value;
                    next_token = static_cast<int>(v);
                }
            }

            for (std::size_t v = 0; v < apss26::VOCAB_SIZE; ++v) {
                logits.at(b, step, v) = one_logits.at(0, v);
            }
            generated_ids[b].push_back(next_token);

            if (next_token == apss26::EOS_TOKEN_ID) break;
            prefix.push_back(next_token);
        }
    }
}
