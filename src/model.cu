#include "model.h"
#include "config.h"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

// layer.cu evaluates every exponential in double and rounds once. Keep that.
inline float accurate_exp(float x) {
    return static_cast<float>(std::exp(static_cast<double>(x)));
}

// RoPE + causal attention for one sequence occupying rows [begin, begin+len)
// of the batched projections. Arithmetic mirrors PhiAttention::forward, with
// each q.k dot product evaluated once instead of once per output dimension.
void attend_sequence(Tensor& q, Tensor& k, const Tensor& v,
                     std::size_t begin, std::size_t len, Tensor& out) {
    constexpr std::size_t D = apss26::HEAD_DIM;
    constexpr std::size_t QH = apss26::NUM_ATTENTION_HEADS;
    constexpr std::size_t KVH = apss26::NUM_KV_HEADS;
    constexpr std::size_t half = D / 2;
    const std::size_t group = QH / KVH;
    const float scale = std::sqrt(static_cast<float>(D));

    float* qbase = q.data() + begin * QH * D;
    float* kbase = k.data() + begin * KVH * D;
    const float* vbase = v.data() + begin * KVH * D;
    float* obase = out.data() + begin * QH * D;

    for (std::size_t s = 0; s < len; ++s) {
        for (std::size_t h = 0; h < QH; ++h) for (std::size_t j = 0; j < half; ++j) {
            const float inv = std::pow(apss26::ROPE_THETA, -2.0f * static_cast<float>(j) / static_cast<float>(D));
            const float c = std::cos(static_cast<float>(s) * inv), sn = std::sin(static_cast<float>(s) * inv);
            float* row = qbase + s * QH * D + h * D;
            const float x0 = row[j], x1 = row[j + half];
            row[j] = x0 * c - x1 * sn;
            row[j + half] = x1 * c + x0 * sn;
        }
        for (std::size_t h = 0; h < KVH; ++h) for (std::size_t j = 0; j < half; ++j) {
            const float inv = std::pow(apss26::ROPE_THETA, -2.0f * static_cast<float>(j) / static_cast<float>(D));
            const float c = std::cos(static_cast<float>(s) * inv), sn = std::sin(static_cast<float>(s) * inv);
            float* row = kbase + s * KVH * D + h * D;
            const float x0 = row[j], x1 = row[j + half];
            row[j] = x0 * c - x1 * sn;
            row[j + half] = x1 * c + x0 * sn;
        }
    }

    std::vector<float> weights(len);
    for (std::size_t qi = 0; qi < len; ++qi) for (std::size_t qh = 0; qh < QH; ++qh) {
        const std::size_t kh = qh / group;
        const float* qrow = qbase + qi * QH * D + qh * D;
        const std::size_t lo = qi + 1 > apss26::SLIDING_WINDOW ? qi + 1 - apss26::SLIDING_WINDOW : 0;

        float maxv = -std::numeric_limits<float>::infinity();
        for (std::size_t ki = lo; ki <= qi; ++ki) {
            const float* krow = kbase + ki * KVH * D + kh * D;
            float score = 0.0f;
            for (std::size_t d = 0; d < D; ++d) score += qrow[d] * krow[d];
            weights[ki] = score / scale;
            maxv = std::max(maxv, weights[ki]);
        }
        float denom = 0.0f;
        for (std::size_t ki = lo; ki <= qi; ++ki) {
            weights[ki] = accurate_exp(weights[ki] - maxv);
            denom += weights[ki];
        }

        float acc[D] = {};
        for (std::size_t ki = lo; ki <= qi; ++ki) {
            const float w = weights[ki] / denom;
            const float* vrow = vbase + ki * KVH * D + kh * D;
            for (std::size_t d = 0; d < D; ++d) acc[d] += w * vrow[d];
        }
        float* orow = obase + qi * QH * D + qh * D;
        for (std::size_t d = 0; d < D; ++d) orow[d] = acc[d];
    }
}

}  // namespace

PhiTinyMoEModel::Layer::Layer(const ModelLoader& loader, std::size_t layer_idx)
    : input_norm_weight(loader.load("model.layers." + std::to_string(layer_idx) + ".input_layernorm.weight")),
      input_norm_bias(loader.load("model.layers." + std::to_string(layer_idx) + ".input_layernorm.bias")),
      post_norm_weight(loader.load("model.layers." + std::to_string(layer_idx) + ".post_attention_layernorm.weight")),
      post_norm_bias(loader.load("model.layers." + std::to_string(layer_idx) + ".post_attention_layernorm.bias")),
      q_proj(loader, "model.layers." + std::to_string(layer_idx) + ".self_attn.q_proj.weight", "model.layers." + std::to_string(layer_idx) + ".self_attn.q_proj.bias"),
      k_proj(loader, "model.layers." + std::to_string(layer_idx) + ".self_attn.k_proj.weight", "model.layers." + std::to_string(layer_idx) + ".self_attn.k_proj.bias"),
      v_proj(loader, "model.layers." + std::to_string(layer_idx) + ".self_attn.v_proj.weight", "model.layers." + std::to_string(layer_idx) + ".self_attn.v_proj.bias"),
      o_proj(loader, "model.layers." + std::to_string(layer_idx) + ".self_attn.o_proj.weight", "model.layers." + std::to_string(layer_idx) + ".self_attn.o_proj.bias"),
      moe(loader, layer_idx) {}

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
    generate({input_ids}, logits);
}

// Every sequence in the batch is packed into a single [T, HIDDEN] activation
// matrix, so the 32 layers are traversed once for the whole batch. Layer norm,
// the projections and the MoE are row-wise, so packing does not change any
// row's arithmetic; only attention needs to know the sequence boundaries.
void PhiTinyMoEModel::generate(
    const std::vector<std::vector<int>>& input_ids,
    Tensor& logits) const {
    if (input_ids.empty()) {
        throw std::runtime_error("generate received an empty input batch");
    }

    const std::size_t batch = input_ids.size();
    std::vector<std::size_t> offset(batch), length(batch);
    std::size_t total = 0;
    for (std::size_t b = 0; b < batch; ++b) {
        const std::size_t s = input_ids[b].size();
        if (s == 0) throw std::invalid_argument("empty input");
        if (s > apss26::MAX_POSITION_EMBEDDINGS) throw std::invalid_argument("sequence is too long");
        offset[b] = total;
        length[b] = s;
        total += s;
    }

    Tensor hidden({total, apss26::HIDDEN_SIZE});
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t si = 0; si < length[b]; ++si) {
            const int token = input_ids[b][si];
            if (token < 0 || static_cast<std::size_t>(token) >= apss26::VOCAB_SIZE) throw std::invalid_argument("token out of vocabulary");
            float* row = hidden.data() + (offset[b] + si) * apss26::HIDDEN_SIZE;
            const float* src = embeddings_.data() + static_cast<std::size_t>(token) * apss26::HIDDEN_SIZE;
            for (std::size_t h = 0; h < apss26::HIDDEN_SIZE; ++h) row[h] = src[h];
        }
    }

    for (const Layer& layer : layers_) {
        Tensor normed(hidden.shape());
        tensor_ops::layer_norm(hidden, layer.input_norm_weight, layer.input_norm_bias, apss26::NORM_EPS, normed);

        Tensor q, k, v;
        layer.q_proj.forward(normed, q);
        layer.k_proj.forward(normed, k);
        layer.v_proj.forward(normed, v);

        Tensor context({total, apss26::NUM_ATTENTION_HEADS * apss26::HEAD_DIM});
#pragma omp parallel for schedule(dynamic)
        for (long long b = 0; b < static_cast<long long>(batch); ++b) {
            attend_sequence(q, k, v, offset[b], length[b], context);
        }

        Tensor attn;
        layer.o_proj.forward(context, attn);
        tensor_ops::add_inplace(attn, hidden);

        Tensor post(attn.shape()), ff;
        tensor_ops::layer_norm(attn, layer.post_norm_weight, layer.post_norm_bias, apss26::NORM_EPS, post);
        layer.moe.forward(post, ff);
        tensor_ops::add_inplace(ff, attn);
        hidden = std::move(ff);
    }

    Tensor normed(hidden.shape());
    tensor_ops::layer_norm(hidden, final_norm_weight_, final_norm_bias_, apss26::NORM_EPS, normed);

    Tensor last({batch, apss26::HIDDEN_SIZE});
    for (std::size_t b = 0; b < batch; ++b) {
        const float* src = normed.data() + (offset[b] + length[b] - 1) * apss26::HIDDEN_SIZE;
        float* dst = last.data() + b * apss26::HIDDEN_SIZE;
        for (std::size_t h = 0; h < apss26::HIDDEN_SIZE; ++h) dst[h] = src[h];
    }
    lm_head_.forward(last, logits);
    logits.reshape({batch, apss26::VOCAB_SIZE});
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
