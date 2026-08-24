#include "layer.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

// Match the reference CUDA/PyTorch path for nonlinear and attention exponentials.
static inline float accurate_exp(float x) {
    return static_cast<float>(std::exp(static_cast<double>(x)));
}

Linear::Linear(const ModelLoader& loader, const std::string& weight, const std::string& bias)
    : weight_(loader.load(weight)) {
    if (!bias.empty() && loader.has(bias)) bias_ = loader.load(bias);
}

void Linear::forward(const Tensor& x, Tensor& y) const {
    const std::size_t rows = x.size() / x.size(x.ndim() - 1);
    y = Tensor({rows, weight_.size(0)});
    Tensor flat = x;
    flat.reshape({rows, x.size(x.ndim() - 1)});
    tensor_ops::matmul_transposed(flat, weight_, y);
    if (bias_.size()) tensor_ops::add_bias_inplace(y, bias_);
}

PhiMLP::PhiMLP(const ModelLoader& loader, const std::string& prefix)
    : w1_(loader.load(prefix + ".w1.weight")),
      w2_(loader.load(prefix + ".w2.weight")),
      w3_(loader.load(prefix + ".w3.weight")) {}

void PhiMLP::forward(const Tensor& x, Tensor& y) const {
    const std::size_t rows = x.size() / x.size(x.ndim() - 1), hidden = x.size(x.ndim() - 1);
    Tensor flat = x;
    flat.reshape({rows, hidden});
    Tensor gate({rows, w1_.size(0)}), up({rows, w3_.size(0)}), activated({rows, w1_.size(0)});
    tensor_ops::matmul_transposed(flat, w1_, gate);
    tensor_ops::silu(gate, activated);
    tensor_ops::matmul_transposed(flat, w3_, up);
    tensor_ops::mul(activated, up, activated);
    Tensor out({rows, w2_.size(0)});
    tensor_ops::matmul_transposed(activated, w2_, out);
    out.reshape(x.shape());
    y = std::move(out);
}

void PhiMoE::route(const Tensor& logits, std::vector<std::pair<int, float>>& routes) const {
    routes.clear();
    std::vector<float> scores(apss26::NUM_EXPERTS);
    for (std::size_t e = 0; e < apss26::NUM_EXPERTS; ++e) {
        const float score = logits[e];
        const float rounded = std::floor(std::fabs(score) /
            apss26::ROUTER_SCORE_QUANTUM + 0.5f) * apss26::ROUTER_SCORE_QUANTUM;
        scores[e] = score < 0.0f ? -rounded : rounded;
    }
    auto select = [](const std::vector<float>& values, int excluded) {
        int best = -1; float best_value = -std::numeric_limits<float>::infinity();
        for (std::size_t e = 0; e < values.size(); ++e) {
            if (static_cast<int>(e) == excluded) continue;
            if (best < 0 || values[e] > best_value + apss26::ROUTER_TIE_EPS) {
                best = static_cast<int>(e); best_value = values[e];
            }
        }
        return best;
    };
    const int first = select(scores, -1);
    const int second = select(scores, first);
    routes.emplace_back(first, 0.5f);
    routes.emplace_back(second, 0.5f);
}

PhiMoE::PhiMoE(const ModelLoader& loader, std::size_t layer_idx) {
    const std::string base = "model.layers." + std::to_string(layer_idx) + ".block_sparse_moe";
    gate_ = loader.load(base + ".gate.weight");
    experts_.reserve(apss26::NUM_EXPERTS);
    for (std::size_t e = 0; e < apss26::NUM_EXPERTS; ++e)
        experts_.emplace_back(loader, base + ".experts." + std::to_string(e));
}

void PhiMoE::forward(const Tensor& x, Tensor& y) const {
    const std::size_t rows = x.size(0), h = x.size(1);
    Tensor flat = x;
    Tensor router({rows, apss26::NUM_EXPERTS});
    tensor_ops::matmul_transposed(flat, gate_, router);
    y = Tensor({rows, h}); y.zero();
    std::vector<std::vector<std::pair<std::size_t, float>>> assignments(apss26::NUM_EXPERTS);
    for (std::size_t t = 0; t < rows; ++t) {
        Tensor one({apss26::NUM_EXPERTS});
        for (std::size_t e = 0; e < apss26::NUM_EXPERTS; ++e) one[e] = router.at(t, e);
        std::vector<std::pair<int, float>> r; route(one, r);
        for (auto [e, w] : r) assignments[e].emplace_back(t, w);
    }
    for (std::size_t e = 0; e < apss26::NUM_EXPERTS; ++e) {
        if (assignments[e].empty()) continue;
        Tensor input({assignments[e].size(), h}), output;
        for (std::size_t i = 0; i < assignments[e].size(); ++i)
            for (std::size_t j = 0; j < h; ++j) input.at(i, j) = flat.at(assignments[e][i].first, j);
        experts_[e].forward(input, output);
        output.reshape({assignments[e].size(), h});
        for (std::size_t i = 0; i < assignments[e].size(); ++i)
            for (std::size_t j = 0; j < h; ++j) y.at(assignments[e][i].first, j) += assignments[e][i].second * output.at(i, j);
    }
}

PhiAttention::PhiAttention(const ModelLoader& loader, std::size_t layer_idx)
    : q_proj_(loader, "model.layers." + std::to_string(layer_idx) + ".self_attn.q_proj.weight", "model.layers." + std::to_string(layer_idx) + ".self_attn.q_proj.bias"),
      k_proj_(loader, "model.layers." + std::to_string(layer_idx) + ".self_attn.k_proj.weight", "model.layers." + std::to_string(layer_idx) + ".self_attn.k_proj.bias"),
      v_proj_(loader, "model.layers." + std::to_string(layer_idx) + ".self_attn.v_proj.weight", "model.layers." + std::to_string(layer_idx) + ".self_attn.v_proj.bias"),
      o_proj_(loader, "model.layers." + std::to_string(layer_idx) + ".self_attn.o_proj.weight", "model.layers." + std::to_string(layer_idx) + ".self_attn.o_proj.bias") {}

void PhiAttention::forward(const Tensor& x, Tensor& y) const {
    const std::size_t s = x.size(0);
    Tensor q, k, v; q_proj_.forward(x, q); k_proj_.forward(x, k); v_proj_.forward(x, v);
    tensor_ops::apply_rope(q, k, s, apss26::NUM_ATTENTION_HEADS, apss26::NUM_KV_HEADS, apss26::HEAD_DIM, apss26::ROPE_THETA);
    Tensor out({s, apss26::NUM_ATTENTION_HEADS * apss26::HEAD_DIM});
    const std::size_t group = apss26::NUM_ATTENTION_HEADS / apss26::NUM_KV_HEADS;
    for (std::size_t qi = 0; qi < s; ++qi) for (std::size_t qh = 0; qh < apss26::NUM_ATTENTION_HEADS; ++qh) {
        const std::size_t kh = qh / group;
        float maxv = -std::numeric_limits<float>::infinity();
        for (std::size_t ki = 0; ki <= qi; ++ki) {
            if (qi - ki >= apss26::SLIDING_WINDOW) continue;
            float score = 0.0f;
            for (std::size_t d = 0; d < apss26::HEAD_DIM; ++d) score += q.at(qi, qh * apss26::HEAD_DIM + d) * k.at(ki, kh * apss26::HEAD_DIM + d);
            maxv = std::max(maxv, score / std::sqrt(static_cast<float>(apss26::HEAD_DIM)));
        }
        float denom = 0.0f;
        for (std::size_t ki = 0; ki <= qi; ++ki) if (qi - ki < apss26::SLIDING_WINDOW) {
            float score = 0.0f;
            for (std::size_t d = 0; d < apss26::HEAD_DIM; ++d) score += q.at(qi, qh * apss26::HEAD_DIM + d) * k.at(ki, kh * apss26::HEAD_DIM + d);
            denom += accurate_exp(score / std::sqrt(static_cast<float>(apss26::HEAD_DIM)) - maxv);
        }
        for (std::size_t d = 0; d < apss26::HEAD_DIM; ++d) {
            float value = 0.0f;
            for (std::size_t ki = 0; ki <= qi; ++ki) if (qi - ki < apss26::SLIDING_WINDOW) {
                float score = 0.0f;
                for (std::size_t z = 0; z < apss26::HEAD_DIM; ++z) score += q.at(qi, qh * apss26::HEAD_DIM + z) * k.at(ki, kh * apss26::HEAD_DIM + z);
                value += accurate_exp(score / std::sqrt(static_cast<float>(apss26::HEAD_DIM)) - maxv) / denom * v.at(ki, kh * apss26::HEAD_DIM + d);
            }
            out.at(qi, qh * apss26::HEAD_DIM + d) = value;
        }
    }
    o_proj_.forward(out, y);
}

PhiDecoderLayer::PhiDecoderLayer(const ModelLoader& loader, std::size_t layer_idx)
    : input_norm_weight_(loader.load("model.layers." + std::to_string(layer_idx) + ".input_layernorm.weight")),
      input_norm_bias_(loader.load("model.layers." + std::to_string(layer_idx) + ".input_layernorm.bias")),
      post_norm_weight_(loader.load("model.layers." + std::to_string(layer_idx) + ".post_attention_layernorm.weight")),
      post_norm_bias_(loader.load("model.layers." + std::to_string(layer_idx) + ".post_attention_layernorm.bias")),
      attention_(loader, layer_idx), moe_(loader, layer_idx) {}

void PhiDecoderLayer::forward(const Tensor& x, Tensor& y) const {
    Tensor normed(x.shape()), attn;
    tensor_ops::layer_norm(x, input_norm_weight_, input_norm_bias_, apss26::NORM_EPS, normed);
    attention_.forward(normed, attn);
    tensor_ops::add_inplace(attn, x);
    Tensor post(attn.shape()), ff;
    tensor_ops::layer_norm(attn, post_norm_weight_, post_norm_bias_, apss26::NORM_EPS, post);
    moe_.forward(post, ff);
    tensor_ops::add_inplace(ff, attn);
    y = std::move(ff);
}
