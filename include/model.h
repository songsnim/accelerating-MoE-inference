#pragma once

#include "layer.h"
#include <string>
#include <vector>

class PhiTinyMoEModel {
public:
    explicit PhiTinyMoEModel(const std::string& model_file);

    void forward(const std::vector<int>& input_ids, Tensor& logits) const;

    void generate(const std::vector<std::vector<int>>& input_ids,
                  Tensor& logits) const;

    void generate_decode(const std::vector<std::vector<int>>& input_ids,
                         std::size_t max_new_tokens,
                         Tensor& logits,
                         std::vector<std::vector<int>>& generated_ids) const;

private:
    // A decoder layer taken apart into its primitives. PhiDecoderLayer keeps
    // its projections private, so the batched path assembles the same weights
    // from layer.h's public classes instead. Still one copy of the weights.
    struct Layer {
        Layer(const ModelLoader& loader, std::size_t layer_idx);

        Tensor input_norm_weight, input_norm_bias;
        Tensor post_norm_weight, post_norm_bias;
        Linear q_proj, k_proj, v_proj, o_proj;
        PhiMoE moe;
    };

    ModelLoader loader_;
    Tensor embeddings_;
    Tensor final_norm_weight_, final_norm_bias_;
    Linear lm_head_;
    std::vector<Layer> layers_;
};
