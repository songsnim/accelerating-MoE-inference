#pragma once

#include "layer.h"
#include <string>
#include <vector>

class PhiTinyMoEModel {
public:
    explicit PhiTinyMoEModel(const std::string& model_file);
    ~PhiTinyMoEModel();

    PhiTinyMoEModel(const PhiTinyMoEModel&) = delete;
    PhiTinyMoEModel& operator=(const PhiTinyMoEModel&) = delete;

    void forward(const std::vector<int>& input_ids, Tensor& logits) const;

    void generate(const std::vector<std::vector<int>>& input_ids,
                  Tensor& logits) const;

    void generate_decode(const std::vector<std::vector<int>>& input_ids,
                         std::size_t max_new_tokens,
                         Tensor& logits,
                         std::vector<std::vector<int>>& generated_ids) const;

private:
    // A Linear whose weight lives in device memory. The host copy is dropped
    // right after the upload, which happens during model construction.
    struct DeviceLinear {
        DeviceLinear() = default;
        DeviceLinear(const ModelLoader& loader, const std::string& weight,
                     const std::string& bias);
        // Two weights stacked row-wise into one matrix (`top` above `bottom`),
        // so one GEMM produces both halves.
        DeviceLinear(const ModelLoader& loader, const std::string& top,
                     const std::string& bottom, bool concat_rows);
        DeviceLinear(const DeviceLinear&) = delete;
        DeviceLinear& operator=(const DeviceLinear&) = delete;
        DeviceLinear(DeviceLinear&& other) noexcept;
        void free();
        // y[rows, out] = x[rows, in] * weight[out, in]^T + bias
        void forward(const float* x, float* y, std::size_t rows) const;

        float* weight = nullptr;
        float* bias = nullptr;
        std::size_t out = 0, in = 0;
    };

    // A layer norm's weight and bias in device memory.
    struct DeviceNorm {
        DeviceNorm() = default;
        DeviceNorm(const ModelLoader& loader, const std::string& weight,
                   const std::string& bias);
        DeviceNorm(const DeviceNorm&) = delete;
        DeviceNorm& operator=(const DeviceNorm&) = delete;
        DeviceNorm(DeviceNorm&& other) noexcept;
        void free();
        // y[rows, cols] = normalise(x[rows, cols]) * weight + bias
        void forward(const float* x, float* y, std::size_t rows) const;

        float* weight = nullptr;
        float* bias = nullptr;
        std::size_t cols = 0;
    };

    // The router plus all 16 experts of one layer, on the device.
    struct DeviceMoE {
        DeviceMoE(const ModelLoader& loader, std::size_t layer_idx);
        DeviceMoE(const DeviceMoE&) = delete;
        DeviceMoE& operator=(const DeviceMoE&) = delete;
        DeviceMoE(DeviceMoE&&) noexcept = default;
        void free();

        DeviceLinear gate;
        // w13[e] is w1 stacked on w3, so the two share one GEMM.
        std::vector<DeviceLinear> w13, w2;
    };

    // A decoder layer taken apart into its primitives. PhiDecoderLayer keeps
    // its projections private, so the batched path assembles the same weights
    // from the loader instead. Still one copy of the weights.
    struct Layer {
        Layer(const ModelLoader& loader, std::size_t layer_idx);

        DeviceNorm input_norm, post_norm;
        DeviceLinear q_proj, k_proj, v_proj, o_proj;
        DeviceMoE moe;
    };

    ModelLoader loader_;
    Tensor embeddings_;
    DeviceNorm final_norm_;
    DeviceLinear lm_head_;
    std::vector<Layer> layers_;
};
