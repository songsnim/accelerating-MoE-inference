#pragma once

#include "layer.h"
#include <memory>
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
        // The same weights as a device-side pointer table, so one grouped
        // launch can serve every expert.
        float** w13_ptrs = nullptr;
        float** w2_ptrs = nullptr;
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

    // Every device buffer generate() needs, defined in model.cu. Owned by the
    // model so its ~1.5 GB of cudaFree lands in the destructor instead of in
    // the measured region; the first generate() still pays for the cudaMalloc.
    struct Scratch;

    ModelLoader loader_;
    mutable std::unique_ptr<Scratch> scratch_;
    Tensor embeddings_;
    // The same table on the device, so the per-token gather is a kernel
    // instead of 255 MB of single-threaded host memcpy plus an upload.
    float* d_embeddings_ = nullptr;
    DeviceNorm final_norm_;
    DeviceLinear lm_head_;
    std::vector<Layer> layers_;
};
