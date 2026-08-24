#pragma once

#include <cstddef>
#include <string>
#include <vector>

class Tensor {
public:
    Tensor() = default;
    explicit Tensor(std::vector<std::size_t> shape);

    std::size_t ndim() const { return shape_.size(); }
    std::size_t size() const { return data_.size(); }
    std::size_t size(std::size_t dim) const;
    const std::vector<std::size_t>& shape() const { return shape_; }

    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }
    float& operator[](std::size_t i) { return data_[i]; }
    const float& operator[](std::size_t i) const { return data_[i]; }

    float& at(std::size_t i);
    float& at(std::size_t i, std::size_t j);
    float& at(std::size_t i, std::size_t j, std::size_t k);
    float& at(std::size_t i, std::size_t j, std::size_t k, std::size_t l);
    const float& at(std::size_t i) const;
    const float& at(std::size_t i, std::size_t j) const;
    const float& at(std::size_t i, std::size_t j, std::size_t k) const;
    const float& at(std::size_t i, std::size_t j, std::size_t k, std::size_t l) const;

    void reshape(std::vector<std::size_t> shape);
    void fill(float value);
    void zero() { fill(0.0f); }

private:
    std::vector<std::size_t> shape_;
    std::vector<float> data_;
    std::size_t offset(std::initializer_list<std::size_t> indices) const;
};

namespace tensor_ops {
void matmul(const Tensor& a, const Tensor& b, Tensor& c);              // [M,K] x [K,N]
void matmul_transposed(const Tensor& a, const Tensor& b, Tensor& c);   // [M,K] x [N,K]
void add_inplace(Tensor& a, const Tensor& b);
void add_bias_inplace(Tensor& a, const Tensor& bias);
void mul(const Tensor& a, const Tensor& b, Tensor& c);
void silu(const Tensor& x, Tensor& y);
void softmax_last_dim(const Tensor& x, Tensor& y);
void layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias,
                float eps, Tensor& y);
void apply_rope(Tensor& q, Tensor& k, std::size_t seq_len,
                std::size_t q_heads, std::size_t kv_heads, std::size_t head_dim,
                float theta);
}
