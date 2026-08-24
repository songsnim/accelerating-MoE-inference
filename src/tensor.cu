#include "tensor.h"
#include "config.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

Tensor::Tensor(std::vector<std::size_t> shape) : shape_(std::move(shape)) {
    std::size_t n = 1;
    for (std::size_t d : shape_) n *= d;
    data_.assign(n, 0.0f);
}

std::size_t Tensor::size(std::size_t dim) const {
    if (dim >= shape_.size()) throw std::out_of_range("Tensor dimension");
    return shape_[dim];
}

std::size_t Tensor::offset(std::initializer_list<std::size_t> indices) const {
    if (indices.size() != shape_.size()) throw std::invalid_argument("Tensor rank mismatch");
    std::size_t out = 0, stride = 1;
    auto it = indices.end();
    for (std::size_t d = shape_.size(); d-- > 0;) {
        --it;
        if (*it >= shape_[d]) throw std::out_of_range("Tensor index");
        out += *it * stride;
        stride *= shape_[d];
    }
    return out;
}

float& Tensor::at(std::size_t i) { return data_[offset({i})]; }
float& Tensor::at(std::size_t i, std::size_t j) { return data_[offset({i, j})]; }
float& Tensor::at(std::size_t i, std::size_t j, std::size_t k) { return data_[offset({i, j, k})]; }
float& Tensor::at(std::size_t i, std::size_t j, std::size_t k, std::size_t l) { return data_[offset({i, j, k, l})]; }
const float& Tensor::at(std::size_t i) const { return data_[offset({i})]; }
const float& Tensor::at(std::size_t i, std::size_t j) const { return data_[offset({i, j})]; }
const float& Tensor::at(std::size_t i, std::size_t j, std::size_t k) const { return data_[offset({i, j, k})]; }
const float& Tensor::at(std::size_t i, std::size_t j, std::size_t k, std::size_t l) const { return data_[offset({i, j, k, l})]; }

void Tensor::reshape(std::vector<std::size_t> shape) {
    std::size_t n = 1;
    for (std::size_t d : shape) n *= d;
    if (n != data_.size()) throw std::invalid_argument("reshape changes tensor size");
    shape_ = std::move(shape);
}

void Tensor::fill(float value) { std::fill(data_.begin(), data_.end(), value); }

namespace tensor_ops {
// Match the reference CUDA/PyTorch path: evaluate exp in double precision
// and round only the final result back to float.
static inline float accurate_exp(float x) {
    return static_cast<float>(std::exp(static_cast<double>(x)));
}

void matmul_transposed(const Tensor& a, const Tensor& b, Tensor& c) {
    const std::size_t m = a.size(0), k = a.size(1), n = b.size(0);
    if (b.size(1) != k || c.size(0) != m || c.size(1) != n) throw std::invalid_argument("matmul_transposed shape");
    c.zero();
#pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(m); ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (std::size_t p = 0; p < k; ++p) sum += a.at(i, p) * b.at(j, p);
            c.at(i, j) = sum;
        }
    }
}

void add_inplace(Tensor& a, const Tensor& b) {
    if (a.size() != b.size()) throw std::invalid_argument("add shape");
#pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(a.size()); ++i) a[i] += b[i];
}

void add_bias_inplace(Tensor& a, const Tensor& bias) {
    if (bias.ndim() != 1 || a.size(a.ndim() - 1) != bias.size(0)) throw std::invalid_argument("bias shape");
    const std::size_t h = bias.size(0);
#pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(a.size()); ++i) a[i] += bias[i % h];
}

void mul(const Tensor& a, const Tensor& b, Tensor& c) {
    if (a.size() != b.size() || c.size() != a.size()) throw std::invalid_argument("mul shape");
#pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(a.size()); ++i) c[i] = a[i] * b[i];
}

void silu(const Tensor& x, Tensor& y) {
    if (x.size() != y.size()) throw std::invalid_argument("silu shape");
#pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(x.size()); ++i) y[i] = x[i] / (1.0f + accurate_exp(-x[i]));
}

void layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps, Tensor& y) {
    const std::size_t h = x.size(x.ndim() - 1), rows = x.size() / h;
    if (weight.size() != h || bias.size() != h || y.size() != x.size()) throw std::invalid_argument("layer norm shape");
#pragma omp parallel for
    for (long long r = 0; r < static_cast<long long>(rows); ++r) {
        const std::size_t base = static_cast<std::size_t>(r) * h;
        float mean = 0.0f;
        for (std::size_t j = 0; j < h; ++j) mean += x[base + j];
        mean /= static_cast<float>(h);
        float var = 0.0f;
        for (std::size_t j = 0; j < h; ++j) { const float d = x[base + j] - mean; var += d * d; }
        const float inv = 1.0f / std::sqrt(var / static_cast<float>(h) + eps);
        for (std::size_t j = 0; j < h; ++j) y[base + j] = (x[base + j] - mean) * inv * weight[j] + bias[j];
    }
}

void apply_rope(Tensor& q, Tensor& k, std::size_t seq_len,
                std::size_t q_heads, std::size_t kv_heads, std::size_t head_dim, float theta) {
    const std::size_t half = head_dim / 2;
    for (std::size_t s = 0; s < seq_len; ++s) {
        for (std::size_t h = 0; h < q_heads; ++h) for (std::size_t j = 0; j < half; ++j) {
            const float inv = std::pow(theta, -2.0f * static_cast<float>(j) / static_cast<float>(head_dim));
            const float c = std::cos(static_cast<float>(s) * inv), sn = std::sin(static_cast<float>(s) * inv);
            const float x0 = q.at(s, h * head_dim + j), x1 = q.at(s, h * head_dim + j + half);
            q.at(s, h * head_dim + j) = x0 * c - x1 * sn;
            q.at(s, h * head_dim + j + half) = x1 * c + x0 * sn;
        }
        for (std::size_t h = 0; h < kv_heads; ++h) for (std::size_t j = 0; j < half; ++j) {
            const float inv = std::pow(theta, -2.0f * static_cast<float>(j) / static_cast<float>(head_dim));
            const float c = std::cos(static_cast<float>(s) * inv), sn = std::sin(static_cast<float>(s) * inv);
            const float x0 = k.at(s, h * head_dim + j), x1 = k.at(s, h * head_dim + j + half);
            k.at(s, h * head_dim + j) = x0 * c - x1 * sn;
            k.at(s, h * head_dim + j + half) = x1 * c + x0 * sn;
        }
    }
}
}
