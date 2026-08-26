// Ablation bench. The kernel text between the markers is copied verbatim from
// src/model.cu so the baseline is the shipping kernel; each ablation is one
// explicit textual edit applied by ablate.sh.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); \
    exit(1); } } while (0)

namespace {
// ---- verbatim from src/model.cu ----
