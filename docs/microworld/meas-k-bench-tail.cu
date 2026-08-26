// ---- end verbatim ----
}  // namespace

int main(int argc, char** argv) {
    // q_proj shape: the largest single projection in the model.
    const int M = argc > 1 ? atoi(argv[1]) : 15583;
    const int K = argc > 2 ? atoi(argv[2]) : 4096;
    const int N = argc > 3 ? atoi(argv[3]) : 2048;
    const int reps = argc > 4 ? atoi(argv[4]) : 20;

    float *a, *b, *bias, *c;
    CK(cudaMalloc(&a, (size_t)M * K * sizeof(float)));
    CK(cudaMalloc(&b, (size_t)N * K * sizeof(float)));
    CK(cudaMalloc(&bias, (size_t)N * sizeof(float)));
    CK(cudaMalloc(&c, (size_t)M * N * sizeof(float)));
    CK(cudaMemset(a, 0x3c, (size_t)M * K * sizeof(float)));
    CK(cudaMemset(b, 0x3c, (size_t)N * K * sizeof(float)));
    CK(cudaMemset(bias, 0, (size_t)N * sizeof(float)));

    const dim3 grid((N + PROJ_BN - 1) / PROJ_BN, (M + PROJ_BM - 1) / PROJ_BM);
    // warm up
    for (int i = 0; i < 3; ++i) gemm_nt_bias<<<grid, GEMM_THREADS>>>(a, b, bias, c, M, K, N);
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());

    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));
    CK(cudaEventRecord(t0));
    for (int i = 0; i < reps; ++i) gemm_nt_bias<<<grid, GEMM_THREADS>>>(a, b, bias, c, M, K, N);
    CK(cudaEventRecord(t1));
    CK(cudaEventSynchronize(t1));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, t0, t1));
    ms /= reps;

    const int Mp = grid.y * PROJ_BM, Np = grid.x * PROJ_BN;
    const double gflop = 2.0 * (double)Mp * Np * K * 1e-9;
    printf("%-14s M=%d K=%d N=%d  grid=(%u,%u)  %8.3f ms  %7.2f TFLOP/s\n",
           ABLATION, M, K, N, grid.x, grid.y, ms, gflop / ms * 1e-3);

    CK(cudaFree(a)); CK(cudaFree(b)); CK(cudaFree(bias)); CK(cudaFree(c));
    CK(cudaEventDestroy(t0)); CK(cudaEventDestroy(t1));
    return 0;
}
