#include "CUDASol.cuh"
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <random>

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << msg << " : " << cudaGetErrorString(err)
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

int main() {
  /* Range of n values and a set of k values */
  std::vector<int> ns = {512, 1024, 2048, 4096};
  std::vector<int> ks = {32, 48, 64, 96, 128};

  for (int n : ns) {
    for (int k : ks) {
      std::cout << "-------------------------------------------\n";
      std::cout << "Testing: n = " << n << ", k = " << k << "\n";

      float *A = new float[n * k];
      float *B = new float[k * n];
      float *C = new float[n * n];

      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<float> dis(0.0f, 1.0f);

      for (int i = 0; i < n * k; i++)
        A[i] = dis(gen);
      for (int i = 0; i < k * n; i++)
        B[i] = dis(gen);

      float *d_A, *d_B, *d_C;
      cudaMalloc(&d_A, n * k * sizeof(float));
      cudaMalloc(&d_B, k * n * sizeof(float));
      cudaMalloc(&d_C, n * n * sizeof(float));
      checkCUDAError("cudaMalloc");

      cudaMemcpy(d_A, A, n * k * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_B, B, k * n * sizeof(float), cudaMemcpyHostToDevice);
      checkCUDAError("cudaMemcpy HostToDevice");

      auto start = std::chrono::high_resolution_clock::now();
      cudaMemset(d_C, 0, n * n * sizeof(float));
      checkCUDAError("cudaMemset");

      auto mult_start = std::chrono::high_resolution_clock::now();
      matmul_naive(d_A, d_B, d_C, n, k);
      checkCUDAError("matmul_naive");

      auto mult_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> mult_time = mult_end - mult_start;

      cudaMemcpy(C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);
      checkCUDAError("cudaMemcpy DeviceToHost");

      double gflops = (2.0 * k * n * n) / (mult_time.count() * 1e9);
      std::cout << "Multiplication time: " << mult_time.count() << " s\n";
      std::cout << "Performance: " << gflops << " GFLOPS\n";

      // Correctness
      /* Compute the reference multiplication using a serial implementation*/
      float *C_ref = new float[n * n];
      auto ref_start = std::chrono::high_resolution_clock::now();
      matmul_ref(A, B, C_ref, n, k);
      auto ref_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> ref_time = ref_end - ref_start;

      float max_error = 0.0f;
      for (int i = 0; i < n * n; i++)
        max_error = std::max(max_error, std::fabs(C[i] - C_ref[i]));

      const float tolerance = 1e-6f;
      std::cout << "Max error: " << max_error
                << (max_error < tolerance ? " PASSED" : " FAILED") << "\n";

      /* Free mem */
      delete[] A;
      delete[] B;
      delete[] C;
      delete[] C_ref;
      cudaFree(d_A);
      cudaFree(d_B);
      cudaFree(d_C);
    }
  }
  return 0;
}
