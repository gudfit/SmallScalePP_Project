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

  /* Each combination of n and k*/
  for (int n : ns) {
    for (int k : ks) {
      std::cout << "-------------------------------------------\n";
      std::cout << "Testing: n = " << n << ", k = " << k << "\n";

      /* Use dynamic allocation since array sizes are determined at runtime */
      double *A = new double[n * k];
      double *B = new double[k * n];
      double *C = new double[n * n];

      /* Initialize matrices A and B with random values */
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<> dis(0.0, 1.0);

      for (int i = 0; i < n * k; i++)
        A[i] = dis(gen);
      for (int i = 0; i < k * n; i++)
        B[i] = dis(gen);

      /* Allocate memory on the GPU for A and B */
      double *d_A, *d_B, *d_C;
      cudaMalloc(&d_A, n * k * sizeof(double));
      cudaMalloc(&d_B, k * n * sizeof(double));
      cudaMalloc(&d_C, n * n * sizeof(double));
      checkCUDAError("cudaMalloc");

      /* Transfer A and B from CPU to GPU */
      cudaMemcpy(d_A, A, n * k * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_B, B, k * n * sizeof(double), cudaMemcpyHostToDevice);
      checkCUDAError("cudaMemcpy HostToDevice");

      auto total_start = std::chrono::high_resolution_clock::now();
      cudaMemset(d_C, 0, n * n * sizeof(double));
      checkCUDAError("cudaMemset d_C to zero");

      auto t_start = std::chrono::high_resolution_clock::now();
      /* Compute C = A * B */
      matmul_naive(d_A, d_B, d_C, n, k);
      checkCUDAError("matmul");

      auto t_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> mult_time = t_end - t_start;

      auto total_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> total_time = total_end - total_start;

      cudaMemcpy(C, d_C, n * n * sizeof(double), cudaMemcpyDeviceToHost);
      checkCUDAError("cudaMemcpy DeviceToHost");

      double flops = 2.0 * k * n * n;
      double gflops = flops / (mult_time.count() * 1e9);

      std::cout << "Multiplication time:  " << mult_time.count() << " s\n";
      std::cout << "Total time:           " << total_time.count() << " s\n";
      std::cout << "Performance:          " << gflops << " GFLOPS\n";

      // Correctness
      /* Compute the reference multiplication using a serial implementation*/
      double *C_ref = new double[n * n];

      auto ref_start = std::chrono::high_resolution_clock::now();
      matmul_ref(A, B, C_ref, n, k);
      auto ref_end = std::chrono::high_resolution_clock::now();

      std::chrono::duration<double> ref_time = ref_end - ref_start;

      double max_error = 0.0;
      for (int i = 0; i < n * n; i++) {
        double diff = std::abs(C[i] - C_ref[i]);
        if (diff > max_error)
          max_error = diff;
      }

      const double tolerance = 1e-13;
      std::cout << "Reference multiplication time: " << ref_time.count()
                << " s\n";
      std::cout << "Maximum difference with reference: " << max_error << "\n";

      if (max_error < tolerance)
        std::cout << "Correctness test PASSED.\n";
      else
        std::cout << "Correctness test FAILED.\n";

      /* Free GPU mem */
      cudaFree(d_A);
      cudaFree(d_B);
      cudaFree(d_C);

      /* Free CPU mem */
      delete[] A;
      delete[] B;
      delete[] C;
      delete[] C_ref;
    }
  }
  return 0;
}
