#include "CUDASol.cuh"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

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
      
      /* Use std::vector instead of 'new' for CPU memory allocation */
      std::vector<double> A(n * k);
      std::vector<double> B(k * n);
      std::vector<double> C(n * n);
      
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
      cudaMemcpy(d_A, A.data(), n * k * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_B, B.data(), k * n * sizeof(double), cudaMemcpyHostToDevice);
      checkCUDAError("cudaMemcpy HostToDevice");
      
      /* CUDA events */
      cudaEvent_t total_start, total_end, mult_start, mult_end;
      cudaEventCreate(&total_start);
      cudaEventCreate(&total_end);
      cudaEventCreate(&mult_start);
      cudaEventCreate(&mult_end);
      
      cudaEventRecord(total_start);
      
      cudaMemset(d_C, 0, n * n * sizeof(double));
      checkCUDAError("cudaMemset d_C to zero");
      
      cudaEventRecord(mult_start);
      
      /* Compute C = A * B */
      matmul_naive(d_A, d_B, d_C, n, k);
      checkCUDAError("matmul");
      
      cudaEventRecord(mult_end);
      cudaEventSynchronize(mult_end);
      
      cudaEventRecord(total_end);
      cudaEventSynchronize(total_end);
      
      float mult_time, total_time;
      cudaEventElapsedTime(&mult_time, mult_start, mult_end);
      cudaEventElapsedTime(&total_time, total_start, total_end);
      
      mult_time /= 1000.0f;
      total_time /= 1000.0f;
      
      cudaMemcpy(C.data(), d_C, n * n * sizeof(double), cudaMemcpyDeviceToHost);
      checkCUDAError("cudaMemcpy DeviceToHost");
      
      double flops = 2.0 * k * n * n;
      double gflops = flops / (mult_time * 1e9);
      
      std::cout << "Multiplication time:  " << mult_time << " s\n";
      std::cout << "Total time:           " << total_time << " s\n";
      std::cout << "Performance:          " << gflops << " GFLOPS\n";
      
      // Correctness
      /* Compute the reference multiplication using a serial implementation*/
      std::vector<double> C_ref(n * n);
      
      auto ref_start = std::chrono::high_resolution_clock::now();
      matmul_ref(A.data(), B.data(), C_ref.data(), n, k);
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
      
      /* Free memory and destroy events */
      cudaFree(d_A);
      cudaFree(d_B);
      cudaFree(d_C);
      
      cudaEventDestroy(total_start);
      cudaEventDestroy(total_end);
      cudaEventDestroy(mult_start);
      cudaEventDestroy(mult_end);
    }
  }
  
  return 0;
}
