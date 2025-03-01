#include "CUDASol.cuh"
#include "Transpose.cuh"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>
#include <cuda_runtime.h>


void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      std::cerr << "CUDA error: " << msg << " : " << cudaGetErrorString(err) << std::endl;
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
      // ----------------- MEM  -------------------- //
      /* Pinned Memory (Faster transfer) */
      float *A, *B;
      cudaMallocHost(&A, n * k * sizeof(float));
      cudaMallocHost(&B, k * n * sizeof(float));

      /* Unity Streams */
      /*
      float *A, *B, *C;
      cudaMallocManaged(&A, n * k * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged(&B, k * n * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged(&C, n * n * sizeof(float), cudaMemAttachGlobal);
      */
      /*
      float *A = new float[n * k];
      float *B = new float[k * n];
      */
      float *C = new float[n * n];

      /* Initialize matrices A and B with random values */
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<> dis(0.0, 1.0);

      for (int i = 0; i < n * k; i++)
        A[i] = dis(gen);
      for (int i = 0; i < k * n; i++)
        B[i] = dis(gen);

      /* Allocate memory on the GPU for A and B */
      float *d_A, *d_B, *d_BT, *d_C;
      cudaMalloc(&d_A, n * k * sizeof(float));
      cudaMalloc(&d_B, k * n * sizeof(float));
      cudaMalloc(&d_BT, n * k * sizeof(float));
      cudaMalloc(&d_C, n * n * sizeof(float));
      checkCUDAError("cudaMalloc");

      /* Transfer A and B from CPU to GPU */
      cudaMemcpy(d_A, A, n * k * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_B, B, k * n * sizeof(float), cudaMemcpyHostToDevice);
      checkCUDAError("cudaMemcpy HostToDevice");

      auto total_start = std::chrono::high_resolution_clock::now();

      cudaMemset(d_C, 0, n * n * sizeof(float));
      checkCUDAError("cudaMemset d_C to zero"); 
      transpose(d_B, d_BT, k,n);
      auto t_start = std::chrono::high_resolution_clock::now();
      // ----------------- COMPUTE -------------------- //
      /* Compute C = A * B */
      // USE d_BT for transpose version
      matmul_shared_BT(d_A, d_BT, d_C, n, k);
      checkCUDAError("matmul");
      auto t_end = std::chrono::high_resolution_clock::now();

      std::chrono::duration<float> mult_time = t_end - t_start;
      auto total_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<float> total_time = total_end - total_start;

      cudaMemcpy(C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);
      checkCUDAError("cudaMemcpy DeviceToHost");

      float flops = 2.0 * k * n * n;
      float gflops = flops / (total_time.count() * 1e9);

      std::cout << "Multiplication time:  " << mult_time.count() << " s\n";
      std::cout << "Total time:           " << total_time.count() << " s\n";
      std::cout << "Performance:          " << gflops << " GFLOPS\n";

      // ----------------- CORRECTNESS CHECK -------------------- //
      /* Compute the reference multiplication using a serial implementation*/
      float *C_ref = new float[n * n];
      auto ref_start = std::chrono::high_resolution_clock::now();
      matmul_ref(A, B, C_ref, n, k);
      auto ref_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<float> ref_time = ref_end - ref_start;

      float max_error = 0.0;
      for (int i = 0; i < n * n; i++) {
        float diff = std::abs(C[i] - C_ref[i]);
        if (diff > max_error)
          max_error = diff;
      }
      const float tolerance = 1e-6;
      std::cout << "Reference multiplication time: " << ref_time.count()
                << " s\n";
      std::cout << "Maximum difference with reference: " << max_error << "\n";
      if (max_error < tolerance)
        std::cout << "Correctness test PASSED.\n";
      else
        std::cout << "Correctness test FAILED.\n";

      /* delete objects and mem */
      /*
      delete[] A;
      delete[] B;
      */

      
      delete[] C;
      /* Unified
      cudaFree(A);
      cudaFree(B);
      cudaFree(C);
      */

      /* Pinned Memory */
      cudaFreeHost(A);
      cudaFreeHost(B);

      delete[] C_ref;

      cudaFree(d_A);
      cudaFree(d_B);
      cudaFree(d_BT);
      cudaFree(d_C);
    }
  }
  return 0;
}
