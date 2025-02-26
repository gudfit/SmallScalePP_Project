#include "CUDASol.cuh"
#include <array>
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
  std::vector<int> ns = {512, 1024, 2048, 4096};
  std::vector<int> ks = {32, 48, 64, 96, 128};

  for (int n : ns) {
    for (int k : ks) {
      std::cout << "-------------------------------------------\n";
      std::cout << "Testing: n = " << n << ", k = " << k << "\n";

      /* Pinned Memory (Faster transfer) */
      /* 
      float *A, *B;
      cudaMallocHost(&A, n * k * sizeof(float));
      cudaMallocHost(&B, k * n * sizeof(float)); 
      */

      /* Unity Streams */
      /*
      float *A, *B, *C;
      cudaMallocManaged(&A, n * k * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged(&B, k * n * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged(&C, n * n * sizeof(float), cudaMemAttachGlobal); 
      */

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

      cudaMemset(d_C, 0, n * n * sizeof(float));
      checkCUDAError("cudaMemset");

      cudaEvent_t start_event, stop_event;
      cudaEventCreate(&start_event);
      checkCUDAError("cudaEventCreate start");
      cudaEventCreate(&stop_event);
      checkCUDAError("cudaEventCreate stop");

      cudaEventRecord(start_event);
      checkCUDAError("cudaEventRecord start");
      matmul_naive(d_A, d_B, d_C, n, k);
      checkCUDAError("matmul_naive");
      cudaEventRecord(stop_event);
      checkCUDAError("cudaEventRecord stop");

      cudaEventSynchronize(stop_event);
      checkCUDAError("cudaEventSynchronize");

      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start_event, stop_event);
      checkCUDAError("cudaEventElapsedTime");
      double mult_time = milliseconds / 1e3;

      cudaMemcpy(C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);
      checkCUDAError("cudaMemcpy DeviceToHost");

      double gflops = (2.0 * k * n * n) / (mult_time * 1e9);
      std::cout << "Multiplication time: " << mult_time << " s\n";
      std::cout << "Performance: " << gflops << " GFLOPS\n";

      float *C_ref = new float[n * n];
      matmul_ref(A, B, C_ref, n, k);

      float max_error = 0.0f;
      for (int i = 0; i < n * n; i++)
        max_error = std::max(max_error, std::fabs(C[i] - C_ref[i]));

      const float tolerance = 1e-6f;
      std::cout << "Max error: " << max_error
                << (max_error < tolerance ? " PASSED" : " FAILED") << "\n";

      /* delete objects and mem */
      
      delete[] A;
      delete[] B;
      delete[] C;

      /* Unified
      cudaFree(A);
      cudaFree(B);
      cudaFree(C);
      */

      /* Pinned Memory
      cudaFreeHost(A);
      cudaFreeHost(B);
      */

      delete[] C_ref;

      cudaFree(d_A);
      cudaFree(d_B);
      cudaFree(d_C);
      cudaEventDestroy(start_event);
      cudaEventDestroy(stop_event);
    }
  }
  return 0;
}
