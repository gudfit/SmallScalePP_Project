#include "OpenMPSol.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

int main() {
  /* Range of n values and a set of k values */
  std::vector<int> ns = {512, 1024, 2048, 4096};
  std::vector<int> ks = {32, 48, 64, 96, 128};

  /* Each combination of n and k*/
  for (int n : ns) {
    for (int k : ks) {
      std::cout << "-------------------------------------------\n";
      std::cout << "Testing: n = " << n << ", k = " << k << "\n";

      /* Allocating mem A: n x k, B: k x n, C: n x n */
      double *A = new double[n * k];
      if (!A) { std::cerr << "Allocation failed for A\n"; return 1; }
      double *B = new double[k * n];
      if (!B) { std::cerr << "Allocation failed for B\n"; return 1; }
      double *C = new double[n * n];
      if (!C) { std::cerr << "Allocation failed for C\n"; return 1; }

      /* Initialize matrices A and B with random values */
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<> dis(0.0, 1.0);

      for (int i = 0; i < n * k; i++)
        A[i] = dis(gen);
      for (int i = 0; i < k * n; i++)
        B[i] = dis(gen);

      auto total_start = std::chrono::high_resolution_clock::now();
      auto t_start = std::chrono::high_resolution_clock::now();
      /* Compute C = A * B using the transposed B */
      matmul(A, B, C, n, k);
      auto t_end = std::chrono::high_resolution_clock::now();

      std::chrono::duration<double> mult_time = t_end - t_start;
      auto total_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> total_time = total_end - total_start;

      double flops = 2.0 * k * n * n;
      double gflops = flops / (total_time.count() * 1e9);

      std::cout << "Multiplication time:  " << mult_time.count() << " s\n";
      std::cout << "Total time:           " << total_time.count() << " s\n";
      std::cout << "Performance:          " << gflops << " GFLOPS\n";

      // Correctness
      /* Compute the reference multiplication using a serial implementation*/
      double *C_ref = new double[n * n];
      if (!C_ref) { std::cerr << "Allocation failed for C_ref\n"; return 1; }
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
      const double tolerance = 1e-9;
      std::cout << "Reference multiplication time: " << ref_time.count()
                << " s\n";
      std::cout << "Maximum difference with reference: " << max_error << "\n";
      if (max_error < tolerance)
        std::cout << "Correctness test PASSED.\n";
      else
        std::cout << "Correctness test FAILED.\n";

      /* Free Mem*/

      delete[] A;
      delete[] B;
      delete[] C;
      delete[] C_ref;
    }
  }
  return 0;
}
