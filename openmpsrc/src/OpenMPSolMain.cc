#include "../includes/OpenMPSol.h"
#include "../includes/Transpose.h"

int main() {
  /* Range of n values and a set of k values */
  std::vector<int> ns = {512, 1024, 2048, 4096};
  std::vector<int> ks = {32, 48, 64, 96, 128};

  int max_threads = omp_get_max_threads();
  omp_set_num_threads(max_threads);

  /* Each combination of n and k */
  for (int n : ns) {
    for (int k : ks) {
      std::cout << "-------------------------------------------\n";
      std::cout << "Testing: n = " << n << ", k = " << k << "\n";

      /* Allocate memory on the heap */
      float *A = new float[n * k];
      float *B = new float[k * n];
      float *C = new float[n * n](); // Zero-initialized

      /* Initialize matrices A and B with random values using OpenMP */
      std::random_device rd;
      unsigned int seed = rd();

#pragma omp parallel sections
      {
#pragma omp section
        {
          std::mt19937 gen(seed);
          std::uniform_real_distribution<> dis(0.0f, 1.0f);
          for (int i = 0; i < n * k; i++)
            A[i] = dis(gen);
        }

#pragma omp section
        {
          std::mt19937 gen(seed + 1);
          std::uniform_real_distribution<> dis(0.0f, 1.0f);
          for (int i = 0; i < k * n; i++)
            B[i] = dis(gen);
        }
      }

      auto total_start = std::chrono::high_resolution_clock::now();
      auto t_start = std::chrono::high_resolution_clock::now();

      /* Compute C = A * B using the optimized OpenMP implementation */
      matmul(A, B, C, n, k);

      auto t_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> mult_time = t_end - t_start;
      auto total_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> total_time = total_end - total_start;

      double flops = 2.0 * k * n * n;
      double gflops = flops / (mult_time.count() * 1e9);
      std::cout << "Multiplication time:  " << mult_time.count() << " s\n";
      std::cout << "Total time:           " << total_time.count() << " s\n";
      std::cout << "Performance:          " << gflops << " GFLOPS\n";

      /* Compute the reference multiplication using a serial implementation */
      float *C_ref = new float[n * n](); // Zero-initialized

      auto ref_start = std::chrono::high_resolution_clock::now();
      matmul_ref(A, B, C_ref, n, k);
      auto ref_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> ref_time = ref_end - ref_start;

      /* Calculate error in parallel */
      double max_error = 0.0f;
#pragma omp parallel
      {
        double local_max = 0.0f;

#pragma omp for nowait
        for (int i = 0; i < n * n; i++) {
          double diff = std::fabs(C[i] - C_ref[i]);
          if (diff > local_max)
            local_max = diff;
        }

#pragma omp critical
        {
          if (local_max > max_error)
            max_error = local_max;
        }
      }

      const double tolerance = 1e-4f;
      std::cout << "Reference multiplication time: " << ref_time.count()
                << " s\n";
      std::cout << "Maximum difference with reference: " << max_error << "\n";

      if (max_error < tolerance)
        std::cout << "Correctness test PASSED.\n";
      else
        std::cout << "Correctness test FAILED.\n";

      /* Cleanup heap memory */
      delete[] A;
      delete[] B;
      delete[] C;
      delete[] C_ref;
    }
  }
  return 0;
}
