#include "OpenMPSol.h"
#include <algorithm>
#include <omp.h>
#include <vector>

/* Define block size for better cache utilization */
#define BLOCK_SIZE 64

/*
 * transpose
 *
 * Transpose B (of size k x n) into BT (of size n x k)
 *
 * @param pointer to B, BT (Transposed B)
 * sizes n,k
 * @return void
 */
void transpose(const double *B, double *BT, int k, int n) {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < k; i++)
    for (int j = 0; j < n; j++)
      BT[j * k + i] = B[i * n + j];
}

/*
 * matmul
 *
 * Computes C = A * B using the transposed version BT
 * Each element C[i][j] is computed as the dot product of
 * row i of A and row j of BT (transposed B)
 *
 * @param pointer to A,B,C
 * sizes n,k
 * A is n x k, B is k x n, C is n x n
 * @return void
 */
void matmul(const double *A, const double *B, double *C, int n, int k) {
  std::vector<double> BT_vector(n * k);
  double *BT = BT_vector.data();
  transpose(B, BT, k, n);

/* Initialize C to zeros first */
#pragma omp parallel for
  for (int i = 0; i < n * n; i++)
    C[i] = 0.0;

/* Tiling technique for better cache performance */
#pragma omp parallel for collapse(2) schedule(guided)
  for (int ii = 0; ii < n; ii += BLOCK_SIZE) {
    for (int jj = 0; jj < n; jj += BLOCK_SIZE) {
      /* Use local block boundaries to avoid repeated min operations */
      const int imax = std::min(ii + BLOCK_SIZE, n);
      const int jmax = std::min(jj + BLOCK_SIZE, n);

      for (int i = ii; i < imax; i++) {
        for (int j = jj; j < jmax; j++) {
          double sum = 0.0;

          for (int kk = 0; kk < k; kk += BLOCK_SIZE / 2) {
            const int kmax = std::min(kk + BLOCK_SIZE / 2, k);

/* Use vector operations when possible */
#pragma omp simd reduction(+ : sum)
            for (int p = kk; p < kmax; p++)
              sum += A[i * k + p] * BT[j * k + p];
          }

          C[i * n + j] = sum;
        }
      }
    }
  }
}

/*
 * matmul_ref
 *
 * Reference serial implementation of matrix multiplication.
 * Optimized with blocking/tiling for better cache usage.
 * Computes C_ref = A * B, where A is n x k and B is k x n.
 *
 * @param pointer to A,B,C_ref
 * sizes n,k
 * @return void
 */
void matmul_ref(const double *A, const double *B, double *C_ref, int n, int k) {
  /* Initialize C_ref to zeros */
  for (int i = 0; i < n * n; i++)
    C_ref[i] = 0.0;

  /* Blocked/tiled implementation for better cache utilization */
  for (int ii = 0; ii < n; ii += BLOCK_SIZE) {
    for (int jj = 0; jj < n; jj += BLOCK_SIZE) {
      const int imax = std::min(ii + BLOCK_SIZE, n);
      const int jmax = std::min(jj + BLOCK_SIZE, n);

      for (int kk = 0; kk < k; kk += BLOCK_SIZE) {
        const int kmax = std::min(kk + BLOCK_SIZE, k);

        for (int i = ii; i < imax; i++) {
          for (int j = jj; j < jmax; j++) {
            double sum = 0.0;
            for (int p = kk; p < kmax; p++)
              sum += A[i * k + p] * B[p * n + j];
            C_ref[i * n + j] += sum;
          }
        }
      }
    }
  }
}
