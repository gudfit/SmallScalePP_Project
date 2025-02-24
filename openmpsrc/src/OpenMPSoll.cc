#include "OpenMPSol.h"
#include <omp.h>

/*
 * matmul
 *
 * Computes C = A * B using the transposed version BT
 * Each element C[i][j] is computed as the dot product of
 * row i of A and row j of B
 *
 * @param pointer to A,B,C
 * sizes n,k
 * A is n x k, B is n x k, C is n x n
 * @return void
 * */
void matmul(const double *A, const double *BT, double *C, int n, int k) {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
        double sum = 0.0;
        for (int p = 0; p < k; p++)
          sum += A[i * k + p] * BT[p * n + j];
        C[i * n + j] = sum;
    }
  }
}

/*
 * matmul_ref
 *
 *
 * Reference serial implementation of matrix multiplication.
 * Computes C_ref = A * B, where A is n x k and B is k x n.
 *
 * @param pointer to A,B,C_ref
 * sizes n,k
 * @return void
 */
void matmul_ref(const double *A, const double *B, double *C_ref, int n, int k) {
  // NOTE: B is stored in row-major order.
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0.0;
      for (int p = 0; p < k; p++)
        sum += A[i * k + p] * B[p * n + j];
      C_ref[i * n + j] = sum;
    }
  }
}
