#include "../includes/OpenMPSol.h"
#include "../includes/Transpose.h"

// --------------------- MATMUL ------------------------- //
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
void matmul(const float *A, const float *B, float *C, int n, int k) {
  std::vector<float, aligned_allocator<float, ALIGN_SIZE>> BT_vector(n * k);
  float *BT = BT_vector.data();
  transpose(B, BT, k, n);
#pragma omp parallel for
  for (int i = 0; i < n * n; i++)
    C[i] = 0.0f;
#pragma omp parallel for collapse(2) schedule(guided)
  for (int ii = 0; ii < n; ii += BLOCK_SIZE) {
    for (int jj = 0; jj < n; jj += BLOCK_SIZE) {
      const int imax = std::min(ii + BLOCK_SIZE, n);
      const int jmax = std::min(jj + BLOCK_SIZE, n);
      for (int i = ii; i < imax; i++) {
        for (int j = jj; j < jmax; j++) {
          float sum = 0.0f;
          int p = 0;
          if (k >= 8) {
            __m256 sum_vec = _mm256_setzero_ps();
            for (; p <= k - 8; p += 8) {
              __m256 a_vec = _mm256_loadu_ps(&A[i * k + p]);
              __m256 bt_vec = _mm256_loadu_ps(&BT[j * k + p]);
              sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(a_vec, bt_vec));
            }
            __m128 high = _mm256_extractf128_ps(sum_vec, 1);
            __m128 low = _mm256_castps256_ps128(sum_vec);
            __m128 sum128 = _mm_add_ps(high, low);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum += _mm_cvtss_f32(sum128);
          }
          for (; p < k; p++)
            sum += A[i * k + p] * BT[j * k + p];
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

void matmul_ref(const float *A, const float *B, float *C_ref, int n, int k) {
  std::vector<float> BT(n * k);
  transpose(B, BT.data(), k, n);
#pragma omp parallel for
  for (int i = 0; i < n * n; i++)
    C_ref[i] = 0.0f;
#pragma omp parallel for collapse(3)
  for (int ii = 0; ii < n; ii += BLOCK_SIZE) {
    for (int jj = 0; jj < n; jj += BLOCK_SIZE) {
      for (int kk = 0; kk < k; kk += BLOCK_SIZE) {
        const int imax = std::min(ii + BLOCK_SIZE, n);
        const int jmax = std::min(jj + BLOCK_SIZE, n);
        const int kmax = std::min(kk + BLOCK_SIZE, k);
        for (int i = ii; i < imax; i++) {
          for (int j = jj; j < jmax; j++) {
            float sum = 0.0f;
#pragma omp simd reduction(+ : sum)
            for (int p = kk; p < kmax; p++)
              sum += A[i * k + p] * BT[p + j * k];
            C_ref[i * n + j] += sum;
          }
        }
      }
    }
  }
}
// --------------------------- OLD CODE --------------------------------------

// void matmul_old(const float *A, const float *B, float *C, int n, int k) {
//   /* Allocate aligned memory for BT for better SIMD performance */
//   std::vector<float, aligned_allocator<float, ALIGN_SIZE>> BT_vector(n * k);
//   float *BT = BT_vector.data();
//   transpose(B, BT, k, n);
/* Initialize C to zeros first */
// #pragma omp parallel for
//   for (int i = 0; i < n * n; i++)
//     C[i] = 0.0f;
///* Tiling technique for better cache performance */
// #pragma omp parallel for collapse(2) schedule(guided)
//   for (int ii = 0; ii < n; ii += BLOCK_SIZE) {
//     for (int jj = 0; jj < n; jj += BLOCK_SIZE) {
//       /* Use local block boundaries to avoid repeated min operations */
//       const int imax = std::min(ii + BLOCK_SIZE, n);
//       const int jmax = std::min(jj + BLOCK_SIZE, n);
//       for (int i = ii; i < imax; i++) {
//         for (int j = jj; j < jmax; j++) {
//           float sum = 0.0f;
//           /* Process chunks that are multiples of 4 using AVX */
//           int p = 0;
//           if (k >= 8) {
//             __m256 sum_vec = _mm256_setzero_ps();
//             for (; p <= k - 8; p += 8) {
//               __m256 a_vec = _mm256_loadu_ps(&A[i * k + p]);
//               __m256 bt_vec = _mm256_loadu_ps(&BT[j * k + p]);
//               /* Multiply and add */
//               sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(a_vec, bt_vec));
//             }
//             __m128 high = _mm256_extractf128_ps(sum_vec, 1);
//             __m128 low = _mm256_castps256_ps128(sum_vec);
//             __m128 sum128 = _mm_add_ps(high, low);
//             sum128 = _mm_hadd_ps(sum128, sum128);
//             sum128 = _mm_hadd_ps(sum128, sum128);
//             sum += _mm_cvtss_f32(sum128);
//           }
//           /* Process the remainder using scalar operations */
//           for (int p_remainder = p; p_remainder < k; p_remainder++)
//             sum += A[i * k + p_remainder] * BT[j * k + p_remainder];
//           C[i * n + j] = sum;
//         }
//       }
//     }
//   }
// }

/* void matmul_ref(const float *A, const float *B, float *C_ref, int n, int k)
 * {
 *   for (int i = 0; i < n; i++) {
 *     for (int j = 0; j < n; j++) {
 *       float sum = 0.0;
 *       for (int p = 0; p < k; p++)
 *         sum += A[i * k + p] * B[p * n + j];
 *       C_ref[i * n + j] = sum;
 *     }
 *   }
 * }*/
