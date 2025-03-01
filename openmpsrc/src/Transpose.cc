#include "Transpose.h"
#include "OpenMPSol.h"

// --------------------- TRANSPOSE ------------------------- //
/*
 * transpose
 *
 * Transpose B (of size k x n) into BT (of size n x k)
 *
 * @param pointer to B, BT (Transposed B)
 * sizes n,k
 * @return void
 */

void transpose(const float *B, float *BT, int k, int n) {
  if (k <= 0 || n <= 0)
    return;

#pragma omp parallel for collapse(2)
  for (int i = 0; i < k; i += CACHELINE_PADDING) {
    for (int j = 0; j < n; j += CACHELINE_PADDING) {
      const int iend = std::min<int>(i + CACHELINE_PADDING, k);
      const int jend = std::min<int>(j + CACHELINE_PADDING, n);
      for (int ii = i; ii < iend; ++ii) {
        const int row_offset = ii * n;
        for (int jj = j; jj < jend; ++jj) {
          const int col_offset = jj * k;
          BT[col_offset + ii] = B[row_offset + jj];
        }
      }
    }
  }
}

// or bottom one is justt a tiny bit faster than top

/*
void transpose(const float *__restrict src, float *__restrict dst, int n,
                 int k) {
  const int TILE = 32;
#pragma omp parallel for collapse(2)
  for (int i = 0; i < n; i += TILE) {
    for (int j = 0; j < k; j += TILE) {
      int i_max = std::min(n, i + TILE);
      int j_max = std::min(k, j + TILE);
      // Transpose the tile [i..i_max) x [j..j_max)
      for (int ii = i; ii + 7 < i_max; ii += 8) {
        for (int jj = j; jj + 7 < j_max; jj += 8) {
          // Use the unaligned load instruction
          __m256 row0 = _mm256_loadu_ps(&src[ii * k + jj]);
          __m256 row1 = _mm256_loadu_ps(&src[(ii + 1) * k + jj]);
          __m256 row2 = _mm256_loadu_ps(&src[(ii + 2) * k + jj]);
          __m256 row3 = _mm256_loadu_ps(&src[(ii + 3) * k + jj]);
          __m256 row4 = _mm256_loadu_ps(&src[(ii + 4) * k + jj]);
          __m256 row5 = _mm256_loadu_ps(&src[(ii + 5) * k + jj]);
          __m256 row6 = _mm256_loadu_ps(&src[(ii + 6) * k + jj]);
          __m256 row7 = _mm256_loadu_ps(&src[(ii + 7) * k + jj]);
          __m256 t0 = _mm256_unpacklo_ps(row0, row1);
          __m256 t1 = _mm256_unpackhi_ps(row0, row1);
          __m256 t2 = _mm256_unpacklo_ps(row2, row3);
          __m256 t3 = _mm256_unpackhi_ps(row2, row3);
          __m256 t4 = _mm256_unpacklo_ps(row4, row5);
          __m256 t5 = _mm256_unpackhi_ps(row4, row5);
          __m256 t6 = _mm256_unpacklo_ps(row6, row7);
          __m256 t7 = _mm256_unpackhi_ps(row6, row7);
          __m256 tt0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
          __m256 tt1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
          __m256 tt2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
          __m256 tt3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
          __m256 tt4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
          __m256 tt5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
          __m256 tt6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
          __m256 tt7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));
          // Blend results to form transposed rows
          row0 = _mm256_permute2f128_ps(tt0, tt4, 0x20);
          row1 = _mm256_permute2f128_ps(tt1, tt5, 0x20);
          row2 = _mm256_permute2f128_ps(tt2, tt6, 0x20);
          row3 = _mm256_permute2f128_ps(tt3, tt7, 0x20);
          row4 = _mm256_permute2f128_ps(tt0, tt4, 0x31);
          row5 = _mm256_permute2f128_ps(tt1, tt5, 0x31);
          row6 = _mm256_permute2f128_ps(tt2, tt6, 0x31);
          row7 = _mm256_permute2f128_ps(tt3, tt7, 0x31);
          // Use unaligned store instructions
          _mm256_storeu_ps(&dst[jj * n + ii], row0);
          _mm256_storeu_ps(&dst[(jj + 1) * n + ii], row1);
          _mm256_storeu_ps(&dst[(jj + 2) * n + ii], row2);
          _mm256_storeu_ps(&dst[(jj + 3) * n + ii], row3);
          _mm256_storeu_ps(&dst[(jj + 4) * n + ii], row4);
          _mm256_storeu_ps(&dst[(jj + 5) * n + ii], row5);
          _mm256_storeu_ps(&dst[(jj + 6) * n + ii], row6);
          _mm256_storeu_ps(&dst[(jj + 7) * n + ii], row7);
        }
        // Remainder row processing
        for (int jj = j_max - (j_max % 8); jj < j_max; ++jj)
          for (int iii = 0; iii < 8 && (ii + iii) < i_max; ++iii)
            dst[jj * n + (ii + iii)] = src[(ii + iii) * k + jj];
      }
      // Balance row processing
      for (int ii = i_max - (i_max % 8); ii < i_max; ++ii)
        for (int jj = j; jj < j_max; ++jj)
          dst[jj * n + ii] = src[ii * k + jj];
    }
  }
}
*/
