#include "OpenMPSol.h"
#include <algorithm>
#include <immintrin.h>
#include <limits>
#include <vector>

/* Define block size for better cache utilization */
#define BLOCK_SIZE 64
#define ALIGN_SIZE 64
#define CACHELINE_PADDING (ALIGN_SIZE / sizeof(float))

/* Class for aligned allocation */
template <class T, size_t N = 32> class aligned_allocator {
public:
  typedef T value_type;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef T *pointer;
  typedef const T *const_pointer;
  typedef T &reference;
  typedef const T &const_reference;

  aligned_allocator() {}
  aligned_allocator(const aligned_allocator &) {}
  template <class U> aligned_allocator(const aligned_allocator<U, N> &) {}
  ~aligned_allocator() {}

  pointer address(reference r) { return &r; }
  const_pointer address(const_reference r) const { return &r; }

  pointer allocate(size_type n) {
    void *p;
    if (posix_memalign(&p, N, n * sizeof(T)) != 0)
      throw std::bad_alloc();
    return static_cast<pointer>(p);
  }

  void deallocate(pointer p, size_type) { free(p); }

  size_type max_size() const {
    return std::numeric_limits<size_type>::max() / sizeof(T);
  }

  template <class U, class... Args> void construct(U *p, Args &&...args) {
    ::new ((void *)p) U(std::forward<Args>(args)...);
  }

  template <class U> void destroy(U *p) { p->~U(); }

  template <class U> struct rebind {
    typedef aligned_allocator<U, N> other;
  };
};

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

// --------------------- MATMUL ------------------------- //

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
