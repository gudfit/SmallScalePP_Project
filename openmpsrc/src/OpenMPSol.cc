#include "OpenMPSol.h"
#include <algorithm>
#include <immintrin.h>
#include <limits>
#include <vector>

/* Define block size for better cache utilization */
#define BLOCK_SIZE 64
#define ALIGN_SIZE 32

/* Class for aligned allocation */
template <class T, size_t N = 16> class aligned_allocator {
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
  // Fix: Split the directives
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
  /* Allocate aligned memory for BT for better SIMD performance */
  std::vector<double, aligned_allocator<double, ALIGN_SIZE>> BT_vector(n * k);
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

          /* Process chunks that are multiples of 4 using AVX */
          int p = 0;
          if (k >= 4) {
            __m256d sum_vec = _mm256_setzero_pd();

            for (; p <= k - 4; p += 4) {
              __m256d a_vec = _mm256_loadu_pd(&A[i * k + p]);
              __m256d bt_vec = _mm256_loadu_pd(&BT[j * k + p]);
              /* Multiply and add */
              sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(a_vec, bt_vec));
            }

            /* Horizontal sum of the 4 doubles in sum_vec */
            __m128d sum_high = _mm256_extractf128_pd(sum_vec, 1);
            __m128d sum_low = _mm256_castpd256_pd128(sum_vec);
            __m128d sum_hl = _mm_add_pd(sum_high, sum_low);
            __m128d sum_lh = _mm_permute_pd(sum_hl, 1);
            __m128d result = _mm_add_pd(sum_hl, sum_lh);
            sum += _mm_cvtsd_f64(result);
          }

          /* Process the remainder using scalar operations */
          for (int p_remainder = p; p_remainder < k; p_remainder++)
            sum += A[i * k + p_remainder] * BT[j * k + p_remainder];
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
#pragma omp parallel for
  for (int i = 0; i < n * n; i++)
    C_ref[i] = 0.0;

  /* Blocked/tiled implementation for better cache utilization */
#pragma omp parallel for collapse(3)
  for (int ii = 0; ii < n; ii += BLOCK_SIZE) {
    for (int jj = 0; jj < n; jj += BLOCK_SIZE) {
      const int imax = std::min(ii + BLOCK_SIZE, n);
      const int jmax = std::min(jj + BLOCK_SIZE, n);

      for (int kk = 0; kk < k; kk += BLOCK_SIZE) {
        const int kmax = std::min(kk + BLOCK_SIZE, k);

        for (int i = ii; i < imax; i++) {
          for (int j = jj; j < jmax; j++) {
            double sum = 0.0;
#pragma omp simd reduction(+ : sum)
            for (int p = kk; p < kmax; p++)
              sum += A[i * k + p] * B[p * n + j];

            C_ref[i * n + j] += sum;
          }
        }
      }
    }
  }
}
