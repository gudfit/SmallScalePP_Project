#include "OpenMPSol.h"
#include <algorithm>
#include <immintrin.h>
#include <limits>
#include <vector>

/* Define block size for better cache utilization */
#define BLOCK_SIZE 64
#define ALIGN_SIZE 32

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
#pragma omp parallel for collapse(2)
  for (int i = 0; i < k; i += ALIGN_SIZE) {
    for (int j = 0; j < n; j += ALIGN_SIZE) {
      const int iend = std::min(i + ALIGN_SIZE, k);
      const int jend = std::min(j + ALIGN_SIZE, n);
      for (int ii = i; ii < iend; ++ii)
        for (int jj = j; jj < jend; ++jj)
          BT[jj * k + ii] = B[ii * n + jj];
    }
  }
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
void matmul(const float *A, const float *B, float *C, int n, int k) {
  std::vector<float, aligned_allocator<float, ALIGN_SIZE>> BT_vector(n * k);
  float *BT = BT_vector.data();
  transpose(B, BT, k, n);

// Initialize C with zeros
#pragma omp parallel for
  for (int i = 0; i < n * n; ++i)
    C[i] = 0.0f;

#pragma omp parallel for collapse(2) schedule(static)
  for (int ii = 0; ii < n; ii += BLOCK_SIZE) {
    for (int jj = 0; jj < n; jj += BLOCK_SIZE) {
      const int imax = std::min(ii + BLOCK_SIZE, n);
      const int jmax = std::min(jj + BLOCK_SIZE, n);
      /* Temporary block storage for better cache locality */
      alignas(ALIGN_SIZE) float C_block[BLOCK_SIZE][BLOCK_SIZE] = {{0}};

      for (int kk = 0; kk < k; kk += BLOCK_SIZE) {
        const int kmax = std::min(kk + BLOCK_SIZE, k);
        for (int i = ii; i < imax; ++i) {
          for (int j = jj; j < jmax; ++j) {
            __m256 sum_vec = _mm256_setzero_ps();
            int p = kk;
            for (; p <= kmax - 8; p += 8) {
              const __m256 a = _mm256_loadu_ps(&A[i * k + p]);
              const __m256 b = _mm256_load_ps(&BT[j * k + p]);
              sum_vec = _mm256_fmadd_ps(a, b, sum_vec);
            }
            /* Horizontal sum */
            __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum_vec, 1),
                                       _mm256_castps256_ps128(sum_vec));
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            float sum = _mm_cvtss_f32(sum128);
            /* Process remainder */
            for (; p < kmax; ++p)
              sum += A[i * k + p] * BT[j * k + p];
            C_block[i - ii][j - jj] += sum;
          }
        }
      }
      /* Commit block to main matrix */
      for (int i = ii; i < imax; ++i)
        for (int j = jj; j < jmax; ++j)
          C[i * n + j] = C_block[i - ii][j - jj];
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
  /* Initialize C_ref to zeros */
#pragma omp parallel for
  for (int i = 0; i < n * n; i++)
    C_ref[i] = 0.0f;

  /* Blocked/tiled implementation for better cache utilization */
#pragma omp parallel for collapse(2)
  for (int ii = 0; ii < n; ii += BLOCK_SIZE) {
    for (int jj = 0; jj < n; jj += BLOCK_SIZE) {
      const int imax = std::min(ii + BLOCK_SIZE, n);
      const int jmax = std::min(jj + BLOCK_SIZE, n);

      for (int kk = 0; kk < k; kk += BLOCK_SIZE) {
        const int kmax = std::min(kk + BLOCK_SIZE, k);

        for (int i = ii; i < imax; i++) {
          for (int j = jj; j < jmax; j++) {
            float sum = 0.0f;
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
