#include "OpenMPSol.h"
#include <algorithm>
#include <immintrin.h>
#include <limits>
#include <vector>

/* Optimized block sizes for Intel cache hierarchy */
#define L1_BLOCK_SIZE 32
#define L2_BLOCK_SIZE 128
#define L3_BLOCK_SIZE 512

/* Prefetch distances - tuned for Intel */
#define PREFETCH_DISTANCE_A 384
#define PREFETCH_DISTANCE_B 256

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
 * Optimized cache-friendly transpose for Intel processors
 * Uses blocking and explicit prefetching
 */
void transpose(const double *B, double *BT, int k, int n) {
#pragma omp parallel
  {
    /* Tiled transpose for better cache usage */
#pragma omp for collapse(2) schedule(static)
    for (int i = 0; i < k; i += L1_BLOCK_SIZE) {
      for (int j = 0; j < n; j += L1_BLOCK_SIZE) {
        const int imax = std::min(i + L1_BLOCK_SIZE, k);
        const int jmax = std::min(j + L1_BLOCK_SIZE, n);

        /* Process 4 elements at a time when possible */
        for (int ii = i; ii < imax; ii++) {
          int jj = j;

          /* Prefetch next row */
          if (ii + 1 < imax)
            _mm_prefetch((const char *)&B[(ii + 1) * n + j], _MM_HINT_T0);

          /* Process vectorizable part */
          for (; jj + 3 < jmax; jj += 4) {
            /* Load 4 elements from B */
            __m256d b_vec = _mm256_loadu_pd(&B[ii * n + jj]);

            /* Store them individually in BT */
            BT[(jj)*k + ii] = _mm256_cvtsd_f64(b_vec);
            BT[(jj + 1) * k + ii] =
                _mm256_cvtsd_f64(_mm256_permute_pd(b_vec, 1));
            __m128d high = _mm256_extractf128_pd(b_vec, 1);
            BT[(jj + 2) * k + ii] = _mm_cvtsd_f64(high);
            BT[(jj + 3) * k + ii] = _mm_cvtsd_f64(_mm_permute_pd(high, 1));
          }

          /* Handle remaining elements */
          for (; jj < jmax; jj++)
            BT[jj * k + ii] = B[ii * n + jj];
        }
      }
    }
  }
}

/*
 * Highly optimized matmul for Intel processors
 * Uses multi-level blocking, FMA instructions, prefetching, and multiple
 * accumulators
 */
void matmul(const double *A, const double *B, double *C, int n, int k) {
  /* Allocate aligned memory for BT for better SIMD performance */
  std::vector<double, aligned_allocator<double, ALIGN_SIZE>> BT_vector(n * k);
  double *BT = BT_vector.data();

  transpose(B, BT, k, n);

  /* Initialize C to zeros first */
#pragma omp parallel for simd schedule(static)
  for (int i = 0; i < n * n; i++)
    C[i] = 0.0;

  /* Three-level blocking for L1, L2, and L3 caches */
#pragma omp parallel
  {
#pragma omp for schedule(guided, 1)
    for (int i3 = 0; i3 < n; i3 += L3_BLOCK_SIZE) {
      for (int j3 = 0; j3 < n; j3 += L3_BLOCK_SIZE) {
        const int imax3 = std::min(i3 + L3_BLOCK_SIZE, n);
        const int jmax3 = std::min(j3 + L3_BLOCK_SIZE, n);

        /* L2 cache blocking */
        for (int i2 = i3; i2 < imax3; i2 += L2_BLOCK_SIZE) {
          for (int j2 = j3; j2 < jmax3; j2 += L2_BLOCK_SIZE) {
            const int imax2 = std::min(i2 + L2_BLOCK_SIZE, imax3);
            const int jmax2 = std::min(j2 + L2_BLOCK_SIZE, jmax3);

            /* L1 cache blocking */
            for (int i1 = i2; i1 < imax2; i1 += L1_BLOCK_SIZE) {
              for (int j1 = j2; j1 < jmax2; j1 += L1_BLOCK_SIZE) {
                const int imax1 = std::min(i1 + L1_BLOCK_SIZE, imax2);
                const int jmax1 = std::min(j1 + L1_BLOCK_SIZE, jmax2);

                /* Compute block */
                for (int i = i1; i < imax1; i++) {
                  for (int j = j1; j < jmax1; j++) {
                    if (j + 1 < jmax1) {
                      _mm_prefetch((const char *)&A[i * k], _MM_HINT_T0);
                      _mm_prefetch((const char *)&BT[(j + 1) * k], _MM_HINT_T0);
                    } else if (i + 1 < imax1) {
                      _mm_prefetch((const char *)&A[(i + 1) * k], _MM_HINT_T0);
                      _mm_prefetch((const char *)&BT[j * k], _MM_HINT_T0);
                    }

                    /* Multiple accumulators to hide FMA latency */
                    __m256d sum_vec1 = _mm256_setzero_pd();
                    __m256d sum_vec2 = _mm256_setzero_pd();
                    __m256d sum_vec3 = _mm256_setzero_pd();
                    __m256d sum_vec4 = _mm256_setzero_pd();

                    int p = 0;
                    for (; p <= k - 16; p += 16) {
                      _mm_prefetch(
                          (const char *)&A[i * k + p + PREFETCH_DISTANCE_A],
                          _MM_HINT_T0);
                      _mm_prefetch(
                          (const char *)&BT[j * k + p + PREFETCH_DISTANCE_B],
                          _MM_HINT_T0);

                      __m256d a_vec1 = _mm256_loadu_pd(&A[i * k + p]);
                      __m256d bt_vec1 = _mm256_loadu_pd(&BT[j * k + p]);
                      sum_vec1 = _mm256_fmadd_pd(a_vec1, bt_vec1, sum_vec1);

                      __m256d a_vec2 = _mm256_loadu_pd(&A[i * k + p + 4]);
                      __m256d bt_vec2 = _mm256_loadu_pd(&BT[j * k + p + 4]);
                      sum_vec2 = _mm256_fmadd_pd(a_vec2, bt_vec2, sum_vec2);

                      __m256d a_vec3 = _mm256_loadu_pd(&A[i * k + p + 8]);
                      __m256d bt_vec3 = _mm256_loadu_pd(&BT[j * k + p + 8]);
                      sum_vec3 = _mm256_fmadd_pd(a_vec3, bt_vec3, sum_vec3);

                      __m256d a_vec4 = _mm256_loadu_pd(&A[i * k + p + 12]);
                      __m256d bt_vec4 = _mm256_loadu_pd(&BT[j * k + p + 12]);
                      sum_vec4 = _mm256_fmadd_pd(a_vec4, bt_vec4, sum_vec4);
                    }

                    for (; p <= k - 4; p += 4) {
                      __m256d a_vec = _mm256_loadu_pd(&A[i * k + p]);
                      __m256d bt_vec = _mm256_loadu_pd(&BT[j * k + p]);
                      sum_vec1 = _mm256_fmadd_pd(a_vec, bt_vec, sum_vec1);
                    }

                    sum_vec1 = _mm256_add_pd(sum_vec1, sum_vec2);
                    sum_vec3 = _mm256_add_pd(sum_vec3, sum_vec4);
                    sum_vec1 = _mm256_add_pd(sum_vec1, sum_vec3);

                    __m128d sum_high = _mm256_extractf128_pd(sum_vec1, 1);
                    __m128d sum_low = _mm256_castpd256_pd128(sum_vec1);
                    __m128d sum_hl = _mm_add_pd(sum_high, sum_low);
                    __m128d sum_lh = _mm_permute_pd(sum_hl, 1);
                    __m128d result = _mm_add_pd(sum_hl, sum_lh);
                    double sum = _mm_cvtsd_f64(result);

                    for (int p_remainder = p; p_remainder < k; p_remainder++)
                      sum += A[i * k + p_remainder] * BT[j * k + p_remainder];
                    C[i * n + j] = sum;
                  }
                }
              }
            }
          }
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
  for (int ii = 0; ii < n; ii += L2_BLOCK_SIZE) {
    for (int jj = 0; jj < n; jj += L2_BLOCK_SIZE) {
      const int imax = std::min(ii + L2_BLOCK_SIZE, n);
      const int jmax = std::min(jj + L2_BLOCK_SIZE, n);

      for (int kk = 0; kk < k; kk += L2_BLOCK_SIZE) {
        const int kmax = std::min(kk + L2_BLOCK_SIZE, k);

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
