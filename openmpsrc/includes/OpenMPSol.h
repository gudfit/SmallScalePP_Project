#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <omp.h>
#include <random>
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

/* Function declarations */
void matmul(const float *A, const float *B, float *C, int n, int k);
void matmul_ref(const float *A, const float *B, float *C_ref, int n, int k);
