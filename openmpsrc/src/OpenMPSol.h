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

/* Function declarations */
void matmul(const float *A, const float *B, float *C, int n, int k);
void matmul_ref(const float *A, const float *B, float *C_ref, int n, int k);
