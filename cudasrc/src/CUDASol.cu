#include "CUDASol.cuh"
#include <cuda_runtime.h>

#define TILE_WIDTH 32
#define BLOCK_DIM 32

#define TILE_WIDTH_PADDED (TILE_WIDTH + 1)

__global__ void matmul_kernel(const double *A, const double *B, double *C,
                              int n, int k) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < n && j < n) {
    double sum = 0.0;
    for (int p = 0; p < k; p++)
      sum += A[i * k + p] * B[p * n + j];

    C[i * n + j] = sum;
  }
}

__global__ void matmul_kernel_shared(const double *A, const double *B,
                                     double *C, int n, int k) {
  __shared__ double tile_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ double tile_B[TILE_WIDTH][TILE_WIDTH];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  double sum = 0.0;

  for (int t = 0; t < (k + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {

    int col_A = t * TILE_WIDTH + threadIdx.y;
    if (i < n && col_A < k)
      tile_A[threadIdx.x][threadIdx.y] = A[i * k + col_A];
    else
      tile_A[threadIdx.x][threadIdx.y] = 0.0;

    int row_B = t * TILE_WIDTH + threadIdx.x;
    if (j < n && row_B < k)
      tile_B[threadIdx.x][threadIdx.y] = B[row_B * n + j];
    else
      tile_B[threadIdx.x][threadIdx.y] = 0.0;

    __syncthreads();
#pragma unroll
    for (int p = 0; p < TILE_WIDTH; ++p)
      sum += tile_A[threadIdx.x][p] * tile_B[p][threadIdx.y];
    __syncthreads();
  }

  if (i < n && j < n)
    C[i * n + j] = sum;
}

__global__ void matmul_kernel_shared_padded(const double *A, const double *B,
                                            double *C, int n, int k) {
  __shared__ double tile_A[TILE_WIDTH][TILE_WIDTH_PADDED];
  __shared__ double tile_B[TILE_WIDTH_PADDED][TILE_WIDTH];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  double sum = 0.0;

  for (int t = 0; t < (k + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {

    int col_A = t * TILE_WIDTH + threadIdx.y;
    if (i < n && col_A < k)
      tile_A[threadIdx.x][threadIdx.y] = A[i * k + col_A];
    else
      tile_A[threadIdx.x][threadIdx.y] = 0.0;

    int row_B = t * TILE_WIDTH + threadIdx.x;
    if (j < n && row_B < k)
      tile_B[threadIdx.x][threadIdx.y] = B[row_B * n + j];
    else
      tile_B[threadIdx.x][threadIdx.y] = 0.0;

    __syncthreads();
#pragma unroll
    for (int p = 0; p < TILE_WIDTH; ++p)
      sum += tile_A[threadIdx.x][p] * tile_B[p][threadIdx.y];
    __syncthreads();
  }

  if (i < n && j < n)
    C[i * n + j] = sum;
}

void matmul_naive(const double *A, const double *B, double *C, int n, int k) {
  dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
               (n + blockDim.y - 1) / blockDim.y);

  matmul_kernel<<<gridDim, blockDim>>>(A, B, C, n, k);
  cudaDeviceSynchronize();
}

void matmul_shared(const double *A, const double *B, double *C, int n, int k) {
  dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
               (n + blockDim.y - 1) / blockDim.y);
  matmul_kernel_shared<<<gridDim, blockDim>>>(A, B, C, n, k);
  cudaDeviceSynchronize();
}

void matmul_shared_padded(const double *A, const double *B, double *C, int n,
                          int k) {
  dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
               (n + blockDim.y - 1) / blockDim.y);
  matmul_kernel_shared_padded<<<gridDim, blockDim>>>(A, B, C, n, k);
  cudaDeviceSynchronize();
}

/*
 * matmul_ref
 *
 * Reference serial implementation of matrix multiplication with OpenMP
 * parallelization. Computes C_ref = A * B, where A is n x k and B is k x n.
 *
 * @param pointer to A, B, C_ref
 * sizes n, k
 * @return void
 */
void matmul_ref(const double *A, const double *B, double *C_ref, int n, int k) {
/* Initialize C_ref to zeros in parallel */
#pragma omp parallel for
  for (int i = 0; i < n * n; i++)
    C_ref[i] = 0.0;

/* Blocked/tiled implementation with OpenMP parallelism */
#pragma omp parallel for collapse(3)
  for (int ii = 0; ii < n; ii += BLOCK_DIM) {
    for (int jj = 0; jj < n; jj += BLOCK_DIM) {
      for (int kk = 0; kk < k; kk += BLOCK_DIM) {
        for (int i = ii; i < std::min(ii + BLOCK_DIM, n); i++) {
          for (int j = jj; j < std::min(jj + BLOCK_DIM, n); j++) {
            double sum = 0.0;
/* Use SIMD vectorization for the innermost loop */
#pragma omp simd reduction(+ : sum)
            for (int p = kk; p < std::min(kk + BLOCK_DIM, k); p++)
              sum += A[i * k + p] * B[p * n + j];

            C_ref[i * n + j] += sum;
          }
        }
      }
    }
  }
}
