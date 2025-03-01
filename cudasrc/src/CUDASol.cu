#include "../includes/CUDASol.cuh"
#include "../includes/Transpose.cuh"

#include <cuda_runtime.h>

#define TILE_WIDTH 32
#define BLOCK_DIM 32

#define TILE_WIDTH_PADDED (TILE_WIDTH + 1)

__global__ void matmul_kernel(const float *A, const float *B, float *C, int n,
                              int k) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < n && j < n) {
    float sum = 0.0;
    for (int p = 0; p < k; p++) {
      sum += A[i * k + p] * B[p * n + j];
    }
    C[i * n + j] = sum;
  }
}

__global__ void matmul_kernel_shared(const float *A, const float *B, float *C,
                                     int n, int k) {
  __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

  int i = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int j = blockIdx.x * TILE_WIDTH + threadIdx.x;

  float sum = 0.0;

  for (int t = 0; t < (k + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
    int col_A = t * TILE_WIDTH + threadIdx.x;
    if (i < n && col_A < k)
      tile_A[threadIdx.y][threadIdx.x] = A[i * k + col_A];
    else
      tile_A[threadIdx.y][threadIdx.x] = 0.0;

    int row_B = t * TILE_WIDTH + threadIdx.y;
    if (row_B < k && j < n)
      tile_B[threadIdx.y][threadIdx.x] = B[row_B * n + j];
    else
      tile_B[threadIdx.y][threadIdx.x] = 0.0;

    /* Correct synchronization position */
    __syncthreads();

    for (int p = 0; p < TILE_WIDTH; ++p)
      sum += tile_A[threadIdx.y][p] * tile_B[p][threadIdx.x];

    __syncthreads();
  }

  if (i < n && j < n)
    C[i * n + j] = sum;
}
__global__ void matmul_kernel_shared_padded(const float *A, const float *B,
                                            float *C, int n, int k) {
  __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH_PADDED];
  __shared__ float tile_B[TILE_WIDTH_PADDED][TILE_WIDTH];

  int i = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int j = blockIdx.x * TILE_WIDTH + threadIdx.x;

  float sum = 0.0;

  for (int t = 0; t < (k + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
    int col_A = t * TILE_WIDTH + threadIdx.x;
    if (i < n && col_A < k)
      tile_A[threadIdx.y][threadIdx.x] = A[i * k + col_A];
    else
      tile_A[threadIdx.y][threadIdx.x] = 0.0;

    int row_B = t * TILE_WIDTH + threadIdx.y;
    if (row_B < k && j < n)
      tile_B[threadIdx.y][threadIdx.x] = B[row_B * n + j];
    else
      tile_B[threadIdx.y][threadIdx.x] = 0.0;

    __syncthreads();

    for (int p = 0; p < TILE_WIDTH; ++p)
      sum += tile_A[threadIdx.y][p] * tile_B[p][threadIdx.x];

    __syncthreads();
  }

  if (i < n && j < n)
    C[i * n + j] = sum;
}

void matmul_naive(const float *A, const float *B, float *C, int n, int k) {
  dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
               (n + blockDim.y - 1) / blockDim.y);

  matmul_kernel<<<gridDim, blockDim>>>(A, B, C, n, k);
  cudaDeviceSynchronize();
}

void matmul_shared(const float *A, const float *B, float *C, int n, int k) {
  dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
               (n + blockDim.y - 1) / blockDim.y);
  matmul_kernel_shared<<<gridDim, blockDim>>>(A, B, C, n, k);
  cudaDeviceSynchronize();
}

void matmul_shared_padded(const float *A, const float *B, float *C, int n,
                          int k) {
  dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
               (n + blockDim.y - 1) / blockDim.y);
  matmul_kernel_shared_padded<<<gridDim, blockDim>>>(A, B, C, n, k);
  cudaDeviceSynchronize();
}
// -------------------- TESTING TRANSPOSE ------------------------
__global__ void matmul_kernel_shared_transpose(const float *A, const float *B,
                                               float *C, int n, int k) {
  __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

  int i = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int j = blockIdx.x * TILE_WIDTH + threadIdx.x;

  float sum = 0.0;

  for (int t = 0; t < (k + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
    int col_A = t * TILE_WIDTH + threadIdx.x;
    if (i < n && col_A < k)
      tile_A[threadIdx.y][threadIdx.x] = A[i * k + col_A];
    else
      tile_A[threadIdx.y][threadIdx.x] = 0.0;

    int row_B = t * TILE_WIDTH + threadIdx.y;
    /* Modify B's access to fit the transposed layout */
    if (row_B < k && j < n)
      tile_B[threadIdx.y][threadIdx.x] = B[j * k + row_B];
    else
      tile_B[threadIdx.y][threadIdx.x] = 0.0;

    __syncthreads();

    for (int p = 0; p < TILE_WIDTH; ++p)
      sum += tile_A[threadIdx.y][p] * tile_B[p][threadIdx.x];

    __syncthreads();
  }

  if (i < n && j < n)
    C[i * n + j] = sum;
}

void matmul_shared_BT(const float *A, const float *B, float *C, int n, int k) {
  dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
               (n + blockDim.y - 1) / blockDim.y);
  matmul_kernel_shared_transpose<<<gridDim, blockDim>>>(A, B, C, n, k);
  cudaDeviceSynchronize();
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
void matmul_ref(const float *A, const float *B, float *C_ref, int n, int k) {
  /* NOTE: B is stored in j-major order. */
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      float sum = 0.0;
      for (int p = 0; p < k; p++)
        sum += A[i * k + p] * B[p * n + j];
      C_ref[i * n + j] = sum;
    }
  }
}
