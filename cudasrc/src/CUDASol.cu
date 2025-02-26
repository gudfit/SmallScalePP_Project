#include "CUDASol.cuh"
#include <algorithm>
#include <cuda_runtime.h>
#include <omp.h>

#define TILE_WIDTH 32
#define BLOCK_DIM 32

#define TILE_WIDTH_PADDED (TILE_WIDTH + 1)

__global__ void transpose_kernel(const float *__restrict__ B,
                                 float *__restrict__ BT, int k, int n) {
  __shared__ float block[BLOCK_DIM][BLOCK_DIM + 1];

  int x = blockIdx.x * BLOCK_DIM + threadIdx.x;
  int y = blockIdx.y * BLOCK_DIM + threadIdx.y;

  if (x < k && y < n) {
    unsigned int index_in = x * n + y;
    block[threadIdx.y][threadIdx.x] = B[index_in];
  }

  /* Ensure all threads have written to shared memory */
  __syncthreads();

  /* Transpose indices for writing back */
  /* Swap blockIdx.x and blockIdx.y */
  x = blockIdx.y * BLOCK_DIM + threadIdx.x;
  y = blockIdx.x * BLOCK_DIM + threadIdx.y;

  /* Write the transposed data back to global memory */
  if (x < n && y < k) {
    unsigned int index_out = x * k + y;
    BT[index_out] = block[threadIdx.x][threadIdx.y];
  }
}

__global__ void matmul_kernel_naive(const float *A, const float *BT, float *C,
                                    int n, int k) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  float sum = 0.0f;

  if (i < n && j < n) {
    for (int p = 0; p < k; p++)
      sum += A[i * k + p] * BT[j * k + p];
    C[i * n + j] = sum;
  }
}

__global__ void matmul_kernel_shared(const float *__restrict__ A,
                                     const float *__restrict__ BT,
                                     float *__restrict__ C, int n, int k) {
  /* shared mem tiles for A and BT */
  __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ float tile_BT[TILE_WIDTH][TILE_WIDTH];

  int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

  float sum = 0.0f;

  for (int t = 0; t < (k + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
    /* A=[n x k], row-major => A[row, (t*tile_width +threadIdx.x)]*/
    int a_col = t * TILE_WIDTH + threadIdx.x;
    /* NOTE: tile_A[y][x] => no bank conflicts if read across row */
    if (row < n && a_col < k)
      tile_A[threadIdx.y][threadIdx.x] = A[row * k + a_col];
    else
      tile_A[threadIdx.y][threadIdx.x] = 0.0f;

    int bt_col = t * TILE_WIDTH + threadIdx.y;
    if (col < n && bt_col < k)
      tile_BT[threadIdx.y][threadIdx.x] = BT[col * k + bt_col];
    else
      tile_BT[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();
#pragma unroll
    for (int p = 0; p < TILE_WIDTH; ++p)
      sum += tile_A[threadIdx.y][p] * tile_BT[p][threadIdx.x];

    __syncthreads();
  }

  if (row < n && col < n)
    C[row * n + col] = sum;
}

__global__ void matmul_kernel_shared_padded(const float *__restrict__ A,
                                            const float *__restrict__ BT,
                                            float *__restrict__ C, int n,
                                            int k) {
  __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH_PADDED];
  __shared__ float tile_BT[TILE_WIDTH][TILE_WIDTH_PADDED];

  int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
  float sum = 0.0f;

  for (int t = 0; t < (k + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
    int a_col = t * TILE_WIDTH + threadIdx.x;
    if (row < n && a_col < k)
      tile_A[threadIdx.y][threadIdx.x] = A[row * k + a_col];
    else
      tile_A[threadIdx.y][threadIdx.x] = 0.0f;

    int bt_col = t * TILE_WIDTH + threadIdx.y;
    if (col < n && bt_col < k)
      tile_BT[threadIdx.y][threadIdx.x] = BT[col * k + bt_col];
    else
      tile_BT[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();
#pragma unroll
    for (int p = 0; p < TILE_WIDTH; ++p)
      sum += tile_A[threadIdx.y][p] * tile_BT[p][threadIdx.x];

    __syncthreads();
  }

  if (row < n && col < n)
    C[row * n + col] = sum;
}

void transpose(const float *B, float *BT, int k, int n) {
  dim3 block(BLOCK_DIM, BLOCK_DIM);
  dim3 grid((k + BLOCK_DIM - 1) / BLOCK_DIM, (n + BLOCK_DIM - 1) / BLOCK_DIM);
  transpose_kernel<<<grid, block>>>(B, BT, k, n);
  cudaDeviceSynchronize();
}

void matmul_naive(const float *A, const float *B, float *C, int n, int k) {
  float *d_BT;
  cudaMalloc(&d_BT, n * k * sizeof(float));
  transpose(B, d_BT, k, n);

  dim3 block(BLOCK_DIM, BLOCK_DIM);
  dim3 grid((n + BLOCK_DIM - 1) / BLOCK_DIM, (n + BLOCK_DIM - 1) / BLOCK_DIM);
  matmul_kernel_naive<<<grid, block>>>(A, d_BT, C, n, k);
  cudaDeviceSynchronize();
  cudaFree(d_BT);
}

void matmul_shared(const float *A, const float *B, float *C, int n, int k) {
  float *d_BT;
  cudaMalloc(&d_BT, n * k * sizeof(float));
  transpose(B, d_BT, k, n);

  dim3 block(TILE_WIDTH, TILE_WIDTH);
  dim3 grid((n + TILE_WIDTH - 1) / TILE_WIDTH,
            (n + TILE_WIDTH - 1) / TILE_WIDTH);
  matmul_kernel_shared<<<grid, block>>>(A, d_BT, C, n, k);
  cudaDeviceSynchronize();
  cudaFree(d_BT);
}

void matmul_shared_padded(const float *A, const float *B, float *C, int n,
                          int k) {
  float *d_BT;
  cudaMalloc(&d_BT, n * k * sizeof(float));
  transpose(B, d_BT, k, n);

  dim3 block(TILE_WIDTH, TILE_WIDTH);
  dim3 grid((n + TILE_WIDTH - 1) / TILE_WIDTH,
            (n + TILE_WIDTH - 1) / TILE_WIDTH);
  matmul_kernel_shared_padded<<<grid, block>>>(A, d_BT, C, n, k);
  cudaDeviceSynchronize();
  cudaFree(d_BT);
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
void matmul_ref(const float *A, const float *B, float *C_ref, int n, int k) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      float sum = 0.0;
      for (int p = 0; p < k; p++)
        sum += A[i * k + p] * B[p * n + j];
      C_ref[i * n + j] = sum;
    }
  }
}
