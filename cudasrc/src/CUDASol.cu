#include "CUDASol.cuh"
#include <algorithm>
#include <cuda_runtime.h>
#include <omp.h>

#define TILE_WIDTH 32
#define BLOCK_DIM 32

#define TILE_WIDTH_PADDED (TILE_WIDTH + 1)

__global__ void transpose_kernel(const float *B, float *BT, int k, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < k && j < n)
    BT[j * k + i] = B[i * n + j];
}

__global__ void matmul_kernel_naive(const float *A, const float *BT, float *C,
                                    int n, int k) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < n && j < n) {
    float sum = 0.0f;
    for (int p = 0; p < k; p++)
      sum += A[i * k + p] * BT[j * k + p];
    C[i * n + j] = sum;
  }
}

__global__ void matmul_kernel_shared(const float *A, const float *BT, float *C,
                                     int n, int k) {
  __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ float tile_BT[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;

  float sum = 0.0f;

  for (int t = 0; t < (k + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
    int a_col = t * TILE_WIDTH + tx;
    tile_A[ty][tx] = (row < n && a_col < k) ? A[row * k + a_col] : 0.0f;

    int bt_col = t * TILE_WIDTH + tx;
    tile_BT[ty][tx] = (col < n && bt_col < k) ? BT[col * k + bt_col] : 0.0f;

    __syncthreads();

#pragma unroll
    for (int p = 0; p < TILE_WIDTH; ++p)
      sum += tile_A[ty][p] * tile_BT[tx][p];

    __syncthreads();
  }

  if (row < n && col < n)
    C[row * n + col] = sum;
}

__global__ void matmul_kernel_shared_padded(const float *A, const float *BT,
                                            float *C, int n, int k) {
  __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH_PADDED];
  __shared__ float tile_BT[TILE_WIDTH][TILE_WIDTH_PADDED];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;

  float sum = 0.0f;

  for (int t = 0; t < (k + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
    int a_col = t * TILE_WIDTH + tx;
    tile_A[ty][tx] = (row < n && a_col < k) ? A[row * k + a_col] : 0.0f;

    int bt_col = t * TILE_WIDTH + tx;
    tile_BT[ty][tx] = (col < n && bt_col < k) ? BT[col * k + bt_col] : 0.0f;

    __syncthreads();

#pragma unroll
    for (int p = 0; p < TILE_WIDTH; ++p)
      sum += tile_A[ty][p] * tile_BT[tx][p];

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
/* Initialize C_ref to zeros in parallel */
#pragma omp parallel for
  for (int i = 0; i < n * n; i++)
    C_ref[i] = 0.0f;

/* Blocked/tiled implementation with OpenMP parallelism */
#pragma omp parallel for collapse(3)
  for (int ii = 0; ii < n; ii += BLOCK_DIM) {
    for (int jj = 0; jj < n; jj += BLOCK_DIM) {
      for (int kk = 0; kk < k; kk += BLOCK_DIM) {
        for (int i = ii; i < std::min(ii + BLOCK_DIM, n); i++) {
          for (int j = jj; j < std::min(jj + BLOCK_DIM, n); j++) {
            float sum = 0.0f;
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
