#include "../includes/CUDASol.cuh"
#include "../includes/Transpose.cuh"

#include <cuda_runtime.h>

#define TILE_WIDTH 32
#define BLOCK_DIM 32

#define TILE_WIDTH_PADDED (TILE_WIDTH + 1)

__global__ void transpose_kernel(const float *__restrict__ B,
                                 float *__restrict__ BT, int k, int n) {
  __shared__ float block[BLOCK_DIM][BLOCK_DIM];

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

void transpose(const float *B, float *BT, int k, int n) {
  dim3 block(BLOCK_DIM, BLOCK_DIM);
  dim3 grid((k + BLOCK_DIM - 1) / BLOCK_DIM, (n + BLOCK_DIM - 1) / BLOCK_DIM);
  transpose_kernel<<<grid, block>>>(B, BT, k, n);
  cudaDeviceSynchronize();
}
