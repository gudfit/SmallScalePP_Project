#pragma once

/* Host function declarations */
void matmul_naive(const float *A, const float *B, float *C, int n, int k);
void matmul_shared(const float *A, const float *B, float *C, int n, int k);
void matmul_shared_BT(const float *A, const float *B, float *C, int n, int k);
void matmul_shared_padded(const float *A, const float *B, float *C, int n, int k);
void matmul_ref(const float *A, const float *B, float *C_ref, int n, int k);

/* Kernel function declarations */
__global__ void matmul_kernel_naive(const float *A, const float *B, float *C, int n, int k);
__global__ void matmul_kernel_shared(const float *A, const float *B, float *C, int n, int k);
__global__ void matmul_kernel_shared_transpose(const float *A, const float *B, float *C, int n, int k);
__global__ void matmul_kernel_shared_padded(const float *A, const float *B, float *C, int n, int k);
