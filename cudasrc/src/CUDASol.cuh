#pragma once

// Host function declarations 
void matmul_naive(const double *A, const double *B, double *C, int n, int k);
void matmul_shared(const double *A, const double *B, double *C, int n, int k);
void matmul_shared_padded(const double *A, const double *B, double *C, int n, int k);
void matmul_ref(const double *A, const double *B, double *C_ref, int n, int k);

// Kernel function declarations
__global__ void matmul_kernel_naive(const double *A, const double *B, double *C, int n, int k);
__global__ void matmul_kernel_shared(const double *A, const double *B, double *C, int n, int k);
__global__ void matmul_kernel_shared_padded(const double *A, const double *B, double *C, int n, int k);