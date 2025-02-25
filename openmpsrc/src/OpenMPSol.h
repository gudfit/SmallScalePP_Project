#pragma once
/* Function declarations only in header */
void matmul(const float *A, const float *B, float *C, int n, int k);
void matmul_ref(const float *A, const float *B, float *C_ref, int n, int k);
