void transpose(const float *B, float *BT, int k, int n);

__global__ void transpose_kernel(const float *__restrict__ B,
                                 float *__restrict__ BT, int k, int n);
