__global__ void compute(float *a, float *b, float *result) {
    const int i = threadIdx.x;
    result[i] = a[i] * b[i];
}
