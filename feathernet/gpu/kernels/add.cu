__global__ void add(float* A, float* B, float* C, int num_elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_elements) {
        C[tid] = A[tid] + B[tid];
    }
}
