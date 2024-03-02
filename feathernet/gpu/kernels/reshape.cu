extern "C"
__global__ void reshape(const float *input, float *output, int totalElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalElements) {
        output[idx] = input[idx];
    }
}
