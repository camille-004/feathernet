__global__ void transpose(float *a, float *result, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height) {
        int idx_in = row * width + col;
        int idx_out = col * height + row;
        result[idx_out] = a[idx_in];
    }
}
