__global__ void matmul(float *a, float *b, float *result, int a_width, int a_height, int b_width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < a_height && col < b_width) {
        float sum = 0.0;
        for (int k = 0; k < a_width; ++k) {
            sum += a[row * a_width + k] * b[k * b_width + col];
        }
        result[row * b_width + col] = sum;
    }
}
