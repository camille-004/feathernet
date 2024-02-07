extern "C" __global__ void compute(float *a, float *b, float *result) {
    const int i = threadIdx.x;
    if (b[i] != 0) {
        result[i] = float(a[i]) / b[i];
    } else {
        result[i] = __int_as_float(0x7fc00000);
    }
}
