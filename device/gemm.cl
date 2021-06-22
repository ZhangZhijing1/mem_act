__kernel void gemm(__global float *A, __global float *B, __global float *C,
                   int wA, int wB, int wC) {
  int tx = get_global_id(0);  // 2D thread ID x
  int ty = get_global_id(1);  // 2D thread ID y
  float value = 0.f;

  for (int k = 0; k < wB; ++k) {
    value += A[ty * wA + k] * B[k * wC + tx];
  }
  C[ty * wA + tx] = value;
}
