__kernel void vec_add(__global float* restrict a,
                      __global float* restrict b,
                      __global float* restrict c,
                      int vec_size) {
  int i = get_global_id(0);
  if (i >= vec_size) {
    return;
  }
  c[i] = a[i] + b[i];
}

