__kernel void Convolute(__global float * restrict in_data,
                        __global float * restrict out_data,
                        __constant float * restrict kernel_data,
                        int in_height,
                        int in_width,
                        int in_size,
                        int out_height,
                        int out_width,
                        int out_size,
                        int in_channels,
                        int out_channels,
                        int kernel_size,
                        int batch_kernel_size,
                        int stride,
                        int padding) {
  // x coordinate of the output pixel.
  const int oj = get_global_id(0);
  // y coordinate of the output pixel.
  const int oi = get_global_id(1);
  // Index of the output channel.
  const int oc = get_global_id(2);
  if ((oj >= out_width) || (oi >= out_height) || (oc >= out_channels)) {
    return;
  }

  const int kernel_radius = kernel_size / 2;
  int kernel_idx = oc * batch_kernel_size;
  const int padded_in_height = in_height + 2 * padding;
  const int padded_in_width = in_width + 2 * padding;
  int in_offset = 0;
  const int ii = oi * stride + kernel_radius;
  const int ij = oj * stride + kernel_radius;

  float acc = 0.f;
  // Accumulate over all input channels.
  for (int ic = 0; ic < in_channels; ic++) {
    for (int r = ii - kernel_radius; r <= ii + kernel_radius; r++) {
      for (int c = ij - kernel_radius; c <= ij + kernel_radius; c++) {
        if ((r >= padding) && (r < padded_in_height - padding) &&
            (c >= padding) && (c < padded_in_width - padding)) {
          acc += in_data[in_offset + (r - padding) * in_width + (c - padding)] *
                 kernel_data[kernel_idx];
        }
        kernel_idx++;
      }
    }
    in_offset += in_size;
  }
  out_data[oc * out_size + oi * out_width + oj] = acc;
}

