__kernel void Convolute(__global float * restrict in_data,
                        __global float * restrict out_data,
                        __constant float * restrict kernel_data,
                        const int in_height,
                        const int in_width,
                        const int in_size,
                        const int out_height,
                        const int out_width,
                        const int out_size,
                        const int in_channels,
                        const int channel_multiplier,
                        const int kernel_size,
                        const int batch_kernel_size,
                        const int stride,
                        const int padding) {
  // x coordinate of the output pixel.
  const int oj = get_global_id(0);
  // y coordinate of the output pixel.
  const int oi = get_global_id(1);
  // Index of the input channel.
  const int ic = get_global_id(2);
  if ((oj >= out_width) || (oi >= out_height) || (ic >= in_channels)) {
    return;
  }

  const int kernel_radius = kernel_size / 2;
  int kernel_idx = ic * batch_kernel_size;
  const int padded_in_height = in_height + 2 * padding;
  const int padded_in_width = in_width + 2 * padding;
  const int in_offset = ic * in_size;
  int out_offset = ic * channel_multiplier * out_size;
  const int ii = oi * stride + kernel_radius;
  const int ij = oj * stride + kernel_radius;

  for (int oc = 0; oc < channel_multiplier; oc++) {
    float acc = 0.f;
    for (int r = ii - kernel_radius; r <= ii + kernel_radius; r++) {
      const int in_row_offset = in_offset + (r - padding) * in_width;
      for (int c = ij - kernel_radius; c <= ij + kernel_radius; c++) {
        if ((r >= padding) && (r < padded_in_height - padding) &&
            (c >= padding) && (c < padded_in_width - padding)) {
          acc += in_data[in_row_offset + (c - padding)] *
                 kernel_data[kernel_idx];
        }
        kernel_idx++;
      }
    }
    out_data[out_offset + oi * out_width + oj] = acc;
    out_offset += out_size;
  }
}

