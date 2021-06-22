#include "depthwise_conv2d.h"

void RunDepthwiseConv2DRef(const float *in_data,
                           float *out_data,
                           const float *kernel_data,
                           int in_height,
                           int in_width,
                           int in_channels,
                           int channel_multiplier,
                           int kernel_size,
                           int stride,
                           int padding) {
  int padded_in_height = in_height + 2 * padding;
  int padded_in_width = in_width + 2 * padding;
  int in_size = in_height * in_width;
  int in_offset = 0;

  int out_height = ((padded_in_height - kernel_size) / stride) + 1;
  int out_width = ((padded_in_width - kernel_size) / stride) + 1;
  int out_idx = 0;

  int kernel_offset = 0;
  int kernel_radius = kernel_size / 2;
  int batch_kernel_size = kernel_size * kernel_size;

  for (int ic = 0; ic < in_channels; ic++) {
    for (int oc = 0; oc < channel_multiplier; oc++) {
      for (int oi = 0; oi < out_height; oi++) {
        for (int oj = 0; oj < out_width; oj++) {
          int ii = oi * stride + kernel_radius;
          int ij = oj * stride + kernel_radius;
          int kernel_idx = kernel_offset;
          float acc = 0.f;
          for (int r = ii - kernel_radius; r <= ii + kernel_radius; r++) {
            for (int c = ij - kernel_radius; c <= ij + kernel_radius; c++) {
              if ((r >= padding) && (r < padded_in_height - padding) &&
                  (c >= padding) && (c < padded_in_width - padding)) {
                acc += in_data[in_offset + (r - padding) * in_width +
                               (c - padding)] *
                       kernel_data[kernel_idx];
              }
              kernel_idx++;
            }
          }
          out_data[out_idx++] = acc;
        }
      }
      kernel_offset += batch_kernel_size;
    }
    in_offset += in_size;
  }
}

void RunDepthwiseConv2DRef(const std::vector<float> &in_data,
                           std::vector<float> &out_data,
                           const std::vector<float> &kernel_data,
                           int in_height,
                           int in_width,
                           int in_channels,
                           int channel_multiplier,
                           int kernel_size,
                           int stride,
                           int padding) {
  RunDepthwiseConv2DRef(in_data.data(), out_data.data(), kernel_data.data(),
                        in_height, in_width, in_channels, channel_multiplier,
                        kernel_size, stride, padding);
}


