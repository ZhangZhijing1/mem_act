#include "conv2d.h"

void RunConv2DRef(const float *in_data,
                  float *out_data,
                  const float *kernel_data,
                  int in_height,
                  int in_width,
                  int in_channels,
                  int out_channels,
                  int kernel_size,
                  int stride,
                  int padding) {
  int padded_in_height = in_height + 2 * padding;
  int padded_in_width = in_width + 2 * padding;
  int in_size = in_height * in_width;
  int out_height = ((padded_in_height - kernel_size) / stride) + 1;
  int out_width = ((padded_in_width - kernel_size) / stride) + 1;
  int out_idx = 0;
  int kernel_offset = 0;
  int kernel_radius = kernel_size / 2;
  int batch_kernel_size = in_channels * kernel_size * kernel_size;

  for (int oc = 0; oc < out_channels; oc++) {
    for (int oi = 0; oi < out_height; oi++) {
      for (int oj = 0; oj < out_width; oj++) {
        float acc = 0.f;
        int in_offset = 0;
        int kernel_idx = kernel_offset;
        int ii = oi * stride + kernel_radius;
        int ij = oj * stride + kernel_radius;
        for (int ic = 0; ic < in_channels; ic++) {
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
          in_offset += in_size;
        }
        out_data[out_idx++] = acc;
      }
    }
    kernel_offset += batch_kernel_size;
  }
}

void RunConv2DRef(const std::vector<float> &in_data,
                  std::vector<float> &out_data,
                  const std::vector<float> &kernel_data,
                  int in_height,
                  int in_width,
                  int in_channels,
                  int out_channels,
                  int kernel_size,
                  int stride,
                  int padding) {
  RunConv2DRef(in_data.data(), out_data.data(), kernel_data.data(), in_height,
               in_width, in_channels, out_channels, kernel_size, stride,
               padding);
}

void RunConv2DRef(const std::vector<float> &in_data,
                  std::vector<float> &out_data,
                  const std::vector<float> &kernel_data,
                  std::vector<int> &tensor_shape,
                  int out_channels,
                  int kernel_size,
                  int stride,
                  int padding) {
  const int in_height = tensor_shape[2];
  const int in_width = tensor_shape[3];
  RunConv2DRef(in_data.data(), out_data.data(), kernel_data.data(), in_height,
               in_width, tensor_shape[1], out_channels, kernel_size, stride,
               padding);
  const int out_height = ((in_height + 2 * padding - kernel_size) / stride) + 1;
  const int out_width = ((in_width + 2 * padding - kernel_size) / stride) + 1;
  tensor_shape[1] = out_channels;
  tensor_shape[2] = out_height;
  tensor_shape[3] = out_width;
}

