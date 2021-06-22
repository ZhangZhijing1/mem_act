#ifndef HOST_INCLUDE_CONV2D_H_
#define HOST_INCLUDE_CONV2D_H_

#include <CL/cl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <functional>
#include <iostream>
#include <numeric>
#include <ratio>
#include <string>
#include <vector>

#include "memory_activation.h"

void RunConv2DRef(const float *in_data,
                  float *out_data,
                  const float *kernel_data,
                  int in_height,
                  int in_width,
                  int in_channels,
                  int out_channels,
                  int kernel_size,
                  int stride,
                  int padding);

void RunConv2DRef(const std::vector<float> &in_data,
                  std::vector<float> &out_data,
                  const std::vector<float> &kernel_data,
                  int in_height,
                  int in_width,
                  int in_channels,
                  int out_channels,
                  int kernel_size,
                  int stride,
                  int padding);

void RunConv2DRef(const std::vector<float> &in_data,
                  std::vector<float> &out_data,
                  const std::vector<float> &kernel_data,
                  std::vector<int> &tensor_shape,
                  int out_channels,
                  int kernel_size,
                  int stride,
                  int padding);

#endif  // HOST_INCLUDE_CONV2D_H_

