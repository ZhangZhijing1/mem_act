#ifndef HOST_INCLUDE_DEPTHWISE_CONV2D_H_
#define HOST_INCLUDE_DEPTHWISE_CONV2D_H_

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

void RunDepthwiseConv2DRef(const float *in_data,
                           float *out_data,
                           const float *kernel_data,
                           int in_height,
                           int in_width,
                           int in_channels,
                           int channel_multiplier,
                           int kernel_size,
                           int stride,
                           int padding);

void RunDepthwiseConv2DRef(const std::vector<float> &in_data,
                           std::vector<float> &out_data,
                           const std::vector<float> &kernel_data,
                           int in_height,
                           int in_width,
                           int in_channels,
                           int channel_multiplier,
                           int kernel_size,
                           int stride,
                           int padding);

#endif  // HOST_INCLUDE_DEPTHWISE_CONV2D_H_

