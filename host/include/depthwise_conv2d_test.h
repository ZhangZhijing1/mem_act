#ifndef HOST_INCLUDE_DEPTHWISE_CONV2D_TEST_H_
#define HOST_INCLUDE_DEPTHWISE_CONV2D_TEST_H_

#include "depthwise_conv2d.h"
#include "depthwise_conv2d_op.h"
#include "test_utils.h"

using namespace std::chrono;

void RunDepthwiseConv2DUnitTest(cl_context context,
                                cl_command_queue command_queue,
                                cl_kernel kernel,
                                int in_height,
                                int in_width,
                                int in_channels,
                                int channel_multiplier,
                                int kernel_size,
                                int stride,
                                int padding,
                                bool enable_timing = false);

void RunDepthwiseConv2DTests(cl_context context,
                             cl_command_queue command_queue,
                             cl_kernel kernel,
                             bool enable_timing = false);

#endif  // HOST_INCLUDE_DEPTHWISE_CONV2D_TEST_H_
