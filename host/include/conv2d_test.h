#ifndef HOST_INCLUDE_CONV2D_TEST_H_
#define HOST_INCLUDE_CONV2D_TEST_H_

#include <chrono>
#include <ctime>
#include <ratio>

#include "conv2d.h"
#include "conv2d_op.h"
#include "test_utils.h"

using namespace std::chrono;

void RunConv2DUnitTest(cl_context context,
                       cl_command_queue command_queue,
                       cl_kernel kernel,
                       const int in_height,
                       const int in_width,
                       const int in_channels,
                       const int out_channels,
                       const int kernel_size,
                       const int stride,
                       const int padding,
                       bool enable_timing = false);

void RunConv2DTests(cl_context context,
                    cl_command_queue command_queue,
                    cl_kernel kernel,
                    bool enable_timing = false);

#endif  // HOST_INCLUDE_CONV2D_TEST_H_
