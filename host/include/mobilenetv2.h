#ifndef HOST_INCLUDE_MOBILENETV2_H_
#define HOST_INCLUDE_MOBILENETV2_H_

#include "conv2d.h"
#include "depthwise_conv2d.h"
#include "test_utils.h"

void RunMobileNetV2(cl_context context,
                    cl_command_queue command_queue,
                    cl_kernel conv_kernel,
                    cl_kernel depthwise_conv_kernel,
                    const int image_height,
                    const int image_width);

#endif  // HOST_INCLUDE_MOBILENETV2_H_
