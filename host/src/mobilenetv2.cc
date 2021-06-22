#include "mobilenetv2.h"

#include <CL/cl.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <numeric>
#include <vector>

#include "test_utils.h"

void RunMobileNetV2(cl_context context,
                    cl_command_queue command_queue,
                    cl_kernel conv_kernel,
                    cl_kernel depthwise_conv_kernel,
                    const int image_height,
                    const int image_width) {
}

