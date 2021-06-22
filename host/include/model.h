#ifndef HOST_INCLUDE_MODEL_H_
#define HOST_INCLUDE_MODEL_H_

#include <CL/cl.h>

#include "batchnorm.h"
#include "batchnorm_op.h"
#include "conv2d.h"
#include "conv2d_op.h"
#include "depthwise_conv2d.h"
#include "depthwise_conv2d_op.h"
#include "test_utils.h"

void RunModel(cl_context context,
              cl_command_queue command_queue,
              cl_kernel conv_kernel,
              cl_kernel batchnorm_kernel,
              std::vector<int> &tensor_shape,
              const std::vector<float> &in_data,
              std::vector<float> &out_data,
              const std::vector<float> &kernel_data,
              const std::vector<float> &weight_data,
              const std::vector<float> &bias_data);

void RunModelRef(std::vector<int> &tensor_shape,
                 std::vector<float> &in_data,
                 std::vector<float> &out_data,
                 const std::vector<float> &kernel_data,
                 const std::vector<float> &weight_data,
                 const std::vector<float> &bias_data);

#endif  // HOST_INCLUDE_MODEL_H_
