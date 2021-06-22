#ifndef HOST_INCLUDE_BATCHNORM_TEST_H_
#define HOST_INCLUDE_BATCHNORM_TEST_H_

#include <vector>

#include "batchnorm.h"
#include "batchnorm_op.h"
#include "test_utils.h"

using namespace std::chrono;

void RunBatchNormUnitTest(cl_context context,
                          cl_command_queue command_queue,
                          cl_kernel batchnorm_kernel,
                          const std::vector<int> &tensor_shape,
                          float relu,
                          bool enable_timing = true);

void RunBatchNormTests(cl_context context,
                       cl_command_queue command_queue,
                       cl_kernel batchnorm_kernel,
                       bool enable_timing = true);

#endif  // HOST_INCLUDE_BATCHNORM_TEST_H_

