#ifndef HOST_INCLUDE_BATCHNORM_H_
#define HOST_INCLUDE_BATCHNORM_H_

#include <CL/cl.h>

#include <iostream>
#include <vector>

#include "memory_activation.h"

void RunBatchNormRef(std::vector<float> &tensor,
                     int batch,
                     int channels,
                     int channel_size,
                     float eps,
                     const std::vector<float> &weights,
                     const std::vector<float> &biases,
                     float relu);

void RunBatchNormRef(std::vector<float> &tensor,
                     const std::vector<int> &tensor_shape,
                     float eps,
                     const std::vector<float> &weights,
                     const std::vector<float> &biases,
                     float relu);

#endif  // HOST_INCLUDE_BATCHNORM_H_
