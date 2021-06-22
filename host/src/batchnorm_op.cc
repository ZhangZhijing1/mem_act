#include "batchnorm_op.h"

#include <algorithm>
#include <functional>
#include <numeric>

#include "memory_activation.h"

BatchNormOp::BatchNormOp(int num_features, float eps, float relu,
                         cl_kernel *kernel,
                         cl_command_queue *command_queue,
                         cl_mem *weights_buf,
                         cl_mem *biases_buf,
                         cl_mem *tensor_buf)
    : num_features_(num_features),
      eps_(eps),
      relu_(relu),
      kernel_(kernel),
      command_queue_(command_queue),
      weights_buf_(weights_buf),
      biases_buf_(biases_buf),
      tensor_buf_(tensor_buf) {}

void BatchNormOp::SetTensorBuffer(cl_mem *buf) {
  tensor_buf_ = buf;
}

void BatchNormOp::SetWeightBuffer(cl_mem *buf) {
  weights_buf_ = buf;
}

void BatchNormOp::SetBiasBuffer(cl_mem *buf) {
  biases_buf_ = buf;
}


void BatchNormOp::Run(const std::vector<int> &shape, bool blocking) {
  const static cl_uint wg_dim = 1;
  const static cl_uint wg_size = 32;

  ASSERT(shape.size() == 4, "Only accepts 4D input");
  ASSERT(shape[1] == num_features_, "Number of input channels");
  ASSERT(weights_buf_ != nullptr, "weight buffer is null");
  ASSERT(biases_buf_ != nullptr, "bias buffer is null");
  ASSERT(tensor_buf_ != nullptr, "tensor buffer is null");

  const int batch = shape[0];
  const int channels = shape[1];
  const int channel_size = shape[2] * shape[3];

  cl_int status;
  const cl_uint total_work_items = RoundUp(channels, wg_size);
  const std::size_t global_size[wg_dim] = {
    static_cast<std::size_t>(total_work_items)
  };
  const std::size_t local_size[wg_dim] = {
    static_cast<std::size_t>(wg_size)
  };

  cl_uint arg_idx = 0;
  status = clSetKernelArg(*kernel_, arg_idx++, sizeof(cl_mem), tensor_buf_);
  ASSERT(status == CL_SUCCESS, "Failed to set the argument " + std::to_string(arg_idx));
  status = clSetKernelArg(*kernel_, arg_idx++, sizeof(int), &batch);
  ASSERT(status == CL_SUCCESS, "Failed to set the argument " + std::to_string(arg_idx));
  status = clSetKernelArg(*kernel_, arg_idx++, sizeof(int), &channels);
  ASSERT(status == CL_SUCCESS, "Failed to set the argument " + std::to_string(arg_idx));
  status = clSetKernelArg(*kernel_, arg_idx++, sizeof(int), &channel_size);
  ASSERT(status == CL_SUCCESS, "Failed to set the argument " + std::to_string(arg_idx));
  status = clSetKernelArg(*kernel_, arg_idx++, sizeof(float), &eps_);
  ASSERT(status == CL_SUCCESS, "Failed to set the argument " + std::to_string(arg_idx));
  status = clSetKernelArg(*kernel_, arg_idx++, sizeof(cl_mem), weights_buf_);
  ASSERT(status == CL_SUCCESS, "Failed to set the argument " + std::to_string(arg_idx));
  status = clSetKernelArg(*kernel_, arg_idx++, sizeof(cl_mem), biases_buf_);
  ASSERT(status == CL_SUCCESS, "Failed to set the argument " + std::to_string(arg_idx));
  status = clSetKernelArg(*kernel_, arg_idx++, sizeof(float), &relu_);
  ASSERT(status == CL_SUCCESS, "Failed to set the argument " + std::to_string(arg_idx));

  status = clEnqueueNDRangeKernel(*command_queue_, *kernel_, wg_dim, nullptr,
                                  global_size, local_size, 0, nullptr, nullptr);
  ASSERT(status == CL_SUCCESS, "Failed to launch the kernel");

  if (blocking) {
    clFinish(*command_queue_);
  }
}

void BatchNormOp::Run(const std::vector<int> &shape, bool blocking,
                      float *out_data) {
  Run(shape, blocking);
  int tensor_size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  std::size_t raw_tensor_size = sizeof(float) * tensor_size;
  clEnqueueReadBuffer(*command_queue_, *tensor_buf_, blocking, 0, raw_tensor_size,
                      out_data, 0, nullptr, nullptr);
}

