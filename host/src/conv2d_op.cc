#include "conv2d_op.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <numeric>

#include "memory_activation.h"

Conv2DOp::Conv2DOp(int in_channels, int out_channels, int kernel_size,
                   int stride, int padding, bool bias, cl_kernel *kernel,
                   cl_command_queue *command_queue, cl_mem *in_buf,
                   cl_mem *out_buf, cl_mem *kernel_buf)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      stride_(stride),
      padding_(padding),
      bias_(bias),
      kernel_size_(kernel_size),
      kernel_(kernel),
      batch_kernel_size_(in_channels * kernel_size * kernel_size),
      command_queue_(command_queue),
      in_buf_(in_buf),
      out_buf_(out_buf),
      kernel_buf_(kernel_buf) {}

void Conv2DOp::SetInBuffer(cl_mem *buf) {
  in_buf_ = buf;
}

void Conv2DOp::SetOutBuffer(cl_mem *buf) {
  out_buf_ = buf;
}

void Conv2DOp::SetKernelBuffer(cl_mem *buf) {
  kernel_buf_ = buf;
}

void Conv2DOp::Run(std::vector<int> &shape, bool blocking) {
  const static cl_uint wg_dim = 3;
  const static cl_uint wg_width = 8;
  const static cl_uint wg_height = 8;
  const static cl_uint wg_depth = 4;

  ASSERT(shape.size() == 4, "Only accepts 4D input");
  ASSERT(shape[1] == in_channels_, "Number of input channels");
  ASSERT(in_buf_ != nullptr, "input buffer is null");
  ASSERT(out_buf_ != nullptr, "output buffer is null");
  ASSERT(kernel_buf_ != nullptr, "kernel buffer is null");

  const int batch = shape[0];
  const int in_height = shape[2];
  const int in_width = shape[3];
  const int in_size = in_height * in_width;
  cl_int status;
  const cl_uint total_work_items_x = RoundUp(in_width, wg_width);
  const cl_uint total_work_items_y = RoundUp(in_height, wg_height);
  const cl_uint total_work_items_z = RoundUp(out_channels_, wg_depth);

  std::size_t global_size[wg_dim] = {
    static_cast<std::size_t>(total_work_items_x),
    static_cast<std::size_t>(total_work_items_y),
    static_cast<std::size_t>(total_work_items_z)
  };
  std::size_t local_size[wg_dim] = {
    static_cast<std::size_t>(wg_width),
    static_cast<std::size_t>(wg_height),
    static_cast<std::size_t>(wg_depth)
  };

  const int out_height = ((in_height + 2 * padding_ - kernel_size_) / stride_) + 1;
  const int out_width = ((in_width + 2 * padding_ - kernel_size_) / stride_) + 1;
  const int out_size = out_height * out_width;
  cl_uint arg_idx = 0;
  status = clSetKernelArg(*kernel_, arg_idx++, sizeof(cl_mem), in_buf_);
  ASSERT(status == CL_SUCCESS, "Failed to set the argument " + std::to_string(arg_idx));
  status = clSetKernelArg(*kernel_, arg_idx++, sizeof(cl_mem), out_buf_);
  ASSERT(status == CL_SUCCESS, "Failed to set the argument " + std::to_string(arg_idx));
  status = clSetKernelArg(*kernel_, arg_idx++, sizeof(cl_mem), kernel_buf_);
  ASSERT(status == CL_SUCCESS, "Failed to set the argument " + std::to_string(arg_idx));
  status = clSetKernelArg(*kernel_, arg_idx++, sizeof(int), &in_height);
  ASSERT(status == CL_SUCCESS, "Failed to set the argument " + std::to_string(arg_idx));
  status = clSetKernelArg(*kernel_, arg_idx++, sizeof(int), &in_width);
  ASSERT(status == CL_SUCCESS, "Failed to set the argument " + std::to_string(arg_idx));
  status = clSetKernelArg(*kernel_, arg_idx++, sizeof(int), &in_size);
  ASSERT(status == CL_SUCCESS, "Failed to set the argument " + std::to_string(arg_idx));
  status = clSetKernelArg(*kernel_, arg_idx++, sizeof(int), &out_height);
  ASSERT(status == CL_SUCCESS, "Failed to set the argument " + std::to_string(arg_idx));
  status = clSetKernelArg(*kernel_, arg_idx++, sizeof(int), &out_width);
  ASSERT(status == CL_SUCCESS, "Failed to set the argument " + std::to_string(arg_idx));
  status = clSetKernelArg(*kernel_, arg_idx++, sizeof(int), &out_size);
  ASSERT(status == CL_SUCCESS, "Failed to set the argument " + std::to_string(arg_idx));
  status = clSetKernelArg(*kernel_, arg_idx++, sizeof(int), &in_channels_);
  ASSERT(status == CL_SUCCESS, "Failed to set the argument " + std::to_string(arg_idx));
  status = clSetKernelArg(*kernel_, arg_idx++, sizeof(int), &out_channels_);
  ASSERT(status == CL_SUCCESS, "Failed to set the argument " + std::to_string(arg_idx));
  status = clSetKernelArg(*kernel_, arg_idx++, sizeof(int), &kernel_size_);
  ASSERT(status == CL_SUCCESS, "Failed to set the argument " + std::to_string(arg_idx));
  status = clSetKernelArg(*kernel_, arg_idx++, sizeof(int), &batch_kernel_size_);
  ASSERT(status == CL_SUCCESS, "Failed to set the argument " + std::to_string(arg_idx));
  status = clSetKernelArg(*kernel_, arg_idx++, sizeof(int), &stride_);
  ASSERT(status == CL_SUCCESS, "Failed to set the argument " + std::to_string(arg_idx));
  status = clSetKernelArg(*kernel_, arg_idx++, sizeof(int), &padding_);
  ASSERT(status == CL_SUCCESS, "Failed to set the argument " + std::to_string(arg_idx));

  status = clEnqueueNDRangeKernel(*command_queue_, *kernel_, wg_dim, nullptr,
                                  global_size, local_size, 0, nullptr, nullptr);
  ASSERT(status == CL_SUCCESS, "Failed to launch the kernel");

  if (blocking) {
    clFinish(*command_queue_);
  }

  shape[1] = out_channels_;
  shape[2] = out_height;
  shape[3] = out_width;
}

void Conv2DOp::Run(std::vector<int> &shape, bool blocking, float *out_data) {
  Run(shape, blocking);
  int tensor_size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  std::size_t raw_tensor_size = sizeof(float) * tensor_size;
  clEnqueueReadBuffer(*command_queue_, *out_buf_, blocking, 0, raw_tensor_size,
                      out_data, 0, nullptr, nullptr);
}

