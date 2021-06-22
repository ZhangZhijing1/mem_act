#ifndef HOST_INCLUDE_DEPTHWISE_CONV_2D_OP_H_
#define HOST_INCLUDE_DEPTHWISE_CONV_2D_OP_H_

#include <CL/cl.h>

#include <vector>

class DepthwiseConv2DOp {
 public:
  DepthwiseConv2DOp(int channels, int kernel_size, int stride, int padding,
                    int channel_multiplier, bool bias, cl_kernel *kernel,
                    cl_command_queue *command_queue, cl_mem *in_buf = nullptr,
                    cl_mem *out_buf = nullptr, cl_mem *kernel_buf = nullptr);

  void SetInBuffer(cl_mem *buf);
  void SetOutBuffer(cl_mem *buf);
  void SetKernelBuffer(cl_mem *buf);

  void Run(std::vector<int> &shape, bool blocking);
  void Run(std::vector<int> &shape, bool blocking, float *out_data);

 private:
  int channels_;
  int kernel_size_;
  int stride_;
  int padding_;
  int channel_multiplier_;
  bool bias_;

  cl_kernel *kernel_;
  cl_command_queue *command_queue_;

  cl_mem *in_buf_;
  cl_mem *out_buf_;
  cl_mem *kernel_buf_;
};

#endif  // HOST_INCLUDE_DEPTHWISE_CONV_2D_OP_H_

