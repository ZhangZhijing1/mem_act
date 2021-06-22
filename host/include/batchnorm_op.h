#ifndef HOST_INCLUDE_BATCHNORM_OP_H_
#define HOST_INCLUDE_BATCHNORM_OP_H_

#include <CL/cl.h>

#include <memory>
#include <vector>

class BatchNormOp {
 public:
  BatchNormOp(int num_features, float eps, float relu,
              cl_kernel *kernel,
              cl_command_queue *command_queue,
              cl_mem *weights_buf = nullptr,
              cl_mem *biases_buf = nullptr,
              cl_mem *tensor_buf = nullptr);

  void SetTensorBuffer(cl_mem *buf);
  void SetWeightBuffer(cl_mem *buf);
  void SetBiasBuffer(cl_mem *buf);

  void Run(const std::vector<int> &shape, bool blocking);
  void Run(const std::vector<int> &shape, bool blocking, float *out_data);

 private:
  int num_features_;
  float eps_;
  float relu_;

  cl_kernel *kernel_;
  cl_command_queue *command_queue_;

  cl_mem *tensor_buf_;
  cl_mem *weights_buf_;
  cl_mem *biases_buf_;
};

#endif  // HOST_INCLUDE_BATCHNORM_OP_H_

