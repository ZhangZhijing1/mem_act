#include "model.h"

inline void Swap(cl_mem **buf1, cl_mem **buf2) {
  cl_mem *temp = *buf1;
  *buf1 = *buf2;
  *buf2 = temp;
}

void RunModel(cl_context context,
              cl_command_queue command_queue,
              cl_kernel conv_kernel,
              cl_kernel batchnorm_kernel,
              std::vector<int> &tensor_shape,
              const std::vector<float> &in_data,
              std::vector<float> &out_data,
              const std::vector<float> &kernel_data,
              const std::vector<float> &weight_data,
              const std::vector<float> &bias_data) {
  const int batch = tensor_shape[0];
  const int in_channels = tensor_shape[1];
  const int in_height = tensor_shape[2];
  const int in_width = tensor_shape[3];
  const int stride = 1;
  const int padding = 1;
  const float eps = 1e-5;
  const float relu = 1.f;
  const std::vector<int> channels{3, 32, 32, 64, 64, 64, 64};

  ASSERT(in_data.size() >= in_channels * in_height * in_width,
         "Input buffer doesn't have enough data");
  const int max_channels = 64;
  const int tensor_size = max_channels * in_height * in_width;
  const int kernel_size = 3;
  const int model_kernel_size =
      max_channels * max_channels * kernel_size * kernel_size;

  int status;

  // Create device buffers.
  cl_mem tensor_buf_a =
      clCreateBuffer(context, CL_MEM_READ_WRITE, tensor_size * sizeof(float),
                     nullptr, &status);
  ASSERT(status == CL_SUCCESS, "Failed to create tensor buffer A");
  cl_mem tensor_buf_b =
      clCreateBuffer(context, CL_MEM_READ_WRITE, tensor_size * sizeof(float),
                     nullptr, &status);
  ASSERT(status == CL_SUCCESS, "Failed to create tensor buffer B");
  cl_mem kernel_buf =
      clCreateBuffer(context, CL_MEM_READ_ONLY,
                     model_kernel_size * sizeof(float), nullptr, &status);
  ASSERT(status == CL_SUCCESS, "Failed to create kernel buffer");
  cl_mem weight_buf =
      clCreateBuffer(context, CL_MEM_READ_ONLY, max_channels * sizeof(float),
                     nullptr, &status);
  ASSERT(status == CL_SUCCESS, "Failed to create weight buffer");
  cl_mem bias_buf =
      clCreateBuffer(context, CL_MEM_READ_ONLY, max_channels * sizeof(float),
                     nullptr, &status);
  ASSERT(status == CL_SUCCESS, "Failed to create bias buffer");

  cl_mem *in_buf = &tensor_buf_a;
  cl_mem *out_buf = &tensor_buf_b;

  // Push data to device buffers.
  clEnqueueWriteBuffer(command_queue, *in_buf, CL_TRUE, 0,
                       in_data.size() * sizeof(float), in_data.data(), 0,
                       nullptr, nullptr);
  clEnqueueWriteBuffer(command_queue, kernel_buf, CL_TRUE, 0,
                       model_kernel_size * sizeof(float), kernel_data.data(), 0,
                       nullptr, nullptr);
  clEnqueueWriteBuffer(command_queue, weight_buf, CL_TRUE, 0,
                       max_channels * sizeof(float), weight_data.data(), 0,
                       nullptr, nullptr);
  clEnqueueWriteBuffer(command_queue, bias_buf, CL_TRUE, 0,
                       max_channels * sizeof(float), bias_data.data(), 0,
                       nullptr, nullptr);

  // Create device operators.
  Conv2DOp conv1(channels[0], channels[1], kernel_size, stride, padding, false,
                 &conv_kernel, &command_queue);
  BatchNormOp bn1(channels[1], eps, relu, &batchnorm_kernel, &command_queue);
  Conv2DOp conv2(channels[1], channels[2], kernel_size, stride, padding, false,
                 &conv_kernel, &command_queue);
  BatchNormOp bn2(channels[2], eps, relu, &batchnorm_kernel, &command_queue);
  Conv2DOp conv3(channels[2], channels[3], kernel_size, stride, padding, false,
                 &conv_kernel, &command_queue);
  BatchNormOp bn3(channels[3], eps, relu, &batchnorm_kernel, &command_queue);
  Conv2DOp conv4(channels[3], channels[4], kernel_size, stride, padding, false,
                 &conv_kernel, &command_queue);
  BatchNormOp bn4(channels[4], eps, relu, &batchnorm_kernel, &command_queue);
  Conv2DOp conv5(channels[3], channels[4], kernel_size, stride, padding, false,
                 &conv_kernel, &command_queue);
  BatchNormOp bn5(channels[4], eps, relu, &batchnorm_kernel, &command_queue);
  Conv2DOp conv6(channels[3], channels[4], kernel_size, stride, padding, false,
                 &conv_kernel, &command_queue);
  BatchNormOp bn6(channels[4], eps, relu, &batchnorm_kernel, &command_queue);

  // Launch the kernels.
  conv1.SetInBuffer(in_buf);
  conv1.SetKernelBuffer(&kernel_buf);
  conv1.SetOutBuffer(out_buf);
  conv1.Run(tensor_shape, true);
  bn1.SetWeightBuffer(&weight_buf);
  bn1.SetBiasBuffer(&bias_buf);
  bn1.SetTensorBuffer(out_buf);
  bn1.Run(tensor_shape, true);

  Swap(&in_buf, &out_buf);
  conv2.SetInBuffer(in_buf);
  conv2.SetKernelBuffer(&kernel_buf);
  conv2.SetOutBuffer(out_buf);
  conv2.Run(tensor_shape, true);
  bn2.SetWeightBuffer(&weight_buf);
  bn2.SetBiasBuffer(&bias_buf);
  bn2.SetTensorBuffer(out_buf);
  bn2.Run(tensor_shape, true);

  Swap(&in_buf, &out_buf);
  conv3.SetInBuffer(in_buf);
  conv3.SetKernelBuffer(&kernel_buf);
  conv3.SetOutBuffer(out_buf);
  conv3.Run(tensor_shape, true);
  bn3.SetWeightBuffer(&weight_buf);
  bn3.SetBiasBuffer(&bias_buf);
  bn3.SetTensorBuffer(out_buf);
  bn3.Run(tensor_shape, true);

  Swap(&in_buf, &out_buf);
  conv4.SetInBuffer(in_buf);
  conv4.SetKernelBuffer(&kernel_buf);
  conv4.SetOutBuffer(out_buf);
  conv4.Run(tensor_shape, true);
  bn4.SetWeightBuffer(&weight_buf);
  bn4.SetBiasBuffer(&bias_buf);
  bn4.SetTensorBuffer(out_buf);
  bn4.Run(tensor_shape, true, out_data.data());

  Swap(&in_buf, &out_buf);
  conv5.SetInBuffer(in_buf);
  conv5.SetKernelBuffer(&kernel_buf);
  conv5.SetOutBuffer(out_buf);
  conv5.Run(tensor_shape, true);
  bn5.SetWeightBuffer(&weight_buf);
  bn5.SetBiasBuffer(&bias_buf);
  bn5.SetTensorBuffer(out_buf);
  bn5.Run(tensor_shape, true, out_data.data());

  Swap(&in_buf, &out_buf);
  conv6.SetInBuffer(in_buf);
  conv6.SetKernelBuffer(&kernel_buf);
  conv6.SetOutBuffer(out_buf);
  conv6.Run(tensor_shape, true);
  bn6.SetWeightBuffer(&weight_buf);
  bn6.SetBiasBuffer(&bias_buf);
  bn6.SetTensorBuffer(out_buf);
  bn6.Run(tensor_shape, true, out_data.data());

  clReleaseMemObject(tensor_buf_a);
  clReleaseMemObject(tensor_buf_b);
  clReleaseMemObject(kernel_buf);
  clReleaseMemObject(weight_buf);
  clReleaseMemObject(bias_buf);
}

void RunModelRef(std::vector<int> &tensor_shape,
                 std::vector<float> &in_data,
                 std::vector<float> &out_data,
                 const std::vector<float> &kernel_data,
                 const std::vector<float> &weight_data,
                 const std::vector<float> &bias_data) {
  const int kernel_size = 3;
  const int stride = 1;
  const int padding = 1;
  const std::vector<int> channels{3, 32, 32, 64, 64, 64, 64};

  ASSERT(tensor_shape[1] == channels[0], "Input channels should be 3");

  RunConv2DRef(in_data, out_data, kernel_data, tensor_shape, channels[1],
               kernel_size, stride, padding);
  RunBatchNormRef(out_data, 1, channels[1], tensor_shape[2] * tensor_shape[3],
                  1e-5, weight_data, bias_data, 1.f);

  in_data.swap(out_data);
  RunConv2DRef(in_data, out_data, kernel_data, tensor_shape, channels[2],
               kernel_size, stride, padding);
  RunBatchNormRef(out_data, 1, channels[2], tensor_shape[2] * tensor_shape[3],
                  1e-5, weight_data, bias_data, 1.f);

  in_data.swap(out_data);
  RunConv2DRef(in_data, out_data, kernel_data, tensor_shape, channels[3],
               kernel_size, stride, padding);
  RunBatchNormRef(out_data, 1, channels[3], tensor_shape[2] * tensor_shape[3],
                  1e-5, weight_data, bias_data, 1.f);

  in_data.swap(out_data);
  RunConv2DRef(in_data, out_data, kernel_data, tensor_shape, channels[4],
               kernel_size, stride, padding);
  RunBatchNormRef(out_data, 1, channels[4], tensor_shape[2] * tensor_shape[3],
                  1e-5, weight_data, bias_data, 1.f);

  in_data.swap(out_data);
  RunConv2DRef(in_data, out_data, kernel_data, tensor_shape, channels[5],
               kernel_size, stride, padding);
  RunBatchNormRef(out_data, 1, channels[5], tensor_shape[2] * tensor_shape[3],
                  1e-5, weight_data, bias_data, 1.f);

  in_data.swap(out_data);
  RunConv2DRef(in_data, out_data, kernel_data, tensor_shape, channels[6],
               kernel_size, stride, padding);
  RunBatchNormRef(out_data, 1, channels[6], tensor_shape[2] * tensor_shape[3],
                  1e-5, weight_data, bias_data, 1.f);
}

