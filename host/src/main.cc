#include <CL/cl.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <functional>
#include <ios>
#include <iostream>
#include <memory>
#include <numeric>
#include <ratio>
#include <string>
#include <vector>

#include "batchnorm.h"
#include "batchnorm_op.h"
#include "batchnorm_test.h"
#include "conv2d.h"
#include "conv2d_op.h"
#include "conv2d_test.h"
#include "depthwise_conv2d.h"
#include "depthwise_conv2d_op.h"
#include "depthwise_conv2d_test.h"
#include "kernel.h"
#include "memory_activation.h"
#include "mobilenetv2.h"
#include "model.h"
#include "tensor.h"
#include "workspace.h"

int main(int argc, char **argv) {
#if 0
  std::ifstream is("../param.dat");
  is.seekg(0, std::ios_base::end);
  std::size_t raw_size = is.tellg();
  is.seekg(0, std::ios_base::beg);
  std::cout << "Number of parameters = " << raw_size / sizeof(float) << '\n';
#endif
#if 0
  std::vector<int> tensor_shape{1, 3, 3, 3};
  // Create the workspace.
  Workspace ws("Intel(R) OpenCL HD Graphics");
  Tensor a(tensor_shape);
  a.ReadFile(is, 10, true, &ws, CL_TRUE, 0, nullptr, nullptr);
  //a.PushToDevice(ws);
  //ws.FinishCommandQueue();
  a.PopToHost(ws);
  ws.FinishCommandQueue();
  for (int i = 0; i < 10; i++) {
    std::cout << a[i] << ' ';
  }
  std::cout << '\n';
#endif

#if 0
  const int in_channels = 3;
  const int out_channels = 32;
  const int in_height = 64;
  const int in_width = 64;
  std::vector<float> in_data(in_channels * in_height * in_width);
  std::vector<float> out_data(out_channels * in_height * in_width);
  std::vector<float> weight_data(3 * 32 * 3 * 3);
  std::vector<float> bias_data(32);
  std::vector<int> tensor_shape{1, in_channels, in_height, in_width};

  std::fill(in_data.begin(), in_data.end(), 1.f);
  std::ifstream is("../param.dat");
  is.read(reinterpret_cast<char *>(&weight_data[0]),
          weight_data.size() * sizeof(float));

  RunConv2DRef(in_data, out_data, weight_data, tensor_shape, out_channels, 3, 1, 1);
  for (int i = 0; i < 20; i++) {
    std::cout << out_data[i] << ' ';
  }
  std::cout << '\n';
#endif

#if 1
  // Create the workspace.
  Workspace ws("Intel(R) OpenCL HD Graphics");
  // Create OpenCL kernels.
  Kernel conv_kernel =
      ws.CreateKernel("/../device/conv2d.cl", "Convolute", false);
  Kernel batchnorm_kernel =
      ws.CreateKernel("/../device/batchnorm2d.cl", "BatchNorm", false);

  // Prepare runtime memory.
  const int max_channels = 64;
  const int kernel_size = 3;
  const int model_kernel_size =
      max_channels * max_channels * kernel_size * kernel_size;
  std::vector<float> kernel_data(model_kernel_size);
  const int in_channels = 3;
  const int in_height = 64;
  const int in_width = 64;
  const int tensor_size = max_channels * in_height * in_width;
  std::vector<float> in_data(tensor_size);
  std::vector<float> out_data(tensor_size);
  std::vector<float> ref_data(tensor_size);
  std::vector<float> weight_data(max_channels);
  std::vector<float> bias_data(max_channels);
  std::vector<int> tensor_shape{1, in_channels, in_height, in_width};

  // Random data for the moment.
  std::generate(in_data.begin(), in_data.end(),
                RandomGenerator(1.f / 500.f, -1.f));
  std::generate(kernel_data.begin(), kernel_data.end(),
                RandomGenerator(1.f / 500.f, -1.f));
  std::generate(weight_data.begin(), weight_data.end(),
                RandomGenerator(1.f / 10000.f, 1.f));
  std::generate(bias_data.begin(), bias_data.end(),
                RandomGenerator(1.f / 10000.f, 0.f));

  // Run the model on device.
  auto start = std::chrono::high_resolution_clock::now();
  RunModel(ws.GetContext(), ws.GetCommandQueue(), conv_kernel.Get(),
           batchnorm_kernel.Get(), tensor_shape, in_data, out_data, kernel_data,
           weight_data, bias_data);
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Device took " << elapsed.count() << " us\n";

  // Run the model on host.
  tensor_shape = {1, in_channels, in_height, in_width};
  start = std::chrono::high_resolution_clock::now();
  RunModelRef(tensor_shape, in_data, ref_data, kernel_data, weight_data,
              bias_data);
  end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Host took " << elapsed.count() << " us\n";

  // Check output.
  int out_size = std::accumulate(tensor_shape.begin(), tensor_shape.end(), 1,
                                 std::multiplies<int>());
  CheckResult(ref_data.data(), out_data.data(), out_size, false, 1e-3);
#endif
  return EXIT_SUCCESS;
}

