#include "batchnorm_test.h"

#include <numeric>

#include "memory_activation.h"

void RunBatchNormUnitTest(cl_context context,
                          cl_command_queue command_queue,
                          cl_kernel batchnorm_kernel,
                          const std::vector<int> &tensor_shape,
                          float relu,
                          bool enable_timing) {
  std::cout << "tensor_shape = [" << tensor_shape[0] << ", " << tensor_shape[1]
            << ", " << tensor_shape[2] << ", " << tensor_shape[3] << "]\n";
  const int tensor_size = std::accumulate(
      tensor_shape.begin(), tensor_shape.end(), 1, std::multiplies<int>());
  const int channel_size = tensor_shape[2] * tensor_shape[3];
  std::vector<float> tensor(tensor_size);
  std::vector<float> weights(tensor_shape[1]);
  std::vector<float> biases(tensor_shape[1]);
  std::vector<float> ref(tensor_size);

  cl_int status;

  // Generate random input and filter data.
  static unsigned int seed = time(nullptr);
  srand(seed);
  std::function<float(float, float)> random_generator = [&](float a,
                                                            float b) -> float {
    return a * static_cast<float>(rand() % 1000) + b;
  };
  std::generate(tensor.begin(), tensor.end(), RandomGenerator(1.f / 500.f, -1.0f));
  std::generate(weights.begin(), weights.end(), RandomGenerator(1.f / 5000.f, 1.0f));
  std::generate(biases.begin(), biases.end(), RandomGenerator(1.f / 10000.f, 0.f));
  float eps = 1e-5;

  // Create device buffers.
  cl_mem tensor_buf =
      clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                     tensor.size() * sizeof(float), tensor.data(), &status);
  ASSERT(status == CL_SUCCESS, "Failed to create tensor buffer");
  cl_mem weight_buf =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     weights.size() * sizeof(float), weights.data(), &status);
  ASSERT(status == CL_SUCCESS, "Failed to create weight buffer");
  cl_mem bias_buf =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     biases.size() * sizeof(float), biases.data(), &status);
  ASSERT(status == CL_SUCCESS, "Failed to create bias buffer");
  BatchNormOp op(tensor_shape[1], eps, relu, &batchnorm_kernel, &command_queue,
                 &weight_buf, &bias_buf, &tensor_buf);

  // Run on device.
  auto tic = high_resolution_clock::now();
  op.Run(tensor_shape, true, tensor.data());
  auto toc = high_resolution_clock::now();
  if (enable_timing) {
    std::cout << "Device took "
              << duration_cast<nanoseconds>(toc - tic).count() / 1000
              << " us\n";
  }

  std::copy(tensor.begin(), tensor.end(), ref.begin());
  // Run on host.
  tic = high_resolution_clock::now();
  RunBatchNormRef(ref, tensor_shape, eps, weights, biases, relu);
  toc = high_resolution_clock::now();
  if (enable_timing) {
    std::cout << "Device took "
              << duration_cast<nanoseconds>(toc - tic).count() / 1000
              << " us\n";
  }

  CheckResult(ref.data(), tensor.data(), tensor_size, false, 1e-3);

  clReleaseMemObject(tensor_buf);
  clReleaseMemObject(weight_buf);
  clReleaseMemObject(bias_buf);
}

void RunBatchNormTests(cl_context context,
                       cl_command_queue command_queue,
                       cl_kernel mean_row_kernel,
                       cl_kernel mean_col_kernel,
                       cl_kernel var_row_kernel,
                       cl_kernel var_col_kernel,
                       cl_kernel batchnorm_kernel,
                       cl_bool enable_timing) {
#if 0
  RunBatchNormUnitTest(context,
                       command_queue,
                       mean_row_kernel,
                       mean_col_kernel,
                       var_row_kernel,
                       var_col_kernel,
                       batchnorm_kernel,
                       64,      /* image_height */
                       64,      /* image_width */
                       16,      /* channels */
                       enable_timing);
  RunBatchNormUnitTest(context,
                       command_queue,
                       mean_row_kernel,
                       mean_col_kernel,
                       var_row_kernel,
                       var_col_kernel,
                       batchnorm_kernel,
                       64,      /* image_height */
                       64,      /* image_width */
                       32,      /* channels */
                       enable_timing);
  RunBatchNormUnitTest(context,
                       command_queue,
                       mean_row_kernel,
                       mean_col_kernel,
                       var_row_kernel,
                       var_col_kernel,
                       batchnorm_kernel,
                       128,     /* image_height */
                       128,     /* image_width */
                       16,      /* channels */
                       enable_timing);
  RunBatchNormUnitTest(context,
                       command_queue,
                       mean_row_kernel,
                       mean_col_kernel,
                       var_row_kernel,
                       var_col_kernel,
                       batchnorm_kernel,
                       128,     /* image_height */
                       128,     /* image_width */
                       32,      /* channels */
                       enable_timing);
  RunBatchNormUnitTest(context,
                       command_queue,
                       mean_row_kernel,
                       mean_col_kernel,
                       var_row_kernel,
                       var_col_kernel,
                       batchnorm_kernel,
                       256,     /* image_height */
                       256,     /* image_width */
                       16,      /* channels */
                       enable_timing);
  RunBatchNormUnitTest(context,
                       command_queue,
                       mean_row_kernel,
                       mean_col_kernel,
                       var_row_kernel,
                       var_col_kernel,
                       batchnorm_kernel,
                       256,     /* image_height */
                       256,     /* image_width */
                       32,      /* channels */
                       enable_timing);
  RunBatchNormUnitTest(context,
                       command_queue,
                       mean_row_kernel,
                       mean_col_kernel,
                       var_row_kernel,
                       var_col_kernel,
                       batchnorm_kernel,
                       512,     /* image_height */
                       512,     /* image_width */
                       16,      /* channels */
                       enable_timing);
  RunBatchNormUnitTest(context,
                       command_queue,
                       mean_row_kernel,
                       mean_col_kernel,
                       var_row_kernel,
                       var_col_kernel,
                       batchnorm_kernel,
                       512,     /* image_height */
                       512,     /* image_width */
                       32,      /* channels */
                       enable_timing);
  RunBatchNormUnitTest(context,
                       command_queue,
                       mean_row_kernel,
                       mean_col_kernel,
                       var_row_kernel,
                       var_col_kernel,
                       batchnorm_kernel,
                       717,     /* image_height */
                       619,     /* image_width */
                       17,      /* channels */
                       enable_timing);
  RunBatchNormUnitTest(context,
                       command_queue,
                       mean_row_kernel,
                       mean_col_kernel,
                       var_row_kernel,
                       var_col_kernel,
                       batchnorm_kernel,
                       316,     /* image_height */
                       428,     /* image_width */
                       97,      /* channels */
                       enable_timing);
#endif
}
