#include "conv2d_test.h"

void RunConv2DUnitTest(cl_context context,
                       cl_command_queue command_queue,
                       cl_kernel kernel,
                       const int in_height,
                       const int in_width,
                       const int in_channels,
                       const int out_channels,
                       const int kernel_size,
                       const int stride,
                       const int padding,
                       bool enable_timing) {
  const int in_size = in_height * in_width;
  const int batch_in_size = in_channels * in_size;
  const int out_height = ((in_height + 2 * padding - kernel_size) / stride) + 1;
  const int out_width = ((in_width + 2 * padding - kernel_size) / stride) + 1;
  const int out_size = out_height * out_width;
  const int batch_out_size = out_channels * out_size;
  const int batch_kernel_size = in_channels * out_channels * kernel_size * kernel_size;
  std::vector<float> in_data(batch_in_size);
  std::vector<float> kernel_data(batch_kernel_size);
  std::vector<float> out_data(batch_out_size);

  cl_int status;

  // generate random input and filter data
  std::function<float(void)> random_generator = [&](void) -> float {
    return static_cast<float>(rand() % 1000) / 500.0f - 1.0f;
  };
  std::generate(in_data.begin(), in_data.end(), random_generator);
  std::generate(kernel_data.begin(), kernel_data.end(), random_generator);
  
  /*std::vector<float> in_data{
      -1.5266, -2.2902, 0.9136,  -0.2422, -0.2680, 1.5673,  -0.2637, 0.6794,
      0.4382,  -0.4530, 0.4100,  -0.0026, -1.5038, 0.1969,  0.1616,  -0.1957,
      -0.9937, 1.7243,  -1.3206, -0.0242, 0.2609,  -1.1991, 1.8211,  -0.9031,
      -0.5881, 0.3906,  -1.5095, -1.0373, 1.5800,  -1.8982, -0.0669, 0.1666,
      0.8502,  -0.1293, 0.2997,  0.4349,

      -0.8013, 1.1970,  -0.4157, 0.6482,  -2.1812, -0.8270, 1.0739,  1.1664,
      0.5779,  0.5153,  -0.1872, 0.1608,  -2.2777, 0.8158,  -0.1283, -0.9327,
      0.3702,  0.6493,  -1.1984, -0.3307, 0.5456,  -0.2534, 0.2064,  -0.6567,
      0.0425,  0.4562,  1.2363,  0.6793,  0.8169,  0.8313,  0.0519,  1.4201,
      -1.1069, -0.5797, -1.5816, -1.0307};
  std::vector<float> kernel_data{0.0384,  -0.4894, 0.3383,  1.1298,  -0.5560,
                                 -0.4984, -1.0636, -0.1496, 0.4134,

                                 -0.7759, -0.8000, 0.5581,  -1.2810, -0.3842,
                                 -0.6451, -1.5266, 0.6783,  -1.3185,

                                 -0.5296, -0.0437, -0.9418, 0.4215,  1.4548,
                                 1.3904,  0.9339,  -1.7418, -1.2548,

                                 -0.3491, 0.7922,  -0.5322, 1.3435,  0.3320,
                                 -1.3207, -1.9783, -0.4233, -0.4674};*/

  // Prepare device buffers.
  cl_mem in_buf =
      clCreateBuffer(context,
                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * in_data.size(),
                     in_data.data(),
                     &status);
  if (status != CL_SUCCESS) {
    std::cout << "Couldn't create the input buffer\n";
    return;
  }
  cl_mem kernel_buf =
      clCreateBuffer(context,
                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * kernel_data.size(),
                     kernel_data.data(),
                     &status);
  if (status != CL_SUCCESS) {
    std::cout << "Couldn't create the filter buffer\n";
    return;
  }
  cl_mem out_buf =
      clCreateBuffer(context,
                     CL_MEM_WRITE_ONLY,
                     sizeof(float) * out_data.size(),
                     nullptr,
                     &status);
  if (status != CL_SUCCESS) {
    std::cout << "Couldn't create the output buffer\n";
    return;
  }

  // Run on device.
  Conv2DOp op(in_channels, out_channels, kernel_size, stride, padding, false,
              &kernel, &command_queue, &in_buf, &out_buf,
              &kernel_buf);
  std::vector<int> shape{1, in_channels, in_height, in_width};
  auto tic = high_resolution_clock::now();
  op.Run(shape, true, out_data.data());
  auto toc = high_resolution_clock::now();
  if (enable_timing) {
    std::cout << "Device took "
              << duration_cast<microseconds>(toc - tic).count()
              << " us\n";
  }
  // Run on host.
  std::vector<float> ref_data(batch_out_size);
  tic = high_resolution_clock::now();
  RunConv2DRef(in_data, ref_data, kernel_data, in_height, in_width,
               in_channels, out_channels, kernel_size, stride, padding);
  toc = high_resolution_clock::now();
  if (enable_timing) {
    std::cout << "Host took "
              << duration_cast<microseconds>(toc - tic).count()
              << " us\n";
  }
  // compare output
  CheckResult(ref_data.data(), out_data.data(), batch_out_size, false, 1e-3f);
  if (shape.size() != 4) {
    std::cout << "Error: output shape changed size\n";
  } else {
    std::cout << "output shape = [" << shape[0] << ", " << shape[1] << ", "
              << shape[2] << ", " << shape[3] << "]\n";
  }
}

void RunConv2DTests(cl_context context,
                    cl_command_queue command_queue,
                    cl_kernel kernel,
                    bool enable_timing) {
#if 0
  std::cout << "Test 1: input shape = [128, 128], "
            << "input channels = 64, "
            << "output channels = 64, "
            << "filter shape = [3, 3]\n";
  RunConv2DUnitTest(context,        /* context */
                    command_queue,  /* command_queue */
                    kernel,         /* kernel */
                    128,            /* image_height */
                    128,            /* image_width */
                    64,             /* in_channels */
                    64,             /* out_channels */
                    3,              /* filter_height */
                    3,              /* filter_width */
                    1,              /* stride */
                    enable_timing   /* enable_timing */);
  std::cout << "Test 2: input shape = [128, 128], "
            << "input channels = 32, "
            << "output channels = 32, "
            << "filter shape = [3, 3]\n";
  RunConv2DUnitTest(context,        /* context */
                    command_queue,  /* command_queue */
                    kernel,         /* kernel */
                    128,            /* image_height */
                    128,            /* image_width */
                    32,             /* in_channels */
                    32,             /* out_channels */
                    3,              /* filter_height */
                    3,              /* filter_width */
                    1,              /* stride */
                    enable_timing   /* enable_timing */);
  std::cout << "Test 3: input shape = [65, 69], "
            << "input channels = 3, "
            << "output channels = 16, "
            << "filter shape = [3, 3]\n";
  RunConv2DUnitTest(context,        /* context */
                    command_queue,  /* command_queue */
                    kernel,         /* kernel */
                    65,             /* image_height */
                    69,             /* image_width */
                    3,              /* in_channels */
                    16,             /* out_channels */
                    3,              /* filter_height */
                    3,              /* filter_width */
                    1,              /* stride */
                    enable_timing   /* enable_timing */);
  std::cout << "Test 4: input shape = [71, 92], "
            << "input channels = 15, "
            << "output channels = 18, "
            << "filter shape = [3, 3]\n";
  RunConv2DUnitTest(context,        /* context */
                    command_queue,  /* command_queue */
                    kernel,         /* kernel */
                    71,             /* image_height */
                    92,             /* image_width */
                    15,             /* in_channels */
                    18,             /* out_channels */
                    3,              /* filter_height */
                    3,              /* filter_width */
                    1,              /* stride */
                    enable_timing   /* enable_timing */);

  std::cout << "Test 5: input shape = [128, 128], "
            << "input channels = 64, "
            << "output channels = 64, "
            << "filter shape = [5, 5]\n";
  RunConv2DUnitTest(context,        /* context */
                    command_queue,  /* command_queue */
                    kernel,         /* kernel */
                    128,            /* image_height */
                    128,            /* image_width */
                    64,             /* in_channels */
                    64,             /* out_channels */
                    5,              /* filter_height */
                    5,              /* filter_width */
                    1,              /* stride */
                    enable_timing   /* enable_timing */);
  std::cout << "Test 6: input shape = [128, 128], "
            << "input channels = 32, "
            << "output channels = 32, "
            << "filter shape = [5, 5]\n";
  RunConv2DUnitTest(context,        /* context */
                    command_queue,  /* command_queue */
                    kernel,         /* kernel */
                    128,            /* image_height */
                    128,            /* image_width */
                    32,             /* in_channels */
                    32,             /* out_channels */
                    5,              /* filter_height */
                    5,              /* filter_width */
                    1,              /* stride */
                    enable_timing   /* enable_timing */);
  std::cout << "Test 7: input shape = [65, 69], "
            << "input channels = 3, "
            << "output channels = 16, "
            << "filter shape = [5, 5]\n";
  RunConv2DUnitTest(context,        /* context */
                    command_queue,  /* command_queue */
                    kernel,         /* kernel */
                    65,             /* image_height */
                    69,             /* image_width */
                    3,              /* in_channels */
                    16,             /* out_channels */
                    5,              /* filter_height */
                    5,              /* filter_width */
                    1,              /* stride */
                    enable_timing   /* enable_timing */);
  std::cout << "Test 8: input shape = [71, 92], "
            << "input channels = 15, "
            << "output channels = 18, "
            << "filter shape = [5, 5]\n";
  RunConv2DUnitTest(context,        /* context */
                    command_queue,  /* command_queue */
                    kernel,         /* kernel */
                    71,             /* image_height */
                    92,             /* image_width */
                    15,             /* in_channels */
                    18,             /* out_channels */
                    5,              /* filter_height */
                    5,              /* filter_width */
                    1,              /* stride */
                    enable_timing   /* enable_timing */);
#endif
}

