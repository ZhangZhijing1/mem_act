#include "depthwise_conv2d_test.h"

void RunDepthwiseConv2DUnitTest(cl_context context,
                                cl_command_queue command_queue,
                                cl_kernel kernel,
                                int in_height,
                                int in_width,
                                int in_channels,
                                int channel_multiplier,
                                int kernel_size,
                                int stride,
                                int padding,
                                bool enable_timing) {
  const int in_size = in_height * in_width;
  const int batch_in_size = in_channels * in_size;
  const int out_height = ((in_height + 2 * padding - kernel_size) / stride) + 1;
  const int out_width = ((in_width + 2 * padding - kernel_size) / stride) + 1;
  const int out_size = out_height * out_width;
  const int out_channels = in_channels * channel_multiplier;
  const int batch_out_size = out_channels * out_size;
  const int batch_kernel_size = in_channels * out_channels * kernel_size * kernel_size;
  std::vector<float> in_data(batch_in_size);
  std::vector<float> kernel_data(batch_kernel_size);
  std::vector<float> out_data(batch_out_size);

  cl_int status;

  // Generate random input and kernel data.
  unsigned int seed = time(NULL);
  srand(seed);
  std::function<float(void)> random_generator = [&](void) -> float {
    return static_cast<float>(rand() % 1000) / 500.0f;
  };
  std::generate(in_data.begin(), in_data.end(), random_generator);
  std::generate(kernel_data.begin(), kernel_data.end(), random_generator);

  /*std::vector<float> in_data{
      -0.0050, -0.5480, 0.4862,  -1.3104, 0.7157,  -0.3664, -0.0154, -0.3711,
      -0.9841, -0.5468, -1.3137, -0.1265, 1.3002,  0.2857,  -0.8587, -0.2488,
      -0.3826, 1.0879,  0.4409,  0.7503,  -0.8366, -0.0686, -2.3727, -0.0298,
      0.3174,  -1.5617, -0.5081, 1.1148,  0.7733,  1.0872,  1.0423,  -1.1458,
      0.3749,  -0.7861, -0.1301, 0.0030,

      -0.1087, 1.8242,  -1.5924, -1.4410, -2.2299, 0.0893,  0.8182,  -0.5112,
      -0.9118, 0.8252,  -1.3567, 0.4459,  -0.2666, -0.4666, 1.5652,  -0.9975,
      -1.1642, -2.9685, 0.5553,  2.3048,  -0.2379, 0.4890,  -1.2577, -0.6463,
      -1.1306, 1.1891,  0.6726,  -0.0068, -0.9092, -0.2958, -0.5933, 0.3563,
      0.4563,  -0.5467, -0.2557, 0.4084};
  std::vector<float> kernel_data{-0.1207, -1.0620, -1.2497, -0.8533, 0.9252,
                              0.1580,  0.9397,  -0.0320, -1.3117,

                              -0.0843, 0.2205,  0.2483,  -0.1726, -0.1987,
                              0.8957,  -0.1877, -0.6403, -0.5226};*/

  // Create device buffers.
  cl_mem in_buf =
      clCreateBuffer(context,                                 /* context */
                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, /* flags */
                     sizeof(float) * in_data.size(),          /* size */
                     in_data.data(),                          /* host_ptr */
                     &status /* errcode_ret */);
  cl_mem out_buf = clCreateBuffer(context,                         /* context */
                                  CL_MEM_WRITE_ONLY,               /* flags */
                                  sizeof(float) * out_data.size(), /* size */
                                  nullptr, /* host_ptr */
                                  &status /* errcode_ret */);
  cl_mem kernel_buf =
      clCreateBuffer(context,                                 /* context */
                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, /* flags */
                     sizeof(float) * kernel_data.size(),      /* size */
                     kernel_data.data(),                      /* host_ptr */
                     &status /* errcode_ret */);

  DepthwiseConv2DOp op(in_channels, kernel_size, stride, padding, channel_multiplier,
                       false, &kernel, &command_queue, &in_buf, &out_buf,
                       &kernel_buf);
  std::vector<int> shape{1, in_channels, in_height, in_width};

  // Run on device.
  auto tic = high_resolution_clock::now();
  op.Run(shape, true, out_data.data());
  auto toc = high_resolution_clock::now();
  if (enable_timing) {
    std::cout << "Device took "
              << duration_cast<microseconds>(toc - tic).count()
              << " us\n";
  }

  std::vector<float> ref_data(batch_out_size);
  // Run on host.
  tic = high_resolution_clock::now();
  RunDepthwiseConv2DRef(in_data, ref_data, kernel_data, in_height,
                        in_width, in_channels, channel_multiplier,
                        kernel_size, stride, padding);
  toc = high_resolution_clock::now();
  if (enable_timing) {
    std::cout << "Host took "
              << duration_cast<microseconds>(toc - tic).count()
              << " us\n";
  }
  CheckResult(ref_data.data(), out_data.data(), batch_out_size, false, 1e-3f);
}

void RunDepthwiseConv2DTests(cl_context context,
                             cl_command_queue command_queue,
                             cl_kernel kernel,
                             bool enable_timing) {
  RunDepthwiseConv2DUnitTest(context,         /* context */
                             command_queue,   /* command_queue */
                             kernel,          /* kernel */
                             128,             /* image_height */
                             128,             /* image_width */
                             16,              /* in_channels */
                             2,               /* channel_multiplier */
                             3,               /* filter_height */
                             3,               /* filter_width */
                             1,               /* stride */
                             CL_TRUE          /* enable_timing */);
  RunDepthwiseConv2DUnitTest(context,         /* context */
                             command_queue,   /* command_queue */
                             kernel,          /* kernel */
                             128,             /* image_height */
                             128,             /* image_width */
                             16,              /* in_channels */
                             3,               /* channel_multiplier */
                             3,               /* filter_height */
                             3,               /* filter_width */
                             1,               /* stride */
                             CL_TRUE          /* enable_timing */);
  RunDepthwiseConv2DUnitTest(context,         /* context */
                             command_queue,   /* command_queue */
                             kernel,          /* kernel */
                             128,             /* image_height */
                             128,             /* image_width */
                             16,              /* in_channels */
                             4,               /* channel_multiplier */
                             3,               /* filter_height */
                             3,               /* filter_width */
                             1,               /* stride */
                             CL_TRUE          /* enable_timing */);

  RunDepthwiseConv2DUnitTest(context,         /* context */
                             command_queue,   /* command_queue */
                             kernel,          /* kernel */
                             127,             /* image_height */
                             127,             /* image_width */
                             32,              /* in_channels */
                             2,               /* channel_multiplier */
                             3,               /* filter_height */
                             3,               /* filter_width */
                             1,               /* stride */
                             CL_TRUE          /* enable_timing */);
  RunDepthwiseConv2DUnitTest(context,         /* context */
                             command_queue,   /* command_queue */
                             kernel,          /* kernel */
                             127,             /* image_height */
                             127,             /* image_width */
                             32,              /* in_channels */
                             3,               /* channel_multiplier */
                             3,               /* filter_height */
                             3,               /* filter_width */
                             1,               /* stride */
                             CL_TRUE          /* enable_timing */);
  RunDepthwiseConv2DUnitTest(context,         /* context */
                             command_queue,   /* command_queue */
                             kernel,          /* kernel */
                             127,             /* image_height */
                             127,             /* image_width */
                             32,              /* in_channels */
                             4,               /* channel_multiplier */
                             3,               /* filter_height */
                             3,               /* filter_width */
                             1,               /* stride */
                             CL_TRUE          /* enable_timing */);

  RunDepthwiseConv2DUnitTest(context,         /* context */
                             command_queue,   /* command_queue */
                             kernel,          /* kernel */
                             64,              /* image_height */
                             64,              /* image_width */
                             16,              /* in_channels */
                             2,               /* channel_multiplier */
                             5,               /* filter_height */
                             5,               /* filter_width */
                             1,               /* stride */
                             CL_TRUE          /* enable_timing */);
  RunDepthwiseConv2DUnitTest(context,         /* context */
                             command_queue,   /* command_queue */
                             kernel,          /* kernel */
                             64,              /* image_height */
                             64,              /* image_width */
                             16,              /* in_channels */
                             2,               /* channel_multiplier */
                             7,               /* filter_height */
                             7,               /* filter_width */
                             1,               /* stride */
                             CL_TRUE          /* enable_timing */);
  RunDepthwiseConv2DUnitTest(context,         /* context */
                             command_queue,   /* command_queue */
                             kernel,          /* kernel */
                             65,              /* image_height */
                             65,              /* image_width */
                             11,              /* in_channels */
                             3,               /* channel_multiplier */
                             7,               /* filter_height */
                             7,               /* filter_width */
                             1,               /* stride */
                             CL_TRUE          /* enable_timing */);
}
