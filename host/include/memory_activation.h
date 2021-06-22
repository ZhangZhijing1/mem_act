#ifndef HOST_INCLUDE_MEMORY_ACTIVATION_H_
#define HOST_INCLUDE_MEMORY_ACTIVATION_H_

#include <CL/cl.h>

#include <exception>
#include <iostream>
#include <vector>

#define PATH_SIZE       2048
#define BUILD_LOG_SIZE  2048

struct RandomGenerator {
  RandomGenerator(float a, float b) : a_(a), b_(b) {}

  float operator()(void) {
    return a_ * static_cast<float>(rand() % 1000) + b_;
  }

  float a_, b_;
};

cl_int InitWorkspace(cl_platform_id *platform,
                     cl_device_id *device,
                     cl_context *context,
                     cl_command_queue *command_queue);

cl_int CreateKernel(cl_context context,
                    cl_device_id device,
                    cl_kernel *kernel,
                    const char *cwd,
                    const char *program_handle,
                    const char *kernel_name,
                    cl_bool binary = CL_FALSE);

cl_int CreateKernel(cl_context context,
                    cl_device_id device,
                    std::vector<cl_kernel> &kernels,
                    const char *cwd,
                    const char *program_handle,
                    std::vector<const char *> kernel_names,
                    cl_bool binary = CL_FALSE);

cl_uint RoundUp(cl_uint value, cl_uint multiple);

#define ASSERT(condition, msg)     \
  if (!(condition)) {              \
    throw std::runtime_error(msg); \
  }

#endif  // HOST_INCLUDE_MEMORY_ACTIVATION_H_

