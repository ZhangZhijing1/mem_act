#include "memory_activation.h"

cl_int InitWorkspace(cl_platform_id *platform,
                     cl_device_id *device,
                     cl_context *context,
                     cl_command_queue *command_queue) {
  cl_int status;
  // Get the OpenCL platform
  status = clGetPlatformIDs(1, platform, nullptr);
  if (status != CL_SUCCESS) {
    printf("Error finding platforms\n");
    return status;
  }
  // Query the available devices
  status = clGetDeviceIDs(*platform, CL_DEVICE_TYPE_ALL, 1, device, nullptr);
  if (status != CL_SUCCESS) {
    printf("Error finding devices\n");
    return status;
  }
  // Create the context
  *context = clCreateContext(nullptr, 1, device, nullptr, nullptr, &status);
  if (status != CL_SUCCESS) {
    printf("Couldn't create the context\n");
    return status;
  }
  // Create the command queue
  *command_queue = clCreateCommandQueue(*context, *device, 0, &status);
  if (status != CL_SUCCESS) {
    printf("Couldn't create the command queue\n");
    return status;
  }

  // Display platform name
  std::size_t ext_size;
  clGetPlatformInfo(*platform, CL_PLATFORM_NAME, 128, nullptr, &ext_size);
  std::vector<char> platform_name(ext_size + 1);
  clGetPlatformInfo(*platform, CL_PLATFORM_NAME, 128, platform_name.data(),
                    nullptr);
  platform_name[ext_size] = '\0';
  std::cout << "Using platform " << platform_name.data() << std::endl;

  return CL_SUCCESS;
}

cl_int CreateKernel(cl_context context,
                    cl_device_id device,
                    cl_kernel *kernel,
                    const char *cwd,
                    const char *program_handle,
                    const char *kernel_name,
                    cl_bool binary) {
  std::vector<char> program_path;
  std::vector<char> program_buffer;
  FILE *program_file;
  cl_program program;
  cl_int status;

  for (const char *c = cwd; *c != '\0'; ++c) {
    program_path.push_back(*c);
  }
  for (const char *c = program_handle; *c != '\0'; ++c) {
    program_path.push_back(*c);
  }
  program_path.push_back('\0');
  if (binary) {
    // TODO
    printf("Unimplemented for binary source yet\n");
    return EXIT_FAILURE;
  } else {
    program_file = fopen(program_path.data(), "r");
    if (program_file == nullptr) {
      printf("Error: couldn't open the program file %s\n", program_path.data());
      return EXIT_FAILURE;
    }
    fseek(program_file, 0, SEEK_END);
    size_t program_size = ftell(program_file);
    rewind(program_file);
    program_buffer.resize(program_size + 1);
    program_buffer[program_size] = '\0';
    size_t num_read = fread(program_buffer.data(), sizeof(char), program_size,
                            program_file);
    if (num_read != program_size) {
      printf("Error: could't read the whole program\n");
      return EXIT_FAILURE;
    }
    char *program_buffer_ptr = program_buffer.data();
    program = clCreateProgramWithSource(
        context,                                  /* context */
        1,                                        /* count */
        (const char **)(&program_buffer_ptr),   /* strings */
        nullptr,                                  /* lengths */
        &status                                   /* errcode_ret */);
    if (status != CL_SUCCESS) {
      printf("Error: couldn't create the program\n");
      return status;
    }
    status = clBuildProgram(program, 0, nullptr, "", nullptr, nullptr);
    if (status != CL_SUCCESS) {
      printf("Error: failed to build the program\n");
      std::vector<char> build_log(BUILD_LOG_SIZE);
      clGetProgramBuildInfo(program,
                            device,
                            CL_PROGRAM_BUILD_LOG,
                            build_log.size(),
                            build_log.data(),
                            nullptr);
      printf("--- Build Log ---\n%s\n", build_log.data());
      return status;
    }
  }
  *kernel = clCreateKernel(program, kernel_name, &status);
  clReleaseProgram(program);
  if (status != CL_SUCCESS) {
    printf("Error: failed to create the kernel\n");
    return status;
  }
  return CL_SUCCESS;
}

cl_int CreateKernel(cl_context context,
                    cl_device_id device,
                    std::vector<cl_kernel> &kernels,
                    const char *cwd,
                    const char *program_handle,
                    std::vector<const char *> kernel_names,
                    cl_bool binary) {
  std::vector<char> program_path;
  std::vector<char> program_buffer;
  FILE *program_file;
  cl_program program;
  cl_int status;

  for (const char *c = cwd; *c != '\0'; ++c) {
    program_path.push_back(*c);
  }
  for (const char *c = program_handle; *c != '\0'; ++c) {
    program_path.push_back(*c);
  }
  program_path.push_back('\0');
  if (binary) {
    // TODO
    printf("Unimplemented for binary source yet\n");
    return EXIT_FAILURE;
  } else {
    program_file = fopen(program_path.data(), "r");
    if (program_file == nullptr) {
      printf("Error: couldn't open the program file %s\n", program_path.data());
      return EXIT_FAILURE;
    }
    fseek(program_file, 0, SEEK_END);
    size_t program_size = ftell(program_file);
    rewind(program_file);
    program_buffer.resize(program_size + 1);
    program_buffer[program_size] = '\0';
    size_t num_read = fread(program_buffer.data(), sizeof(char), program_size,
                            program_file);
    if (num_read != program_size) {
      printf("Error: could't read the whole program\n");
      return EXIT_FAILURE;
    }
    char *program_buffer_ptr = program_buffer.data();
    program = clCreateProgramWithSource(
        context,                              /* context */
        1,                                    /* count */
        (const char **)(&program_buffer_ptr), /* strings */
        nullptr,                              /* lengths */
        &status /* errcode_ret */);
    if (status != CL_SUCCESS) {
      printf("Error: couldn't create the program\n");
      return status;
    }
    status = clBuildProgram(program, 0, nullptr, "", nullptr, nullptr);
    if (status != CL_SUCCESS) {
      printf("Error: failed to build the program\n");
      std::vector<char> build_log(BUILD_LOG_SIZE);
      clGetProgramBuildInfo(program,
                            device,
                            CL_PROGRAM_BUILD_LOG,
                            build_log.size(),
                            build_log.data(),
                            nullptr);
      printf("--- Build Log ---\n%s\n", build_log.data());
      return status;
    }
  }
  cl_uint num_kernels = kernel_names.size();
  kernels.resize(num_kernels);
  for (cl_uint i = 0; i < num_kernels; ++i) {
    kernels[i] = clCreateKernel(program, kernel_names[i], &status);
    if (status != CL_SUCCESS) {
      std::cout << "Error: failed to create the kernel " << kernel_names[i]
                << std::endl;
      clReleaseProgram(program);
      return status;
    }
  }
  clReleaseProgram(program);
  return CL_SUCCESS;
}

cl_int SaveKernel(cl_context context,
                  cl_device_id device,
                  cl_kernel *kernel,
                  const char *cwd,
                  const char *program_handle,
                  const char *kernel_name,
                  cl_bool binary) {
  return CL_SUCCESS;
}

cl_uint RoundUp(cl_uint value, cl_uint multiple) {
  cl_uint remainder = value % multiple;
  if (remainder != 0) {
    value += (multiple - remainder);
  }
  return value;
}

