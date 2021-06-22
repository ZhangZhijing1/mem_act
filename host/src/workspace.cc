#include "workspace.h"

#include <unistd.h>

#include <cstring>
#include <functional>
#include <memory>
#include <vector>

#include "memory_activation.h"

const int kMaxNumPlatforms = 8;

Workspace::Workspace(const std::string &platform_name)
    : platform_(nullptr),
      device_(nullptr),
      context_(nullptr),
      command_queue_(nullptr) {
  // Get the OpenCL platform.
  GetPlatform(platform_name);
  // Get the device.
  GetDevice();
  // Create the context.
  CreateContext();
  // Create the command queue.
  CreateCommandQueue();

  // Get the current work directory.
  cwd_.reset(new char[PATH_SIZE]);
  getcwd(cwd_.get(), PATH_SIZE);
}

Workspace::~Workspace() {
  if (command_queue_ != nullptr) {
    clReleaseCommandQueue(command_queue_);
  }
  if (context_ != nullptr) {
    clReleaseContext(context_);
  }
  if (device_ != nullptr) {
    clReleaseDevice(device_);
  }
}

void Workspace::GetPlatform(const std::string &platform_name) {
  cl_int status;
  cl_uint num_platforms;
  status = clGetPlatformIDs(kMaxNumPlatforms, nullptr, &num_platforms);
  ASSERT(status == CL_SUCCESS, "Couldn't get the number of platforms");
  std::vector<cl_platform_id> platforms(num_platforms);
  status = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
  ASSERT(status == CL_SUCCESS, "Couldn't get the platform IDs");
  for (cl_uint i = 0; i < num_platforms; i++) {
    std::size_t ext_size;
    status = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, nullptr,
                               &ext_size);
    std::vector<char> ext(ext_size);
    status = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, ext_size,
                               ext.data(), nullptr);
    if (strcmp(ext.data(), platform_name.c_str()) == 0) {
      platform_ = platforms[i];
      break;
    }
  }
  ASSERT(platform_ != nullptr, "Couldn't find the platform " + platform_name);
}

void Workspace::GetDevice() {
  cl_int status;
  status = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_ALL, 1, &device_, nullptr);
  ASSERT(status == CL_SUCCESS, "Couldn't get the device");
}

void Workspace::CreateContext() {
  cl_int status;
  context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &status);
  ASSERT(status == CL_SUCCESS, "Couldn't create the context");
}

void Workspace::CreateCommandQueue() {
  cl_int status;
  command_queue_ = clCreateCommandQueue(context_, device_, 0, &status);
  ASSERT(status == CL_SUCCESS, "Couldn't create the command queue");
}

cl_platform_id Workspace::GetPlatformID() {
  return platform_;
}

const cl_platform_id Workspace::GetPlatformID() const {
  return platform_;
}

cl_device_id &Workspace::GetDeviceID() {
  return device_;
}

const cl_device_id &Workspace::GetDeviceID() const {
  return device_;
}

cl_context &Workspace::GetContext() {
  return context_;
}

const cl_context &Workspace::GetContext() const {
  return context_;
}

cl_command_queue &Workspace::GetCommandQueue() {
  return command_queue_;
}

const cl_command_queue &Workspace::GetCommandQueue() const {
  return command_queue_;
}

void Workspace::FinishCommandQueue() {
  clFinish(command_queue_);
}

Kernel Workspace::CreateKernel(const char *program_handle,
                               const char *kernel_name, bool binary) const {
  cl_kernel kernel;
  std::vector<char> program_path;
  std::vector<char> program_buffer;
  FILE *program_file;
  cl_program program = nullptr;
  cl_int status;

  std::unique_ptr<cl_program, std::function<void(cl_program *)>>
      program_deleter(&program, [&](cl_program *prog_ptr) {
        if ((prog_ptr != nullptr) && (*prog_ptr != nullptr)) {
          clReleaseProgram(*prog_ptr);
        }
      });

  const char *cwd = cwd_.get();
  for (const char *c = cwd; *c != '\0'; ++c) {
    program_path.push_back(*c);
  }
  for (const char *c = program_handle; *c != '\0'; ++c) {
    program_path.push_back(*c);
  }
  program_path.push_back('\0');
  if (binary) {
    // TODO
    ASSERT(false, "Unimplemented for binary source yet\n");
  } else {
    program_file = fopen(program_path.data(), "r");
    ASSERT(program_file != nullptr, "Error: couldn't open the program file ");
    fseek(program_file, 0, SEEK_END);
    size_t program_size = ftell(program_file);
    rewind(program_file);
    program_buffer.resize(program_size + 1);
    program_buffer[program_size] = '\0';
    size_t num_read = fread(program_buffer.data(), sizeof(char), program_size,
                            program_file);
    ASSERT(num_read == program_size, "Error: could't read the whole program");
    char *program_buffer_ptr = program_buffer.data();
    program = clCreateProgramWithSource(
        context_,                             /* context */
        1,                                    /* count */
        (const char **)(&program_buffer_ptr), /* strings */
        nullptr,                              /* lengths */
        &status /* errcode_ret */);
    ASSERT(status == CL_SUCCESS, "Error: couldn't create the program");
    status = clBuildProgram(program, 0, nullptr, "", nullptr, nullptr);
    if (status != CL_SUCCESS) {
      std::vector<char> build_log(BUILD_LOG_SIZE);
      clGetProgramBuildInfo(program,
                            device_,
                            CL_PROGRAM_BUILD_LOG,
                            build_log.size(),
                            build_log.data(),
                            nullptr);
      printf("--- Build Log ---\n%s\n", build_log.data());
      ASSERT(false, "Error: failed to build the program");
    }
  }
  kernel = clCreateKernel(program, kernel_name, &status);
  ASSERT(status == CL_SUCCESS, "Error: failed to create the kernel");
  return Kernel{kernel};
}

