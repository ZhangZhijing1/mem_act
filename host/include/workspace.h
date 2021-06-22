#ifndef HOST_INCLUDE_WORKSPACE_H_
#define HOST_INCLUDE_WORKSPACE_H_

#include <CL/cl.h>

#include <memory>
#include <string>

#include "kernel.h"

class Workspace {
 public:
  Workspace(const std::string &platform_name);
  virtual ~Workspace();

  cl_platform_id GetPlatformID();
  const cl_platform_id GetPlatformID() const;

  cl_device_id &GetDeviceID();
  const cl_device_id &GetDeviceID() const;

  cl_context &GetContext();
  const cl_context &GetContext() const;

  cl_command_queue &GetCommandQueue();
  const cl_command_queue &GetCommandQueue() const;
  void FinishCommandQueue();

  Kernel CreateKernel(const char *program_handle, const char *kernel_name,
                      bool binary = false) const;

 private:
  cl_platform_id platform_;
  cl_device_id device_;
  cl_context context_;
  cl_command_queue command_queue_;

  std::unique_ptr<char[]> cwd_;

  void GetPlatform(const std::string &platform_name);
  void GetDevice();
  void CreateContext();
  void CreateCommandQueue();
};

#endif  // HOST_INCLUDE_WORKSPACE_H_

