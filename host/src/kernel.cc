#include "kernel.h"

Kernel::Kernel(cl_kernel kernel) : kernel_(kernel) {}

Kernel::~Kernel() {
  if (kernel_ != nullptr) {
    clReleaseKernel(kernel_);
  }
}

cl_kernel &Kernel::Get() {
  return kernel_;
}

const cl_kernel &Kernel::Get() const {
  return kernel_;
}

