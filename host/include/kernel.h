#ifndef HOST_INCLUDE_KERNEL_H_
#define HOST_INCLUDE_KERNEL_H_

#include <CL/cl.h>

class Kernel {
 public:
  explicit Kernel(cl_kernel kernel);
  ~Kernel();

  cl_kernel &Get();
  const cl_kernel &Get() const;

 private:
  cl_kernel kernel_;
};

#endif  // HOST_INCLUDE_KERNEL_H_

