#ifndef HOST_INCLUDE_TENSOR_H_
#define HOST_INCLUDE_TENSOR_H_

#include <CL/cl.h>

#include <fstream>
#include <string>
#include <vector>

#include "workspace.h"

class Tensor {
 public:
  Tensor(const std::vector<int> &shape, bool allocate_device = false,
         Workspace *ws = nullptr);
  Tensor(const std::vector<int> &shape, std::ifstream &is,
         bool allocate_device = false, Workspace *ws = nullptr);
  virtual ~Tensor();

  // Disable copy.
  Tensor(const Tensor &) = delete;
  Tensor(Tensor &&) = delete;
  Tensor &operator=(const Tensor &) = delete;
  Tensor &operator=(Tensor &&) = delete;

  // Element access.
  float &Get(const std::vector<int> &coord);
  const float &Get(const std::vector<int> &coord) const;
  float &Get(int idx);
  const float &Get(int idx) const;
  float &operator[](int idx);
  const float &operator[](int idx) const;

  // Host data access.
  std::vector<float> &GetData();
  const std::vector<float> &GetData() const;

  // Allocate device buffer.
  void AllocateDevice(Workspace &ws, bool copy_host = true,
                      cl_bool blocking = CL_FALSE,
                      cl_uint num_events_in_wait_list = 0,
                      const cl_event *event_wait_list = nullptr,
                      cl_event *event = nullptr);
  // Push host data to device.
  void PushToDevice(Workspace &ws, cl_bool blocking = CL_FALSE,
                    cl_uint num_events_in_wait_list = 0,
                    const cl_event *event_wait_list = nullptr,
                    cl_event *event = nullptr);
  // Pop device data to host.
  void PopToHost(Workspace &ws, cl_bool blocking = CL_FALSE,
                 cl_uint num_events_in_wait_list = 0,
                 const cl_event *event_wait_list = nullptr,
                 cl_event *event = nullptr);

  // Device data access.
  cl_mem &GetDeviceData();
  const cl_mem &GetDeviceData() const;

  // Read data from file.
  void ReadFile(std::ifstream &is, std::size_t size, bool to_device = false,
                Workspace *ws = nullptr, cl_bool blocking = CL_FALSE,
                cl_uint num_events_in_wait_list = 0,
                const cl_event *event_wait_list = nullptr,
                cl_event *event = nullptr);
  // Generate random data.
  void GenerateRandom(float weight, float bias, bool to_device = false,
                      Workspace *ws = nullptr);

 private:
  std::vector<int> shape_;
  int size_;
  std::vector<float> data_;
  bool has_device_data_;
  cl_mem device_data_;
};

#endif  // HOST_INCLUDE_TENSOR_H_

