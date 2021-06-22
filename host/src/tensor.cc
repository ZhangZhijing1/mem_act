#include "tensor.h"

#include <algorithm>
#include <ctime>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>

#include "memory_activation.h"

Tensor::Tensor(const std::vector<int> &shape, bool allocate_device,
               Workspace *ws)
    : shape_(shape), has_device_data_(allocate_device) {
  size_ = std::accumulate(shape.begin(), shape.end(), 1,
                          std::multiplies<int>());
  data_.resize(size_, 0.f);
  if (allocate_device) {
    ASSERT(ws != nullptr, "Workspace is null");
    device_data_ = clCreateBuffer(ws->GetContext(), CL_MEM_READ_WRITE,
                                  size_ * sizeof(float), nullptr, nullptr);
  }
}

Tensor::Tensor(const std::vector<int> &shape, std::ifstream &is,
               bool allocate_device, Workspace *ws)
    : shape_(shape), has_device_data_(allocate_device) {
  size_ = std::accumulate(shape.begin(), shape.end(), 1,
                          std::multiplies<int>());
  std::size_t raw_size = size_ * sizeof(float);
  data_.resize(size_);
  is.read(reinterpret_cast<char *>(&data_[0]), raw_size);
  if (allocate_device) {
    ASSERT(ws != nullptr, "Workspace is null");
    device_data_ = clCreateBuffer(ws->GetContext(),
                                  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  raw_size, data_.data(), nullptr);
  }
}

Tensor::~Tensor() {
  if (has_device_data_) {
    clReleaseMemObject(device_data_);
  }
}

float &Tensor::Get(const std::vector<int> &coord) {
  ASSERT(coord.size() == shape_.size(),
         "Coordinate vector must have size " + std::to_string(shape_.size()));
  for (int i = 0; i < shape_.size(); i++) {
    ASSERT(coord[i] >= 0, "Coordinate " + std::to_string(i) + " is negative");
    ASSERT(coord[i] < shape_[i], "Index " + std::to_string(i) +
                                     " ( = " + std::to_string(coord[i]) +
                                     " out of range");
  }
  int idx = std::accumulate(coord.begin(), coord.end(), 1,
                            std::multiplies<int>());
  return data_[idx];
}

const float &Tensor::Get(const std::vector<int> &coord) const {
  ASSERT(coord.size() == shape_.size(),
         "Coordinate vector must have size " + std::to_string(shape_.size()));
  for (int i = 0; i < shape_.size(); i++) {
    ASSERT(coord[i] >= 0, "Coordinate " + std::to_string(i) + " is negative");
    ASSERT(coord[i] < shape_[i], "Coordinate " + std::to_string(i) +
                                     " ( = " + std::to_string(coord[i]) +
                                     ") must be less than " +
                                     std::to_string(shape_[i]));
  }
  int idx = std::accumulate(coord.begin(), coord.end(), 1,
                            std::multiplies<int>());
  return data_[idx];
}

float &Tensor::Get(int idx) {
  ASSERT(idx >= 0, "Index is negative");
  ASSERT(idx < size_, "Index out of range");
  return data_[idx];
}

const float &Tensor::Get(int idx) const {
  ASSERT(idx >= 0, "Index is negative");
  ASSERT(idx < size_, "Index out of range");
  return data_[idx];
}

float &Tensor::operator[](int idx) {
  ASSERT(idx >= 0, "Index is negative");
  ASSERT(idx < size_, "Index out of range");
  return data_[idx];
}

const float &Tensor::operator[](int idx) const {
  ASSERT(idx >= 0, "Index is negative");
  ASSERT(idx < size_, "Index out of range");
  return data_[idx];
}

std::vector<float> &Tensor::GetData() {
  return data_;
}

const std::vector<float> &Tensor::GetData() const {
  return data_;
}

void Tensor::AllocateDevice(Workspace &ws, bool copy_host, cl_bool blocking,
                            cl_uint num_events_in_wait_list,
                            const cl_event *event_wait_list, cl_event *event) {
  int status;
  std::size_t raw_size = size_ * sizeof(float);
  if (!has_device_data_) {
    if (copy_host) {
      device_data_ = clCreateBuffer(
          ws.GetContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
          raw_size, data_.data(), &status);
    } else {
      device_data_ = clCreateBuffer(ws.GetContext(), CL_MEM_READ_WRITE,
                                    raw_size, nullptr, &status);
    }
    has_device_data_ = true;
  } else {
    if (copy_host) {
      status = clEnqueueWriteBuffer(
          ws.GetCommandQueue(),
          device_data_,
          blocking,
          0,
          raw_size,
          data_.data(),
          num_events_in_wait_list,
          event_wait_list, event);
    }
  }
  ASSERT(status == CL_SUCCESS, "Failed to allocate or push device data");
}

void Tensor::PushToDevice(Workspace &ws, cl_bool blocking,
                          cl_uint num_events_in_wait_list,
                          const cl_event *event_wait_list, cl_event *event) {
  cl_int status;
  std::size_t raw_size = size_ * sizeof(float);
  if (!has_device_data_) {
    device_data_ = clCreateBuffer(ws.GetContext(),
                                  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  raw_size, data_.data(), &status);
    has_device_data_ = true;
  } else {
    status = clEnqueueWriteBuffer(
        ws.GetCommandQueue(), device_data_, blocking, 0, raw_size, data_.data(),
        num_events_in_wait_list, event_wait_list, event);
  }
  ASSERT(status == CL_SUCCESS, "Failed to push data to device");
}

void Tensor::PopToHost(Workspace &ws, cl_bool blocking,
                       cl_uint num_events_in_wait_list,
                       const cl_event *event_wait_list, cl_event *event) {
  cl_int status;
  ASSERT(has_device_data_, "The tensor doesn't have device data");
  status = clEnqueueReadBuffer(ws.GetCommandQueue(), device_data_, blocking, 0,
                               size_ * sizeof(float), data_.data(),
                               num_events_in_wait_list, event_wait_list, event);
}

void Tensor::ReadFile(std::ifstream &is, std::size_t size, bool to_device,
                      Workspace *ws, cl_bool blocking,
                      cl_uint num_events_in_wait_list,
                      const cl_event *event_wait_list, cl_event *event) {
  ASSERT(size <= size_, "Size to read is too large");
  std::size_t raw_size = size * sizeof(float);
  is.read(reinterpret_cast<char *>(&data_[0]), raw_size);
  if (to_device) {
    cl_int status;
    ASSERT(ws != nullptr, "Workspace is null");
    std::size_t raw_size = size_ * sizeof(float);
    if (!has_device_data_) {
      device_data_ = clCreateBuffer(ws->GetContext(),
                                    CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                    raw_size, data_.data(), &status);
      has_device_data_ = true;
    } else {
      status = clEnqueueWriteBuffer(
          ws->GetCommandQueue(), device_data_, CL_TRUE, 0,
          raw_size, data_.data(), num_events_in_wait_list,
          event_wait_list, event);
    }
    ASSERT(status == CL_SUCCESS, "Failed to push data to device");
  }
}

void Tensor::GenerateRandom(float weight, float bias, bool to_device,
                            Workspace *ws) {
  static unsigned int seed = std::time(nullptr);
  static const int kMask = (1 << 10) - 1;
  std::generate(data_.begin(), data_.end(), [=](void) -> float {
    return static_cast<float>(rand() & kMask) * weight + bias;
  });
  if (to_device) {
    cl_int status;
    ASSERT(ws != nullptr, "Workspace is null");
    std::size_t raw_size = size_ * sizeof(float);
    if (!has_device_data_) {
      device_data_ = clCreateBuffer(ws->GetContext(),
                                    CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                    raw_size, data_.data(), &status);
      has_device_data_ = true;
    } else {
      status = clEnqueueWriteBuffer(
          ws->GetCommandQueue(), device_data_, CL_TRUE, 0,
          raw_size, data_.data(), 0, nullptr, nullptr);
    }
    ASSERT(status == CL_SUCCESS, "Failed to push data to device");
  }
}

cl_mem &Tensor::GetDeviceData() {
  return device_data_;
}

const cl_mem &Tensor::GetDeviceData() const {
  return device_data_;
}

