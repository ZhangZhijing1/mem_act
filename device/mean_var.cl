__kernel void SumRow(__global float * restrict image,
                     __global float * restrict row_cache,
                     int image_height,
                     int image_width,
                     int channels) {
  // row index
  int y = get_global_id(0);
  // channel index
  int z = get_global_id(1);
  if (y >= image_height || z >= channels) {
    return;
  }
  int in_offset = z * image_height * image_width + y * image_width;
  int out_idx = z * image_height + y;
  float sum = 0.f;
  for (int c = 0; c < image_width; ++c) {
    sum += image[in_offset + c];
  }
  row_cache[out_idx] = sum;
}

__kernel void SumCol(__global float * restrict row_cache,
                     __global float * restrict result,
                     int image_height,
                     int image_width,
                     int channels) {
  // channel index
  int z = get_global_id(0);
  if (z >= channels) {
    return;
  }
  int image_size = image_height * image_width;
  int in_offset = z * image_height;
  float sum = 0.f;
  for (int r = 0; r < image_height; ++r) {
    sum += row_cache[in_offset + r];
  }
  result[z] = sum / image_size;
}

__kernel void VarRow(__global float * restrict image,
                     __global float * restrict mean,
                     __global float * restrict row_cache,
                     int image_height,
                     int image_width,
                     int channels) {
  // row index
  int y = get_global_id(0);
  // channel index
  int z = get_global_id(1);
  if (y >= image_height || z >= channels) {
    return;
  }
  int in_offset = z * image_height * image_width + y * image_width;
  int out_idx = z * image_height + y;
  float sum = 0.f;
  float channel_mean = mean[z];
  for (int c = 0; c < image_width; ++c) {
    float delta = image[in_offset + c] - channel_mean;
    sum += delta * delta;
  }
  row_cache[out_idx] = sum;
}

__kernel void VarCol(__global float * restrict row_cache,
                     __global float * restrict result,
                     float eps,
                     int image_height,
                     int image_width,
                     int channels) {
  // channel index
  int z = get_global_id(0);
  if (z >= channels) {
    return;
  }
  int image_size = image_height * image_width;
  int in_offset = z * image_height;
  float sum = 0.f;
  for (int r = 0; r < image_height; ++r) {
    sum += row_cache[in_offset + r];
  }
  result[z] = sqrt((sum / image_size) + eps);
}
