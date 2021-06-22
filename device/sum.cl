__kernel void sum(__global const float *input,
                  __global float *partial_sums,
                  __local float *local_sums) {
  uint global_id = get_global_id(0);
  uint local_id = get_local_id(0);
  uint group_size = get_local_size(0);
  uint group_id = get_group_id(0);
  
  // Copy from global to local memory
  local_sums[local_id] = input[global_id];
  // Loop for computing local sums: divide work-group into 2 parts
  for (uint stride = group_size / 2; stride > 0; stride /= 2) {
    // Waiting for each 2x2 addition into given work-group
    barrier(CLK_LOCAL_MEM_FENCE);
    // Add elements 2 by 2 between local_id and local_id + stride
    if (local_id < stride) {
      local_sums[local_id] += local_sums[local_id + stride];
    }
  }
  // Write result into partial_sums[n_work_group]
  if (local_id == 0) {
    partial_sums[group_id] = local_sums[0];
  }
}