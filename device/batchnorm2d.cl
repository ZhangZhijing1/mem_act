__kernel void BatchNorm(__global float *restrict tensor,
                        int batch,
                        int channels,
                        int channel_size,
                        float eps,
                        __constant float *weights,
                        __constant float *biases,
                        float relu) {
  // Index of the channel.
  int channel = get_global_id(0);
  if (channel >= channels) {
    return;
  }
  const int offset = channel * channel_size;
  float mean = 0.f;
  for (int i = 0; i < channel_size; i++) {
    mean += tensor[offset + i];
  }
  mean /= channel_size;
  float var = 0.f;
  for (int i = 0; i < channel_size; i++) {
    float delta = tensor[offset + i] - mean;
    var += delta * delta;
  }
  var /= channel_size;
  var = sqrt(var + eps);

  float weight = weights[channel];
  float bias = biases[channel];
  for (int i = 0; i < channel_size; i++) {
    float activation = (weight * (tensor[offset + i] - mean) / var) + bias;
    if (relu > 0.f) {
      tensor[offset + i] = (activation > 0.f) ? relu * activation : 0.f;
    } else {
      tensor[offset + i] = activation;
    }
  }
}

