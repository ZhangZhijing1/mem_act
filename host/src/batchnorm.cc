#include "batchnorm.h"

#include <cmath>

void RunBatchNormRef(std::vector<float> &tensor,
                     int batch,
                     int channels,
                     int channel_size,
                     float eps,
                     const std::vector<float> &weights,
                     const std::vector<float> &biases,
                     float relu) {
  std::vector<float> mean(channels, 0.f);
  std::vector<float> var(channels, 0.f);
  int idx = 0;
  for (int c = 0; c < channels; c++) {
    for (int i = 0; i < channel_size; i++) {
      mean[c] += tensor[idx++];
    }
    mean[c] /= channel_size;
  }
  idx = 0;
  for (int c = 0; c < channels; c++) {
    for (int i = 0; i < channel_size; i++) {
      float delta = tensor[idx++] - mean[c];
      var[c] += delta * delta;
    }
    var[c] /= channel_size;
    var[c] = std::sqrt(var[c] + eps);
  }
  idx = 0;
  for (int c = 0; c < channels; c++) {
    for (int i = 0; i < channel_size; i++) {
      float activation =
          (weights[c] * (tensor[idx] - mean[c]) / var[c]) + biases[c];
      if (relu > 0.f) {
        tensor[idx] = (activation > 0.f) ? relu * activation : 0.f;
      } else {
        tensor[idx] = activation;
      }
      idx++;
    }
  }
}

void RunBatchNormRef(std::vector<float> &tensor,
                     const std::vector<int> &tensor_shape,
                     float eps,
                     const std::vector<float> &weights,
                     const std::vector<float> &biases,
                     float relu) {
  RunBatchNormRef(tensor, tensor_shape[0], tensor_shape[1],
                  tensor_shape[2] * tensor_shape[3], eps, weights, biases,
                  relu);
}

