#include "test_utils.h"

void CheckResult(float *expected, float *result, size_t size,
                 bool display_errors, float rel_err) {
  int num_errors = 0;
  for (size_t i = 0; i < size; ++i) {
    if (std::abs(expected[i] - result[i]) > rel_err) {
      if (display_errors) {
        std::cout << "Error at " << i << ":\texpected " << expected[i]
                  << ",\tgot " << result[i] << std::endl;
      }
      ++num_errors;
    }
  }
  if (num_errors > 0) {
    std::cout << "Found " << num_errors << " errors\n";
  } else {
    std::cout << "Result is consistent with expected\n";
  }
}

void CheckSimilarity(float *expected, float *result, size_t size,
                     bool display, float abs_err) {
  float norm_a = 0.f, norm_b = 0.f, dot = 0.f;
  for (size_t i = 0; i < size; ++i) {
    float a = expected[i], b = result[i];
    norm_a += a * a;
    norm_b += b * b;
    dot += a * b;
  }
  float similarity = dot / std::sqrt(norm_a * norm_b);
  if (display) {
    std::cout << "Similarity = " << similarity << std::endl;
  }
  if (std::abs(1.0f - similarity) > abs_err) {
    std::cout << "Similarity test failed\n";
  } else {
    std::cout << "Similarity test passed\n";
  }
}
