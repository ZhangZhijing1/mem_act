#ifndef HOST_INCLUDE_TEST_UTILS_H_
#define HOST_INCLUDE_TEST_UTILS_H_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <functional>
#include <iostream>
#include <numeric>
#include <ratio>

void CheckResult(float *expected, float *result, size_t size,
                 bool display_errors = false, float rel_err = 1e-3f);

void CheckSimilarity(float *expected, float *result, size_t size,
                     bool display = false, float abs_err = 1e-2f);

#endif  //  HOST_INCLUDE_TEST_UTILS_H_