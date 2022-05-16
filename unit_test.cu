#include <iostream>
#include <assert.h>
#include "int_divider.h"

#define checkCUDA(expression)                              \
{                                                          \
  cudaError_t status = (expression);                       \
  if (status != cudaSuccess) {                             \
    std::cerr << "Error on line " << __LINE__ << ": "      \
              << cudaGetErrorString(status) << std::endl;  \
    std::exit(EXIT_FAILURE);                               \
  }                                                        \
}
__global__ void test_division_kernel(const int* numerators, int size_0,
                                     const int* divisors, int size_1,
                                     bool* result) {
  bool pass = true;
  for (int i = 0; i < size_0; i++) {
    for (int j = 0; j < size_1; j++) {
      int ref_div = numerators[i] / divisors[j];
      int ref_mod = numerators[i] % divisors[j];

      FastDividerUint32 divisor(divisors[j]);
      int new_div = numerators[i] / divisor;
      int new_mod = numerators[i] % divisor;
      if (ref_div != new_div || ref_mod != new_mod) {
        printf("Reference: %d /(%%) %d = %d(%d) But, we got: %d(%d)\n",
               numerators[i], divisors[i], ref_div, ref_mod, new_div, new_mod);
        pass = false;
        break;
      }
    }
  }
  *result = pass;
}

bool test_division(const int* numerators, int size_0, const int* divisors,
                   int size_1) {
  bool pass = true;
  for (int i = 0; i < size_0; i++) {
    for (int j = 0; j < size_1; j++) {
      int ref_div = numerators[i] / divisors[j];
      int ref_mod = numerators[i] % divisors[j];

      FastDividerUint32 divisor(divisors[j]);
      int new_div = numerators[i] / divisor;
      int new_mod = numerators[i] % divisor;
      if (ref_div != new_div || ref_mod != new_mod) {
        printf("Reference: %d /(%%) %d = %d(%d) But, we got: %d(%d)\n",
               numerators[i], divisors[i], ref_div, ref_mod, new_div, new_mod);
        pass = false;
        break;
      }
    }
  }
  return pass;
}

int main() {
  const int count = 100;
  int* general_numerators = new int[count];
  int special_numerators[] = {0, 1, INT32_MAX};
  int* general_divisors = new int[count];
  int special_divisors[] = {1, INT32_MAX};
  for (int i = 0; i < count; i++) {
    general_numerators[i] = rand() % INT32_MAX + 1;
    general_divisors[i] = rand() % INT32_MAX + 1;
  }
  // CPU
  bool pass = test_division(general_numerators, count, general_divisors, count);
  printf("CPU Test general_numerators/general_divsors %s!\n",
         (pass ? "PASSED" : "FAILED"));
  pass = test_division(general_numerators, count, special_divisors, 2);
  printf("CPU Test general_numerators/special_divsors %s!\n",
         (pass ? "PASSED" : "FAILED"));
  pass = test_division(special_numerators, 3, general_divisors, count);
  printf("CPU Test special_numerators/general_divsors %s!\n",
         (pass ? "PASSED" : "FAILED"));
  pass = test_division(special_numerators, 3, special_divisors, 2);
  printf("CPU Test special_numerators/special_divsors %s!\n",
         (pass ? "PASSED" : "FAILED"));

  // GPU
  int* d_general_numerators;
  int* d_special_numerators;
  int* d_general_divisors;
  int* d_special_divisors;
  bool* d_result;
  checkCUDA(cudaMalloc((void**)(&d_general_numerators), sizeof(int) * count));
  checkCUDA(cudaMalloc((void**)(&d_special_numerators), sizeof(int) * 3));
  checkCUDA(cudaMalloc((void**)(&d_general_divisors), sizeof(int) * count));
  checkCUDA(cudaMalloc((void**)(&d_special_divisors), sizeof(int) * 2));
  checkCUDA(cudaMalloc((void**)(&d_result), sizeof(bool)));
  checkCUDA(cudaMemcpy(d_general_numerators, general_numerators,
                       sizeof(int) * count, cudaMemcpyHostToDevice));
  checkCUDA(cudaMemcpy(d_special_numerators, special_numerators,
                       sizeof(int) * 3, cudaMemcpyHostToDevice));
  checkCUDA(cudaMemcpy(d_general_divisors, general_divisors,
                       sizeof(int) * count, cudaMemcpyHostToDevice));
  checkCUDA(cudaMemcpy(d_special_divisors, special_divisors,
                       sizeof(int) * 2, cudaMemcpyHostToDevice));
  checkCUDA(cudaMemset(d_result, 0, sizeof(bool)));

  test_division_kernel<<<1, 1>>>(d_general_numerators, count,
                                 d_general_divisors, count, d_result);
  checkCUDA(cudaMemcpy(&pass, d_result, sizeof(bool), cudaMemcpyDeviceToHost));
  printf("GPU Test general_numerators/general_divsors %s!\n",
         (pass ? "PASSED" : "FAILED"));
  test_division_kernel<<<1, 1>>>(d_general_numerators, count,
                                 d_special_divisors, 2, d_result);
  checkCUDA(cudaMemcpy(&pass, d_result, sizeof(bool), cudaMemcpyDeviceToHost));
  printf("GPU Test general_numerators/special_divsors %s!\n",
         (pass ? "PASSED" : "FAILED"));
  test_division_kernel<<<1, 1>>>(d_special_numerators, 3,
                                 d_general_divisors, count, d_result);
  checkCUDA(cudaMemcpy(&pass, d_result, sizeof(bool), cudaMemcpyDeviceToHost));
  printf("GPU Test special_numerators/general_divsors %s!\n",
         (pass ? "PASSED" : "FAILED"));
  test_division_kernel<<<1, 1>>>(d_special_numerators, 3,
                                 d_special_divisors, 2, d_result);
  checkCUDA(cudaMemcpy(&pass, d_result, sizeof(bool), cudaMemcpyDeviceToHost));
  printf("GPU Test special_numerators/special_divsors %s!\n",
         (pass ? "PASSED" : "FAILED"));
}

