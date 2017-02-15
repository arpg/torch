#include <cuda_runtime.h>

namespace torch
{

__global__ void EvaluateKernel(const float* params, float* residual,
    float* jacobian, size_t size);

void Evaluate(const float* params, float* residual, float* jacobian,
    size_t size);

} // namespace torch