#include <cuda_runtime.h>

namespace torch
{

__global__ void EvaluateKernel(const float* params, float* residual,
    float* jacobian, size_t size, float bias, float inScale,
    float outScale);

void Evaluate(const float* params, float* residual, float* jacobian,
    size_t size, float bias, float inScale, float outScale);

} // namespace torch