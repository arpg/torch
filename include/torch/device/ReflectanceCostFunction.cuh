#include <cuda_runtime.h>

namespace torch
{

__global__ void EvaluateKernel(const float* params, float* residual,
    float* jacobian, size_t size, const uint* map, const uint* offsets,
    float chromThreshold, float weight);

void Evaluate(const float* params, float* residual, float* jacobian,
    size_t size, const uint* map, const uint* offsets, float chromThreshold,
    float weight);

} // namespace torch