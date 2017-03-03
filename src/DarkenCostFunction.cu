#include <torch/device/DarkenCostFunction.cuh>
#include <lynx/Exception.h>

namespace torch
{

__global__ void EvaluateKernel(const float* params, float* residual,
    float* jacobian, float* values, float weight, size_t size)
{
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size)
  {
    const float value = values[index];
    const float param = params[index];

    if (value > param)
    {
      residual[index] = weight * (value - param);

      if (jacobian)
      {
        jacobian[index] = -weight;
      }
    }
  }
}

void Evaluate(const float* params, float* residual, float* jacobian,
    float* values, float weight, size_t size)
{
  LYNX_ASSERT(size <= (1024 * 65535), "unsupported launch size");
  const size_t blockDim = (size > 1024) ? 1024 : size;
  const size_t gridDim = (size + blockDim - 1) / blockDim;

  EvaluateKernel<<<blockDim, gridDim>>>(params, residual, jacobian,
      values, weight, size);
}

} // namespace torch