#include <torch/device/VoxelActivationCostFunction.cuh>
#include <lynx/Exception.h>

namespace torch
{

__global__ void EvaluateKernel(const float* params, float* residual,
    float* jacobian, size_t size, float bias, float inScale, float outScale)
{
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int offset = 3 * index;
  const float power = params[offset] + params[offset + 1] + params[offset + 2];
  residual[index] = outScale * logf(bias + inScale * power);

  if (jacobian)
  {
    const float deriv = (outScale * inScale) / (bias + inScale * power);
    jacobian[3 * index + 0] = deriv;
    jacobian[3 * index + 1] = deriv;
    jacobian[3 * index + 2] = deriv;
  }
}

void Evaluate(const float* params, float* residual, float* jacobian,
    size_t size, float bias, float inScale, float outScale)
{
  LYNX_ASSERT(size <= (1024 * 65535), "unsupported launch size");
  const size_t blockDim = (size > 1024) ? 1024 : size;
  const size_t gridDim = (size + blockDim - 1) / blockDim;

  EvaluateKernel<<<gridDim, blockDim>>>(params, residual, jacobian, size,
      bias, inScale, outScale);
}

} // namespace torch