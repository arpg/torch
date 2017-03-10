#include <torch/device/MeshCostKernel.cuh>
#include <lynx/lynx.h>

namespace torch
{

__global__ void EvaluateRKernel(const unsigned int* plist,
    const unsigned int* qlist, const float* weights, const float* references,
    const float* shading, float* residuals, unsigned int count)
{
  const unsigned int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadIndex < count)
  {
    // const float epsilon = 1E-6f;
    const double epsilon = 1E-6f;
    const unsigned int channel = threadIndex % 3;
    const unsigned int p = plist[threadIndex / 3];
    const unsigned int q = qlist[threadIndex / 3];
    // const float w = weights[threadIndex / 3];
    // const float ip = references[3 * p + channel];
    // const float iq = references[3 * q + channel];
    // const float sp = logf(shading[3 * p + channel] + epsilon);
    // const float sq = logf(shading[3 * q + channel] + epsilon);

    const double w = weights[threadIndex / 3];
    const double ip = references[3 * p + channel];
    const double iq = references[3 * q + channel];
    // const double sp = log(shading[3 * p + channel] + epsilon);
    // const double sq = log(shading[3 * q + channel] + epsilon);
    const double sp = log(fmaxf(0.0f, shading[3 * p + channel]) + epsilon);
    const double sq = log(fmaxf(0.0f, shading[3 * q + channel]) + epsilon);

    residuals[threadIndex] = w * ((ip - sp) - (iq - sq));
  }
}

__global__ void EvaluateJKernel(const unsigned int* plist,
    const unsigned int* qlist, const float* weights, const float* references,
    const float* shading, const float* coeffs, float* jacobians,
    unsigned int pairCount, unsigned int voxelCount,
    unsigned int vertexCount, unsigned int launchPairCount)
{
  const unsigned int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int maxThreadIndex = 3 * launchPairCount * voxelCount;

  // TODO: half computation as deriv of q is -deriv of p

  if (threadIndex < maxThreadIndex)
  {
    // const float epsilon = 1E-6f;
    const double epsilon = 1E-6f;
    const unsigned int pairIndex = (threadIndex / 3) % launchPairCount;
    const unsigned int voxelIndex = (threadIndex / 3) / launchPairCount;
    const unsigned int channel = threadIndex % 3;

    const unsigned int p = plist[pairIndex];
    const unsigned int q = qlist[pairIndex];
    // const float w = weights[pairIndex];
    // const float Sp = shading[3 * p + channel] + epsilon;
    // const float Sq = shading[3 * q + channel] + epsilon;
    // const float cp = -coeffs[3 * voxelIndex * vertexCount + 3 * p + channel];
    // const float cq = -coeffs[3 * voxelIndex * vertexCount + 3 * q + channel];
    const double w = weights[pairIndex];
    const double Sp = fmaxf(0.0f, shading[3 * p + channel]) + epsilon;
    const double Sq = fmaxf(0.0f, shading[3 * q + channel]) + epsilon;
    const double cp = -coeffs[3 * voxelIndex * vertexCount + 3 * p + channel];
    const double cq = -coeffs[3 * voxelIndex * vertexCount + 3 * q + channel];

    const unsigned int jacobIndex =
        3 * (voxelIndex * launchPairCount + pairIndex) + channel;
    // const unsigned int jacobIndex =
    //     3 * (pairIndex * voxelCount + voxelIndex) + channel;

    jacobians[jacobIndex] = -w * ((cp / Sp) - (cq / Sq));

    // printf("%d: (%d %d) (%f) (%f %f) (%f %f)\n", threadIndex,
    //        p, q, w, cp, cq, Sp, Sq);
  }
}

void EvaluateR(const unsigned int* plist, const unsigned int* qlist,
    const float* weights, const float* references, const float* shading,
    float* residuals, unsigned int count)
{
  LYNX_ASSERT(count <= (1024 * 65535), "unsupported launch size");
  const size_t blockDim = (count > 1024) ? 1024 : count;
  const size_t gridDim = (count + blockDim - 1) / blockDim;

  EvaluateRKernel<<<gridDim, blockDim>>>(plist, qlist, weights, references,
      shading, residuals, count);
}

void EvaluateJ(const unsigned int* plist, const unsigned int* qlist,
    const float* weights, const float* references, const float* shading,
    const float* coeffs, float* jacobians, unsigned int pairCount,
    unsigned int voxelCount, unsigned int vertexCount,
    unsigned int launchPairCount)
{
  const size_t count = 3 * launchPairCount * voxelCount;
  LYNX_ASSERT(count <= (1024 * 65535), "unsupported launch size");
  const size_t blockDim = (count > 1024) ? 1024 : count;
  const size_t gridDim = (count + blockDim - 1) / blockDim;

  EvaluateJKernel<<<gridDim, blockDim>>>(plist, qlist, weights, references,
      shading, coeffs, jacobians, pairCount, voxelCount, vertexCount,
      launchPairCount);
}

} // namespace torch