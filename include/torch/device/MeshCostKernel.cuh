#include <cuda_runtime.h>

namespace torch
{

__global__ void EvaluateRKernel(const unsigned int* plist,
    const unsigned int* qlist, const float* weights, const float* references,
    const float* shading, float* residuals, unsigned int count);

__global__ void EvaluateJKernel(const unsigned int* plist,
    const unsigned int* qlist, const float* weights, const float* references,
    const float* shading, const float* coeffs, float* jacobians,
    unsigned int pairCount, unsigned int voxelCount,
    unsigned int vertexCount, unsigned int launchPairCount);

void EvaluateR(const unsigned int* plist, const unsigned int* qlist,
    const float* weights, const float* references, const float* shading,
    float* residuals, unsigned int count);

void EvaluateJ(const unsigned int* plist, const unsigned int* qlist,
    const float* weights, const float* references, const float* shading,
    const float* coeffs, float* jacobians, unsigned int pairCount,
    unsigned int voxelCount, unsigned int vertexCount,
    unsigned int launchPairCount);

} // namespace torch