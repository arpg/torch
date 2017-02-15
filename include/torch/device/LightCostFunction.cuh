#include <cuda_runtime.h>

namespace torch
{

void XEvaluate(const float* params, float* residual, float* jacobian,
    size_t size, float bias, float inScale, float outScale);

} // namespace torch