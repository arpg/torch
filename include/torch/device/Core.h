#pragma once

#include <optixu/optixu_aabb.h>
#include <optixu/optixu_matrix.h>

#define TORCH_MEMBER __inline__ __host__ __device__
#define TORCH_HOSTDEVICE static __inline__ __host__ __device__
#define TORCH_DEVICE static __inline__ __device__
#define TORCH_HOST static __inline__ __host__

namespace torch
{

} // namespace torch