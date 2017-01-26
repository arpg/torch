#pragma once

#include <optix_math.h>

namespace torch
{

enum RayType
{
  RAY_TYPE_RADIANCE,
  RAY_TYPE_SHADOW,
  RAY_TYPE_COUNT,
};

struct RadianceData
{
  float3 radiance;
  float3 throughput;
};

} // namespace torch