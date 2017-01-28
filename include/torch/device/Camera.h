#pragma once

#include <optix_math.h>

namespace torch
{

struct CameraData
{
  float2 center;
  float3 position;
  float3 u;
  float3 v;
  float3 w;
};

} // namespace torch