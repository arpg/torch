#pragma once

#include <optix_math.h>

namespace torch
{

enum LightType
{
  LIGHT_TYPE_POINT,
  LIGHT_TYPE_COUNT
};

struct PointLightData
{
  float luminance;
  float3 position;
  float3 intensity;
};

} // namespace torch