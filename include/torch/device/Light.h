#pragma once

#include <optix_math.h>

namespace torch
{

enum LightType
{
  LIGHT_TYPE_AREA,
  LIGHT_TYPE_DISTANT,
  LIGHT_TYPE_POINT,
  LIGHT_TYPE_COUNT
};

struct LightSample
{
  unsigned int seed;
  float3 origin;
  float3 radiance;
  float3 direction;
  float tmin;
  float tmax;
  float pdf;
};

struct AreaLightData
{
  unsigned int geometry;
  float3 radiance;
  float area;
  float luminance;
};

struct DistantLightData
{
  float3 radiance;
  float3 direction;
  float luminance;
};

struct PointLightData
{
  float3 position;
  float3 intensity;
  float luminance;
};

} // namespace torch