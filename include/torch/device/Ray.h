#pragma once

#include <torch/device/Core.h>

namespace torch
{

enum RayType
{
  RAY_TYPE_RADIANCE,
  RAY_TYPE_SHADOW,
  RAY_TYPE_COUNT,
};

struct RayBounce
{
  float3 origin;
  float3 direction;
  float3 throughput;
};

struct RadianceData
{
  float3 radiance;
  float3 throughput;
  unsigned int seed;
  RayBounce bounce;
  unsigned int depth;
};

struct ShadowData
{
  bool occluded;
};

} // namespace torch