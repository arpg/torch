#pragma once

#include <torch/device/Core.h>

namespace torch
{

enum LightType
{
  LIGHT_TYPE_AREA,
  LIGHT_TYPE_DISTANT,
  LIGHT_TYPE_ENVIRONMENT,
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

struct EnvironmentLightData
{
  unsigned int rowCount;
  unsigned int* offsets;
  optix::Matrix3x3 rotation;
  float3* radiance;
  float luminance;
};

struct PointLightData
{
  float3 position;
  float3 intensity;
  float luminance;
};

} // namespace torch