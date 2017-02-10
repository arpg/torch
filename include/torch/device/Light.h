#pragma once

#include <torch/device/Core.h>
#include <torch/device/Geometry.h>

namespace torch
{

enum LightType
{
  LIGHT_TYPE_AREA,
  LIGHT_TYPE_DIRECTIONAL,
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
  float3 normal;
  float3 snormal;
  float3 throughput;
  float tmin;
  float tmax;
  float pdf;
};

struct AreaLightData
{
  GeometryType geomType;
  unsigned int geomId;
  float3 radiance;
  float area;
  float luminance;
};

struct DirectionalLightData
{
  float3 radiance;
  float3 direction;
  float luminance;
};

struct EnvironmentLightData
{
  optix::Matrix3x3 rotation;
  int distributionId;
  int offsetsId;
  int radianceId;
  float luminance;
};

struct PointLightData
{
  float3 position;
  float3 intensity;
  float luminance;
};

} // namespace torch