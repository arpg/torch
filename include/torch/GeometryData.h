#pragma once

#include <optix_math.h>

namespace torch
{

enum GeometryType
{
  GEOM_TYPE_SPHERE,
  GEOM_TYPE_COUNT
};

struct GeometrySample
{
  unsigned int type;
  unsigned int index;
  unsigned int seed;
  float3 origin;
  float3 position;
  float tmin;
  float pdf;
};

struct SphereData
{
  // transform : transform
  // area : float
};

} // namespace torch