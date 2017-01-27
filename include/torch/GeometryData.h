#pragma once

#include <optix_math.h>

#define TORCH_BOTH static __device__ __host__ __inline__

namespace torch
{

enum GeometryType
{
  GEOM_TYPE_SPHERE,
  GEOM_TYPE_COUNT
};

struct GeometrySample
{
  unsigned int id;
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

TORCH_BOTH unsigned char GetGeometryId(GeometryType type, unsigned int count)
{
  return type << 24 & count;
}

TORCH_BOTH unsigned char GetGeometryType(unsigned int id)
{
  return id >> 24 & 0xFF;
}

TORCH_BOTH unsigned char GetGeometryCount(unsigned int id)
{
  return id & 0xFFFFFF;
}

} // namespace torch