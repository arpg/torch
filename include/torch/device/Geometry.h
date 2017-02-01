#pragma once

#include <torch/device/Core.h>

namespace torch
{

enum GeometryType
{
  GEOM_TYPE_GROUP,
  GEOM_TYPE_MESH,
  GEOM_TYPE_SPHERE,
  GEOM_TYPE_COUNT
};

struct GeometrySample
{
  GeometryType type;
  unsigned int id;
  unsigned int seed;
  float3 origin;
  float3 direction;
  float tmin;
  float tmax;
  float pdf;
};

struct GeometryGroupData
{

};

struct MeshData
{
  // vertices : unsigned int / rtBuffer<float3, 1>
  // normals : unsigned int / rtBuffer<float3, 1>
  // faces : unsigned int / rtBuffer<uint3, 1>
  // T_wl : optix::Matrx4x4
  // T_lw : optix::Matrx4x4
  // area : float
};

struct SphereData
{
  optix::Matrix4x4 T_wl;
  optix::Matrix4x4 T_lw;
  float area;
};

} // namespace torch