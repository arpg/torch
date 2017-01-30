#pragma once

#include <torch/device/Core.h>

namespace torch
{

enum GeometryType
{
  GEOM_TYPE_MESH,
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

struct GeometryGroupData
{
  // ???
  // area : float
};

} // namespace torch