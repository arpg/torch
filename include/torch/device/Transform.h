#pragma once

#include <torch/device/Core.h>

rtDeclareVariable(optix::Matrix4x4, T_wl, , );
rtDeclareVariable(optix::Matrix4x4, T_lw, , );

TORCH_DEVICE float3 PointToLocal(const float3& p)
{
  return make_float3(T_lw * make_float4(p, 1));
}

TORCH_DEVICE float3 PointToWorld(const float3& p)
{
  return make_float3(T_wl * make_float4(p, 1));
}

TORCH_DEVICE float3 VectorToLocal(const float3& v)
{
  return make_float3(T_lw * make_float4(v, 0));
}

TORCH_DEVICE float3 VectorToWorld(const float3& v)
{
  return make_float3(T_wl * make_float4(v, 0));
}

TORCH_DEVICE float3 NormalToLocal(const float3& n)
{
  return make_float3(T_wl.transpose() * make_float4(n, 0));
}

TORCH_DEVICE float3 NormalToWorld(const float3& n)
{
  return make_float3(T_lw.transpose() * make_float4(n, 0));
}

TORCH_DEVICE void BoundsToWorld(const float3& bmin, const float3& bmax,
    float bounds[6])
{
  optix::Aabb* aabb = (optix::Aabb*)bounds;
  aabb->invalidate();

  aabb->include(PointToWorld(make_float3(bmin.x, bmin.y, bmin.z)));
  aabb->include(PointToWorld(make_float3(bmin.x, bmin.y, bmax.z)));
  aabb->include(PointToWorld(make_float3(bmin.x, bmax.y, bmin.z)));
  aabb->include(PointToWorld(make_float3(bmin.x, bmax.y, bmax.z)));
  aabb->include(PointToWorld(make_float3(bmax.x, bmin.y, bmin.z)));
  aabb->include(PointToWorld(make_float3(bmax.x, bmin.y, bmax.z)));
  aabb->include(PointToWorld(make_float3(bmax.x, bmax.y, bmin.z)));
  aabb->include(PointToWorld(make_float3(bmax.x, bmax.y, bmax.z)));
}