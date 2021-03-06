#pragma once

#include <torch/device/Core.h>

namespace torch
{

struct CameraData
{
  uint2 imageSize;
  float2 center;
  float3 position;
  float3 u;
  float3 v;
  float3 w;
  optix::Matrix3x3 K;
  optix::Matrix3x3 Kinv;
  optix::Matrix4x4 Twc;
  optix::Matrix4x4 Tcw;
  unsigned int samples;
  unsigned int maxDepth;
};

struct PixelSample
{
  uint camera;
  uint2 uv;
};

} // namespace torch