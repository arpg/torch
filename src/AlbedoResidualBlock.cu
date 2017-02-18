#include <cfloat>
#include <torch/device/Camera.h>

rtDeclareVariable(unsigned int, launchIndex, rtLaunchIndex, );
rtDeclareVariable(torch::CameraData, camera, , );
rtDeclareVariable(optix::Matrix4x4, Tcw, , );
rtDeclareVariable(optix::Matrix4x4, Twc, , );
rtBuffer<unsigned int> neighborOffsets;
rtBuffer<unsigned int> neighborIndices;
rtBuffer<uint4> boundingBoxes;
rtBuffer<float3> vertices;

TORCH_DEVICE float2 GetImageCoords(unsigned int index)
{
  const float3 Xcp = make_float3(Tcw * make_float4(vertices[index], 1));
  const float3 uvw = camera.K * Xcp;
  return make_float2(uvw.x / uvw.z, uvw.y / uvw.z);
}

TORCH_DEVICE void Union(float4& bbox, const float2& uv)
{
  bbox.x = fminf(uv.x, bbox.x);
  bbox.y = fminf(uv.y, bbox.y);
  bbox.z = fmaxf(uv.x, bbox.z);
  bbox.w = fmaxf(uv.y, bbox.w);
}

RT_PROGRAM void GetBoundingBoxes()
{
  float4 bbox;
  bbox.x = FLT_MAX;
  bbox.y = FLT_MAX;
  bbox.z = FLT_MIN;
  bbox.w = FLT_MIN;

  const unsigned int start = neighborOffsets[launchIndex];
  const unsigned int stop = neighborOffsets[launchIndex + 1];
  Union(bbox, GetImageCoords(launchIndex));

  for (unsigned int i = start; i < stop; ++i)
  {
    unsigned int index = neighborIndices[i];
    Union(bbox, GetImageCoords(index));
  }

  boundingBoxes[launchIndex].x = max(0u, uint(floor(bbox.x)));
  boundingBoxes[launchIndex].y = max(0u, uint(floor(bbox.y)));
  boundingBoxes[launchIndex].z = min(camera.imageSize.x, uint(ceil(bbox.z)));
  boundingBoxes[launchIndex].w = min(camera.imageSize.y, uint(ceil(bbox.w)));
}