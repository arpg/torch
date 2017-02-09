#include <torch/device/Camera.h>

rtDeclareVariable(unsigned int, launchIndex, rtLaunchIndex, );
rtDeclareVariable(torch::CameraData, camera, , );
rtDeclareVariable(optix::Matrix4x4, Tcw, , );
rtDeclareVariable(optix::Matrix4x4, Twc, , );
rtBuffer<unsigned int, 1> neighborOffsets;
rtBuffer<unsigned int, 1> neighborIndices;
rtBuffer<float3, 1> vertices;
rtBuffer<float3, 1> normals;
rtBuffer<uint4, 1> boundingBoxes;

TORCH_DEVICE bool GetImageCoords(unsigned int index, uint2& uv)
{
  const float3 Xcp = make_float3(Tcw * make_float4(vertices[index], 1));
  const float3 Xcn = make_float3(Twc * make_float4(normals[index], 0));

  return false;
}

TORCH_DEVICE void Union(uint4& bbox, const uint2& uv)
{
  bbox.x = (uv.x < bbox.x) ? uv.x : bbox.x;
  bbox.y = (uv.y < bbox.y) ? uv.y : bbox.y;

  bbox.z = (uv.x > bbox.z) ? uv.x : bbox.z;
  bbox.w = (uv.y > bbox.w) ? uv.y : bbox.w;
}

RT_PROGRAM void GetBoundingBoxes()
{
  boundingBoxes[launchIndex].x = camera.imageSize.x;
  boundingBoxes[launchIndex].y = camera.imageSize.y;
  boundingBoxes[launchIndex].z = 0;
  boundingBoxes[launchIndex].w = 0;

  uint2 uv;

  if (GetImageCoords(launchIndex, uv)) Union(boundingBoxes[launchIndex], uv);

  const unsigned int start = neighborOffsets[launchIndex];
  const unsigned int stop = neighborOffsets[launchIndex + 1];

  for (unsigned int i = start; i < stop; ++i)
  {
    if (GetImageCoords(i, uv)) Union(boundingBoxes[launchIndex], uv);
  }
}