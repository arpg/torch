#include <torch/device/Ray.h>

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(torch::DepthData, depthData, rtPayload, );
rtDeclareVariable(float, hitDist, rtIntersectionDistance, );

RT_PROGRAM void ClosestHitDepth()
{
  depthData.depth = hitDist;
}