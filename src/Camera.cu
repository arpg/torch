#include <optix.h>
#include <torch/Ray.h>

rtDeclareVariable(uint2, pixelIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, imageSize, rtLaunchDim, );

rtDeclareVariable(rtObject, sceneRoot, , );
rtDeclareVariable(float, sceneEpsilon, , );

rtDeclareVariable(float3, position, , );
rtDeclareVariable(float2, center, , );
rtDeclareVariable(float3, u, , );
rtDeclareVariable(float3, v, , );
rtDeclareVariable(float3, w, , );
rtBuffer<float3, 2> buffer;

#define TORCH_DEVICE static __device__ __inline__

static __inline__ __device__
void GetDirection(float3& direction)
{
  const float2 pixel = make_float2(pixelIndex);
  const float2 size = make_float2(imageSize);
  const float2 ratio = (pixel / size) - center;
  direction = ratio.x * u + ratio.y * v + w;
  direction = normalize(direction);
}

TORCH_DEVICE void InitializeRay(optix::Ray& ray, torch::RadianceData& data)
{
  ray.origin = position;
  ray.tmin = sceneEpsilon;
  ray.tmax = RT_DEFAULT_MAX;
  ray.ray_type = torch::RAY_TYPE_RADIANCE;
  data.radiance = make_float3(0, 0, 0);
  GetDirection(ray.direction);
}

RT_PROGRAM void Capture()
{
  optix::Ray ray;
  torch::RadianceData data;
  InitializeRay(ray, data);
  rtTrace(sceneRoot, ray, data);
  buffer[pixelIndex] = data.radiance;
}