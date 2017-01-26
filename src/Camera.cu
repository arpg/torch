#include <optix.h>
#include <torch/Random.h>
#include <torch/Ray.h>

struct Camera
{
  float2 center;
  float3 position;
  float3 u;
  float3 v;
  float3 w;
};

rtDeclareVariable(uint2, pixelIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, imageSize, rtLaunchDim, );

rtDeclareVariable(rtObject, sceneRoot, , );
rtDeclareVariable(float, sceneEpsilon, , );

rtDeclareVariable(unsigned int, sampleCount, , );
rtDeclareVariable(Camera, camera, , );
rtBuffer<float3, 2> buffer;

#define TORCH_DEVICE static __device__ __inline__

static __inline__ __device__
void GetDirection(float3& direction, unsigned int& seed)
{
  const float2 size = make_float2(imageSize);
  const float2 pixel = make_float2(pixelIndex) + torch::randf2(seed);
  const float2 ratio = (pixel / size) - camera.center;
  direction = ratio.x * camera.u + ratio.y * camera.v + camera.w;
  direction = normalize(direction);
}

TORCH_DEVICE void InitializeRay(optix::Ray& ray, torch::RadianceData& data)
{
  ray.origin = camera.position;
  ray.tmin = sceneEpsilon;
  ray.tmax = RT_DEFAULT_MAX;
  ray.ray_type = torch::RAY_TYPE_RADIANCE;
}

TORCH_DEVICE unsigned int InitializeSeed()
{
  unsigned int a = pixelIndex.x;
  unsigned int b = pixelIndex.y;
  return torch::init_seed<16>(a, b);
}

RT_PROGRAM void Capture()
{
  optix::Ray ray;
  torch::RadianceData data;
  unsigned int seed;

  seed = InitializeSeed();
  InitializeRay(ray, data);
  data.radiance = make_float3(0, 0, 0);
  data.seed = seed;

  for (unsigned int i = 0; i < sampleCount; ++i)
  {
    GetDirection(ray.direction, seed);
    data.throughput = make_float3(1.0 / sampleCount);
    rtTrace(sceneRoot, ray, data);
    InitializeRay(ray, data);
  }

  buffer[pixelIndex] = data.radiance;
}