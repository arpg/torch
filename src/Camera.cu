#include <optix.h>
#include <torch/device/Core.h>
#include <torch/device/Camera.h>
#include <torch/device/Random.h>
#include <torch/device/Ray.h>

rtDeclareVariable(uint2, pixelIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, imageSize, rtLaunchDim, );

rtDeclareVariable(rtObject, sceneRoot, , );
rtDeclareVariable(float, sceneEpsilon, , );

rtDeclareVariable(unsigned int, sampleCount, , );
rtDeclareVariable(torch::CameraData, camera, , );
rtBuffer<float3, 2> buffer;

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

TORCH_DEVICE float GetY(const float3& L)
{
  return 0.212671f * L.x + 0.715160f * L.y + 0.072169f * L.z;
}

RT_PROGRAM void Capture()
{
  optix::Ray ray;
  torch::RadianceData data;
  unsigned int seed;

  seed = InitializeSeed();
  InitializeRay(ray, data);
  data.radiance = make_float3(0, 0, 0);

  for (unsigned int i = 0; i < sampleCount; ++i)
  {
    InitializeRay(ray, data);
    GetDirection(ray.direction, seed);
    data.bounce.origin = make_float3(0, 0, 0);
    data.bounce.direction = make_float3(0, 0, 0);
    data.bounce.throughput = make_float3(0, 0, 0);
    data.throughput = make_float3(1.0f / sampleCount);

    for (unsigned int depth = 0; depth < 8; ++depth)
    {
      data.depth = depth;
      data.seed = seed;
      rtTrace(sceneRoot, ray, data);
      InitializeRay(ray, data);
      seed = data.seed;

      torch::RayBounce& bounce = data.bounce;

      if (length(bounce.direction) < 1E-8)
      {
        break;
      }

      if (depth > 3)
      {
        const float continueProb = min(0.5f, GetY(bounce.throughput));

        if (torch::randf(seed) > continueProb)
        {
          break;
        }

        bounce.throughput /= continueProb;
      }

      ray.origin = data.bounce.origin;
      ray.direction = data.bounce.direction;
      data.throughput = data.bounce.throughput;
    }
  }

  buffer[pixelIndex] = data.radiance;
}