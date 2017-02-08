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
rtBuffer<float, 2> depthBuffer;

static __inline__ __device__
void GetDirection(float3& direction, unsigned int& seed, unsigned int i)
{
  const float xo = floorf(i / sampleCount) / sampleCount + torch::randf(seed) / sampleCount;
  const float yo = float(i % sampleCount) / sampleCount + torch::randf(seed) / sampleCount;

  const float2 size = make_float2(imageSize);
  const float2 pixel = make_float2(pixelIndex) + make_float2(xo, yo);
  const float2 ratio = (pixel / size) - camera.center;
  direction = ratio.x * camera.u + ratio.y * camera.v + camera.w;
  direction = normalize(direction);
}

TORCH_DEVICE void InitializeRay(optix::Ray& ray)
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

// TORCH_DEVICE float GetY(const float3& L)
// {
//   return 0.212671f * L.x + 0.715160f * L.y + 0.072169f * L.z;
// }

RT_PROGRAM void Capture()
{
  optix::Ray ray;
  torch::RadianceData data;
  unsigned int seed;

  seed = InitializeSeed();
  InitializeRay(ray);
  data.radiance = make_float3(0, 0, 0);
  const unsigned int totalSamples = sampleCount * sampleCount;

  for (unsigned int i = 0; i < totalSamples; ++i)
  {
    data.sample = i;
    InitializeRay(ray);
    GetDirection(ray.direction, seed, i);
    data.bounce.origin = make_float3(0, 0, 0);
    data.bounce.direction = make_float3(0, 0, 0);
    data.bounce.throughput = make_float3(0, 0, 0);
    data.throughput = make_float3(1.0f / totalSamples);

    for (unsigned int depth = 0; depth < 6; ++depth)
    {
      data.depth = depth;
      data.seed = seed;
      rtTrace(sceneRoot, ray, data);
      InitializeRay(ray);
      seed = data.seed;

      torch::RayBounce& bounce = data.bounce;

      if (length(bounce.direction) < 1E-8)
      {
        break;
      }

      if (depth > 3)
      {
        // const float continueProb = fminf(0.5f, GetY(bounce.throughput));
        const float continueProb = 0.5f;
        // const float continueProb = fmaxf(0.05f, fminf(0.5f, GetY(bounce.throughput)));
        // const float continueProb = fminf(0.5f, fminf(bounce.throughput.z, fminf(bounce.throughput.x, bounce.throughput.y)));

        if (torch::randf(seed) > continueProb)
        {
          break;
        }

        bounce.throughput /= continueProb;
      }

      ray.origin = data.bounce.origin;
      ray.direction = data.bounce.direction;
      data.throughput = data.bounce.throughput;

      data.bounce.origin = make_float3(0, 0, 0);
      data.bounce.direction = make_float3(0, 0, 0);
      data.bounce.throughput = make_float3(0, 0, 0);
    }
  }

  buffer[pixelIndex] = data.radiance;
}

RT_PROGRAM void CaptureDepth()
{
  optix::Ray ray;
  torch::DepthData data;
  unsigned int seed;

  seed = InitializeSeed();
  depthBuffer[pixelIndex] = 0;
  const unsigned int totalSamples = sampleCount * sampleCount;

  for (unsigned int i = 0; i < totalSamples; ++i)
  {
    InitializeRay(ray);
    ray.ray_type = torch::RAY_TYPE_DEPTH;
    data.depth = RT_DEFAULT_MAX;
    GetDirection(ray.direction, seed, i);
    rtTrace(sceneRoot, ray, data);
    depthBuffer[pixelIndex] = fmaxf(depthBuffer[pixelIndex], data.depth);
  }

  buffer[pixelIndex] = make_float3(depthBuffer[pixelIndex]);
}

RT_PROGRAM void CaptureMask()
{
  const int pad = 3;
  float depth = depthBuffer[pixelIndex];

  for (int i = pixelIndex.x - pad; i < pixelIndex.x + pad; ++i)
  {
    for (int j = pixelIndex.y - pad; j < pixelIndex.y + pad; ++j)
    {
      if (i < 0 || i >= imageSize.x || j < 0 || j > imageSize.y ||
          (i == 0 && j == 0))
      {
        continue;
      }

      const float otherDepth = depthBuffer[make_uint2(i, j)];

      if (fabs(depth - otherDepth) > 0.025)
      {
        depth = RT_DEFAULT_MAX;
        break;
      }
    }

    if (depth == RT_DEFAULT_MAX)
    {
      break;
    }
  }

  depth = (depth == RT_DEFAULT_MAX) ? 0 : 1;
  buffer[pixelIndex] = make_float3(depth);
}
