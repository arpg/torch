#include <torch/device/Camera.h>
#include <torch/device/Random.h>
#include <torch/device/Ray.h>

rtDeclareVariable(rtObject, sceneRoot, , );
rtDeclareVariable(float, sceneEpsilon, , );
rtDeclareVariable(uint, launchIndex, rtLaunchIndex, );
rtBuffer<torch::CameraData> cameras;
rtBuffer<torch::PixelSample> pixelSamples;
rtBuffer<float3> render;
rtBuffer<float3, 2> lightDerivatives;

static __inline__ __device__
void GetDirection(float3& direction, unsigned int& seed, unsigned int i)
{
  const unsigned int cameraIndex = pixelSamples[launchIndex].camera;
  const torch::CameraData& camera = cameras[cameraIndex];
  const unsigned int sampleCount = camera.samples;
  const uint2& imageSize = camera.imageSize;
  const uint2& pixelIndex = pixelSamples[launchIndex].uv;

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
  const unsigned int cameraIndex = pixelSamples[launchIndex].camera;
  const torch::CameraData camera = cameras[cameraIndex];
  ray.origin = camera.position;
  ray.tmin = sceneEpsilon;
  ray.tmax = RT_DEFAULT_MAX;
  ray.ray_type = torch::RAY_TYPE_RADIANCE;
}

TORCH_DEVICE unsigned int InitializeSeed()
{
  const uint2& pixelIndex = pixelSamples[launchIndex].uv;
  unsigned int a = pixelIndex.x;
  unsigned int b = pixelIndex.y;
  return torch::init_seed<16>(a, b);
}

RT_PROGRAM void Capture()
{
  const unsigned int cameraIndex = pixelSamples[launchIndex].camera;
  const torch::CameraData& camera = cameras[cameraIndex];
  const unsigned int sampleCount = camera.samples;
  const uint2& pixelIndex = pixelSamples[launchIndex].uv;

  optix::Ray ray;
  torch::RadianceData data;
  unsigned int seed;

  seed = InitializeSeed();
  InitializeRay(ray);
  data.radiance = make_float3(0, 0, 0);
  const unsigned int totalSamples = sampleCount * sampleCount;

  for (size_t i = 0; i < lightDerivatives.size().x; ++i)
  {
    const uint2 derivIndex = make_uint2(i, launchIndex);
    lightDerivatives[derivIndex] = make_float3(0, 0, 0);
  }

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
        const float continueProb = 0.5f;

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

  render[launchIndex] = data.radiance;
}