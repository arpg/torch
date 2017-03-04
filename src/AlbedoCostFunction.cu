#include <cfloat>
#include <torch/device/Camera.h>
#include <torch/device/Random.h>
#include <torch/device/Ray.h>

rtDeclareVariable(rtObject, sceneRoot, , );
rtDeclareVariable(float, sceneEpsilon, , );
rtDeclareVariable(uint, launchIndex, rtLaunchIndex, );
rtBuffer<torch::CameraData> cameras;
rtBuffer<torch::PixelSample> pixelSamples;
rtBuffer<float3> render;

rtBuffer<uint4> boundingBoxes;
rtBuffer<unsigned int> neighborOffsets;
rtBuffer<unsigned int> neighborIndices;
rtBuffer<float3> vertices;

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
      if (isnan(ray.direction.x))
      {
        rtPrintf("%d: NANS!!!\n", depth);
      }

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

TORCH_DEVICE float2 GetImageCoords(uint image, uint vertex)
{
  const torch::CameraData& camera = cameras[image];
  const float3 Xcp = make_float3(camera.Tcw * make_float4(vertices[vertex], 1));
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

  const uint image = launchIndex / vertices.size();
  const uint vertex = launchIndex % vertices.size();
  const torch::CameraData& camera = cameras[image];

  uint4& boundingBox = boundingBoxes[image * vertices.size() + vertex];

  const unsigned int start = neighborOffsets[vertex];
  const unsigned int stop = neighborOffsets[vertex + 1];
  Union(bbox, GetImageCoords(image, vertex));

  for (unsigned int i = start; i < stop; ++i)
  {
    unsigned int neighborVertex = neighborIndices[i];
    Union(bbox, GetImageCoords(image, neighborVertex));
  }

  boundingBox.x = max(0u, uint(floor(bbox.x)));
  boundingBox.y = max(0u, uint(floor(bbox.y)));
  boundingBox.z = min(camera.imageSize.x - 1, uint(ceil(bbox.z)));
  boundingBox.w = min(camera.imageSize.y - 1, uint(ceil(bbox.w)));
}