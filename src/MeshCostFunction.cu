#include <torch/device/Light.h>
#include <torch/device/Random.h>
#include <torch/device/Ray.h>

typedef rtCallableProgramX<void(torch::LightSample&)> SampleLightFunction;
rtDeclareVariable(SampleLightFunction, SampleLights, , );
rtDeclareVariable(float, sceneEpsilon, , );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(torch::RadianceData, rayData, rtPayload, );
rtDeclareVariable(unsigned int, launchIndex, rtLaunchIndex, );
rtDeclareVariable(unsigned int, sampleCount, , );
rtDeclareVariable(unsigned int, iteration, , );
rtDeclareVariable(rtObject, dummyRoot, , );
rtBuffer<float3> vertices;
rtBuffer<float3> normals;

TORCH_DEVICE float3 GetNormal()
{
  return normalize(normals[launchIndex]);
}

TORCH_DEVICE void InitializeRay(optix::Ray& ray)
{
  ray.origin = make_float3(0, 0, 0);
  ray.direction = make_float3(0, 0, 1);
  ray.ray_type = torch::RAY_TYPE_RADIANCE;
  ray.tmax = RT_DEFAULT_MAX;
  ray.tmin = 0.0f;
}

TORCH_DEVICE unsigned int InitializeSeed()
{
  return torch::init_seed<16>(launchIndex, iteration);
}

RT_PROGRAM void Capture()
{
  optix::Ray ray;
  InitializeRay(ray);
  torch::RadianceData data;
  data.seed = InitializeSeed();
  rtTrace(dummyRoot, ray, data);
}

RT_PROGRAM void ClosestHit()
{
  // const float throughput = 1.0f / (sampleCount * M_PIf);
  const float throughput = 1.0f / M_PIf;

  torch::LightSample sample;
  sample.origin = vertices[launchIndex];
  sample.tmin = sceneEpsilon;
  sample.seed = rayData.seed;
  sample.normal = GetNormal();
  sample.snormal = sample.normal;
  sample.throughput = make_float3(throughput);

  for (unsigned int i = 0; i < sampleCount; ++i)
  {
    SampleLights(sample);
  }
}

RT_PROGRAM void Intersect(unsigned int index)
{
  if (rtPotentialIntersection(1.0f)) rtReportIntersection(0);
}

RT_PROGRAM void GetBounds(unsigned int index, float bounds[6])
{
  bounds[0] = -1; bounds[1] = -1; bounds[2] = -1;
  bounds[3] = +1; bounds[4] = +1; bounds[5] = +1;
}