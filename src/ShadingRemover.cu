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
rtDeclareVariable(rtObject, dummyRoot, , );
rtBuffer<float3> vertices;
rtBuffer<float3> normals;
rtBuffer<float3> albedos;

TORCH_DEVICE void UpdateAlbedo(const float3& shading)
{
  // TODO: handle divide-by-zero better

  float3 albedo = albedos[launchIndex];

  if (shading.x > 1E-4) albedo.x /= shading.x; else albedo.x = 1.0f;
  if (shading.y > 1E-4) albedo.y /= shading.y; else albedo.y = 1.0f;
  if (shading.z > 1E-4) albedo.z /= shading.z; else albedo.z = 1.0f;

  albedos[launchIndex] = albedo;
}

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
  return torch::init_seed<16>(launchIndex, 7919);
}

RT_PROGRAM void Remove()
{
  optix::Ray ray;
  InitializeRay(ray);
  torch::RadianceData data;
  data.seed = InitializeSeed();
  rtTrace(dummyRoot, ray, data);
}

RT_PROGRAM void ClosestHit()
{
  float3 shading = make_float3(0, 0, 0);
  const float throughput = 1.0f / (sampleCount * M_PIf);

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
    shading += sample.radiance;
  }

  UpdateAlbedo(shading);
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