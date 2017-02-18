#include <torch/device/Light.h>
#include <torch/device/Random.h>
#include <torch/device/Ray.h>
#include <torch/device/Sampling.h>

typedef rtCallableProgramX<void(torch::LightSample&)> SampleLightFunction;
rtDeclareVariable(SampleLightFunction, SampleLights, , );
rtDeclareVariable(rtObject, sceneRoot, , );
rtDeclareVariable(float, sceneEpsilon, , );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(torch::RadianceData, rayData, rtPayload, );
rtDeclareVariable(unsigned int, launchIndex, rtLaunchIndex, );
rtDeclareVariable(unsigned int, sampleCount, , );
rtDeclareVariable(rtObject, bakeRoot, , );
rtBuffer<float3> vertices;
rtBuffer<float3> normals;
rtBuffer<float3> albedos;
rtBuffer<float3> scratch;

TORCH_DEVICE float3 GetAlbedo()
{
  return albedos[launchIndex];
}

TORCH_DEVICE float3 GetNormal()
{
  return normalize(normals[launchIndex]);
}

TORCH_DEVICE optix::Matrix3x3 NormalToRotation(const float3& n)
{
 float3 u = make_float3(1, 0, 0);
 float3 v = make_float3(0, 0, 1);
 if (dot(u, n) < dot(v, n)) u = v;
 v = normalize(cross(n, u));
 u = normalize(cross(n, v));

 optix::Matrix3x3 R;
 R.setCol(0, u);
 R.setCol(1, v);
 R.setCol(2, n);
 return R;
}

TORCH_DEVICE void SampleBrdf(torch::BrdfSample& sample)
{
  float3 direction;
  torch::SampleHemisphereCosine(torch::randf2(sample.seed), direction);
  sample.pdf = direction.z / M_PIf;
  sample.direction = NormalToRotation(sample.normal) * direction;
  sample.throughput = GetAlbedo() / M_PIf;
}

TORCH_DEVICE void InitializeRay(optix::Ray& ray)
{
  ray.origin = make_float3(0, 0, 0);
  ray.direction = make_float3(0, 0, 1);
  ray.tmin = 0.0f;
  ray.tmax = RT_DEFAULT_MAX;
  ray.ray_type = torch::RAY_TYPE_RADIANCE;
}

TORCH_DEVICE unsigned int InitializeSeed()
{
  return torch::init_seed<16>(launchIndex, 7919);
}

RT_PROGRAM void Bake()
{
  optix::Ray ray;
  torch::RadianceData data;
  unsigned int seed;

  seed = InitializeSeed();
  data.radiance = make_float3(0, 0, 0);
  const unsigned int totalSamples = sampleCount * sampleCount;

  for (unsigned int i = 0; i < totalSamples; ++i)
  {
    data.sample = i;
    InitializeRay(ray);
    data.bounce.origin = make_float3(0, 0, 0);
    data.bounce.direction = make_float3(0, 0, 0);
    data.bounce.throughput = make_float3(0, 0, 0);
    data.throughput = make_float3(1.0f / totalSamples);

    for (unsigned int depth = 0; depth < 6; ++depth)
    {
      data.depth = depth;
      data.seed = seed;

      (depth == 0) ?
          rtTrace(bakeRoot, ray, data) :
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

  scratch[launchIndex] = data.radiance;
}

RT_PROGRAM void Copy()
{
  albedos[launchIndex] = scratch[launchIndex];
}

RT_PROGRAM void ClosestHit()
{
  torch::LightSample sample;
  sample.origin = vertices[launchIndex];
  sample.tmin = sceneEpsilon;
  sample.seed = rayData.seed;
  sample.normal = GetNormal();
  sample.snormal = GetNormal();
  const unsigned int lightSamples = 16;

  const float3 albedo = GetAlbedo();
  sample.throughput = (albedo * rayData.throughput) / (lightSamples * M_PIf);

  for (unsigned int i = 0; i < lightSamples; ++i)
  {
    SampleLights(sample);
    rayData.seed = sample.seed;

    if (length(sample.radiance) > 0)
    {
      const float3 albedo = GetAlbedo();
      const float theta = dot(GetNormal(), sample.direction);
      const float3 throughput = (rayData.throughput * sample.radiance * theta / sample.pdf) / (lightSamples * M_PIf);
      rayData.radiance += albedo * throughput;
    }
  }

  torch::BrdfSample brdfSample;
  brdfSample.normal = GetNormal();
  brdfSample.seed = rayData.seed;
  SampleBrdf(brdfSample);

  if (brdfSample.pdf > 1E-8)
  {
    float theta = dot(GetNormal(), brdfSample.direction);
    rayData.bounce.origin = sample.origin;
    rayData.bounce.direction = brdfSample.direction;
    rayData.bounce.throughput = theta * rayData.throughput * brdfSample.throughput / brdfSample.pdf;
  }
}

RT_PROGRAM void Intersect(unsigned int index)
{
  if (rtPotentialIntersection(1.0f))
  {
    rtReportIntersection(0);
  }
}

RT_PROGRAM void GetBounds(unsigned int index, float bounds[6])
{
  bounds[0] = -1; bounds[1] = -1; bounds[2] = -1;
  bounds[3] = +1; bounds[4] = +1; bounds[5] = +1;
}