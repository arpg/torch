#include <optix.h>
#include <torch/device/Light.h>
#include <torch/device/Random.h>
#include <torch/device/Ray.h>
#include <torch/device/Sampling.h>

rtDeclareVariable(rtObject, sceneRoot, , );
rtDeclareVariable(float, sceneEpsilon, , );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(torch::RadianceData, rayData, rtPayload, );
rtDeclareVariable(float, hitDist, rtIntersectionDistance, );
rtDeclareVariable(float3, geometricNormal, attribute geometricNormal, );
rtDeclareVariable(float3, shadingNormal, attribute shadingNormal, );
rtDeclareVariable(uint3, triFace, attribute triFace, );
rtDeclareVariable(float2, triScales, attribute triScales, );
rtBuffer<float3, 1> albedos;

rtDeclareVariable(optix::Ray, sray, rtCurrentRay, );
rtDeclareVariable(torch::ShadowData, srayData, rtPayload, );

typedef rtCallableProgramX<void(torch::LightSample&)> SampleLightFunction;
rtDeclareVariable(SampleLightFunction, SampleLights, , );

typedef rtCallableProgramId<void(uint, uint, float)> JacobianAddFunction;
rtBuffer<JacobianAddFunction, 1> AddToAlbedoJacobian;

TORCH_DEVICE float3 GetAlbedo()
{
  const uint x = (triFace.x >= albedos.size()) ? albedos.size() - 1 : triFace.x;
  const uint y = (triFace.y >= albedos.size()) ? albedos.size() - 1 : triFace.y;
  const uint z = (triFace.z >= albedos.size()) ? albedos.size() - 1 : triFace.z;
  const float b = fminf(1, triScales.x);
  const float c = fminf(1 - b, triScales.y);
  const float a = 1 - b - c;
  return a * albedos[x] + b * albedos[y] + c * albedos[z];
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

RT_PROGRAM void ClosestHit()
{
  torch::LightSample sample;
  sample.origin = ray.origin + hitDist * ray.direction;
  sample.tmin = sceneEpsilon;
  sample.seed = rayData.seed;
  sample.normal = geometricNormal;
  const unsigned int lightSamples = 16;

  for (unsigned int i = 0; i < lightSamples; ++i)
  {
    SampleLights(sample);
    rayData.seed = sample.seed;

    if (sample.radiance.x > 0 || sample.radiance.y > 0 || sample.radiance.z > 0)
    {
      const float3 brdf = GetAlbedo() / M_PIf;
      const float theta = dot(shadingNormal, sample.direction);
      rayData.radiance += (rayData.throughput * brdf * sample.radiance * theta / sample.pdf) / lightSamples;
    }
  }

  torch::BrdfSample brdfSample;
  brdfSample.normal = shadingNormal;
  brdfSample.seed = rayData.seed;
  SampleBrdf(brdfSample);

  float theta = dot(shadingNormal, brdfSample.direction);
  rayData.bounce.origin = sample.origin;
  rayData.bounce.direction = brdfSample.direction;
  rayData.bounce.throughput = theta * rayData.throughput * brdfSample.throughput / brdfSample.pdf;
}

RT_PROGRAM void AnyHit()
{
  srayData.occluded = true;
  rtTerminateRay();
}
