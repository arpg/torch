#include <optix.h>
#include <torch/device/Light.h>
#include <torch/device/Random.h>
#include <torch/device/Ray.h>
#include <torch/device/Sampling.h>

rtDeclareVariable(rtObject, sceneRoot, , );
rtDeclareVariable(float, sceneEpsilon, , );

rtDeclareVariable(float3, albedo, , );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(torch::RadianceData, rayData, rtPayload, );
rtDeclareVariable(float, hitDist, rtIntersectionDistance, );
rtDeclareVariable(float3, geometricNormal, attribute geometricNormal, );
rtDeclareVariable(float3, shadingNormal, attribute shadingNormal, );

rtDeclareVariable(optix::Ray, sray, rtCurrentRay, );
rtDeclareVariable(torch::ShadowData, srayData, rtPayload, );

typedef rtCallableProgramX<void(torch::LightSample&)> SampleLightFunction;
rtDeclareVariable(SampleLightFunction, SampleLights, , );

 TORCH_DEVICE optix::Matrix3x3 NormalToRotation(const float3& n)
 {
   float3 u = make_float3(1, 0, 0);
   float3 v = make_float3(0, 0, 1);
   if (dot(u, n) < dot(v, n)) u = v;
   v = normalize(cross(n, u));
   u = normalize(cross(n, v));

   optix::Matrix3x3 R;
   R.setCol(0, u);
   R.setCol(1, n);
   R.setCol(2, v);
   return R;
 }

TORCH_DEVICE void SampleBrdf(torch::BrdfSample& sample)
{
  float3 direction;
  torch::SampleHemisphereCosine(torch::randf2(sample.seed), direction);
  sample.pdf = direction.z / M_PIf;
  sample.direction = NormalToRotation(sample.normal) * direction;
  sample.throughput = 0.5 * albedo / M_PIf;
}

RT_PROGRAM void ClosestHit()
{
  torch::LightSample sample;
  sample.origin = ray.origin + hitDist * ray.direction;
  sample.tmin = sceneEpsilon;
  sample.seed = rayData.seed;

  SampleLights(sample);
  rayData.seed = sample.seed;

  float theta = dot(geometricNormal, sample.direction);

  if (theta > 0.0f)
  {
    optix::Ray shadowRay;
    shadowRay.origin = sample.origin;
    shadowRay.direction = sample.direction;
    shadowRay.ray_type = torch::RAY_TYPE_SHADOW;
    shadowRay.tmin = sample.tmin;
    shadowRay.tmax = sample.tmax;

    torch::ShadowData shadowData;
    shadowData.occluded = false;

    rtTrace(sceneRoot, shadowRay, shadowData);

    // if (!shadowData.occluded && rayData.depth > 0)
    // {
      float3 brdf = albedo / M_PIf;
      theta = dot(shadingNormal, sample.direction);
      rayData.radiance += rayData.throughput * brdf * sample.radiance * theta / sample.pdf;
    // }
  }

  torch::BrdfSample brdfSample;
  brdfSample.normal = shadingNormal;
  brdfSample.seed = rayData.seed;
  SampleBrdf(brdfSample);

  theta = dot(shadingNormal, brdfSample.direction);
  rayData.bounce.origin = sample.origin;
  rayData.bounce.direction = brdfSample.direction;
  rayData.bounce.throughput = make_float3(0.5); // theta * rayData.throughput * brdfSample.throughput / brdfSample.pdf;
}

RT_PROGRAM void AnyHit()
{
  srayData.occluded = true;
  rtTerminateRay();
}