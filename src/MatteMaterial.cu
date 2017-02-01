#include <optix.h>
#include <torch/device/Light.h>
#include <torch/device/Ray.h>

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

    if (!shadowData.occluded)
    {
      float3 brdf = albedo / M_PIf;
      theta = dot(shadingNormal, sample.direction);
      rayData.radiance += rayData.throughput * brdf * sample.radiance * theta / sample.pdf;
    }
  }
}

RT_PROGRAM void AnyHit()
{
  srayData.occluded = true;
  rtTerminateRay();
}