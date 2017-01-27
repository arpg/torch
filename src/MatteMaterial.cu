#include <optix.h>
#include <torch/LightData.h>
#include <torch/Ray.h>

rtDeclareVariable(rtObject, sceneRoot, , );
rtDeclareVariable(float, sceneEpsilon, , );

rtDeclareVariable(float3, albedo, , );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(torch::RadianceData, rayData, rtPayload, );
rtDeclareVariable(float, hitDist, rtIntersectionDistance, );
rtDeclareVariable(float3, geometricNormal, attribute geometricNormal, );
rtDeclareVariable(float3, shadingNormal, attribute shadingNormal, );

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

  if (theta > 0)
  {
    // TODO: trace shadow ray
    float3 brdf = albedo / M_PIf;
    theta = dot(shadingNormal, sample.direction);
    rayData.radiance += rayData.throughput * brdf * sample.radiance * theta / sample.pdf;
  }
}