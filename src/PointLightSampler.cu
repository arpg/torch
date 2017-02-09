#include <optix.h>
#include <optix_math.h>
#include <torch/device/Light.h>
#include <torch/device/Random.h>
#include <torch/device/Visibility.h>

typedef rtCallableProgramX<unsigned int(float, float&)> SampleFunction;
rtDeclareVariable(SampleFunction, GetLightIndex, , );
rtBuffer<torch::PointLightData, 1> lights;

RT_CALLABLE_PROGRAM void Sample(torch::LightSample& sample)
{
  const float rand = torch::randf(sample.seed);
  const unsigned int index = GetLightIndex(rand, sample.pdf);

  const torch::PointLightData& light = lights[index];
  const float3 difference = light.position - sample.origin;
  const float distanceSquared = dot(difference, difference);

  sample.radiance = light.intensity / distanceSquared;
  sample.direction = normalize(difference);
  sample.tmax = sqrt(distanceSquared);

  if (!torch::IsVisible(sample)) sample.radiance = make_float3(0, 0, 0);
}