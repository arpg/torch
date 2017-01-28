#include <optix.h>
#include <optix_math.h>
#include <torch/device/LightData.h>
#include <torch/device/Random.h>

typedef rtCallableProgramX<unsigned int(float, float&)> SampleFunction;
rtDeclareVariable(SampleFunction, GetLightIndex, , );
rtBuffer<torch::DistantLightData, 1> lights;

RT_CALLABLE_PROGRAM void Sample(torch::LightSample& sample)
{
  const float rand = torch::randf(sample.seed);
  const unsigned int index = GetLightIndex(rand, sample.pdf);
  const torch::DistantLightData& light = lights[index];
  sample.radiance = light.radiance;
  sample.direction = -light.direction;
  sample.tmax = RT_DEFAULT_MAX;
}