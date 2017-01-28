#include <optix.h>
#include <optix_math.h>
#include <torch/device/Light.h>
#include <torch/device/Geometry.h>
#include <torch/device/Random.h>

typedef rtCallableProgramX<unsigned int(float, float&)> SampleLightFunction;
typedef rtCallableProgramX<void(torch::GeometrySample&)> SampleGeomFunction;
rtDeclareVariable(SampleLightFunction, GetLightIndex, , );
rtDeclareVariable(SampleGeomFunction, SampleGeometry, , );
rtBuffer<torch::AreaLightData, 1> lights;

RT_CALLABLE_PROGRAM void Sample(torch::LightSample& sample)
{
  const float rand = torch::randf(sample.seed);
  const unsigned int index = GetLightIndex(rand, sample.pdf);
  const torch::AreaLightData& light = lights[index];

  torch::GeometrySample geomSample;
  geomSample.id = light.geometry;
  geomSample.origin = sample.origin;
  geomSample.tmin = sample.tmin;
  SampleGeometry(geomSample);

  const float3 difference = geomSample.position - sample.origin;
  sample.radiance = light.radiance;
  sample.direction = normalize(difference);
  sample.tmax = length(difference);
  sample.pdf *= geomSample.pdf;
}