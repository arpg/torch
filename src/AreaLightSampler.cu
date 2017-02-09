#include <optix.h>
#include <optix_math.h>
#include <torch/device/Light.h>
#include <torch/device/Geometry.h>
#include <torch/device/Random.h>
#include <torch/device/Visibility.h>

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
  geomSample.type = light.geomType;
  geomSample.id = light.geomId;
  geomSample.origin = sample.origin;
  geomSample.tmin = sample.tmin;
  geomSample.seed = sample.seed;
  SampleGeometry(geomSample);

  sample.radiance = light.radiance;
  sample.direction = geomSample.direction;
  sample.tmax = geomSample.tmax;
  sample.pdf *= geomSample.pdf;
  sample.seed = geomSample.seed;

  if (!torch::IsVisible(sample)) sample.radiance = make_float3(0, 0, 0);
}