#include <optix.h>
#include <torch/LightData.h>
#include <torch/Random.h>

typedef rtCallableProgramX<unsigned int(float, float&)> TypeFunction;
typedef rtCallableProgramX<void(torch::LightSample&)> SampleFunction;

rtDeclareVariable(TypeFunction,   GetLightType, , );
rtDeclareVariable(SampleFunction, SamplePointLights, , );

RT_CALLABLE_PROGRAM void Sample(torch::LightSample& sample)
{
  float pdf;
  const float rand = torch::randf(sample.seed);
  const unsigned int type = GetLightType(rand, pdf);

  switch (type)
  {
    case torch::LIGHT_TYPE_POINT:
      SamplePointLights(sample);
      break;
  }

  sample.pdf *= pdf;
}