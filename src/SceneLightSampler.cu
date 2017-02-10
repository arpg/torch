#include <optix.h>
#include <torch/device/Light.h>
#include <torch/device/Random.h>

typedef rtCallableProgramX<unsigned int(float, float&)> TypeFunction;
typedef rtCallableProgramX<void(torch::LightSample&)> SampleFunction;

rtDeclareVariable(TypeFunction,   GetLightType, , );
rtDeclareVariable(SampleFunction, SampleAreaLights, , );
rtDeclareVariable(SampleFunction, SampleDirectionalLights, , );
rtDeclareVariable(SampleFunction, SampleEnvironmentLights, , );
rtDeclareVariable(SampleFunction, SamplePointLights, , );

RT_CALLABLE_PROGRAM void Sample(torch::LightSample& sample)
{
  float pdf;
  const float rand = torch::randf(sample.seed);
  const unsigned int type = GetLightType(rand, pdf);

  if (isnan(pdf))
  {
    rtPrintf("Warning: no lights have been added to scene\n");
    return;
  }

  switch (type)
  {
    case torch::LIGHT_TYPE_AREA:
      SampleAreaLights(sample);
      break;

    case torch::LIGHT_TYPE_DIRECTIONAL:
      SampleDirectionalLights(sample);
      break;

    case torch::LIGHT_TYPE_ENVIRONMENT:
      SampleEnvironmentLights(sample);
      break;

    case torch::LIGHT_TYPE_POINT:
      SamplePointLights(sample);
      break;
  }

  sample.pdf *= pdf;
}