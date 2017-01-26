#include <optix.h>
#include <torch/LightData.h>

typedef rtCallableProgramX<unsigned int(float, float&)> SampleFunction;
rtDeclareVariable(SampleFunction, GetLightIndex, , );
rtBuffer<torch::PointLightData, 1> lights;

RT_CALLABLE_PROGRAM void Sample()
{
}