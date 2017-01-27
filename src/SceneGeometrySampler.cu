#include <optix.h>
#include <torch/GeometryData.h>
#include <torch/Random.h>

typedef rtCallableProgramX<unsigned int(float, float&)> TypeFunction;
typedef rtCallableProgramX<void(torch::GeometrySample&)> SampleFunction;

rtDeclareVariable(TypeFunction,   GetGeometryType, , );
rtDeclareVariable(SampleFunction, SampleSpheres, , );

RT_CALLABLE_PROGRAM void Sample(torch::GeometrySample& sample)
{
  switch (torch::GetGeometryType(sample.id))
  {
    case torch::GEOM_TYPE_SPHERE:
      SampleSpheres(sample);
      break;
  }
}