#include <optix.h>
#include <torch/device/Geometry.h>
#include <torch/device/Random.h>

typedef rtCallableProgramX<unsigned int(float, float&)> TypeFunction;
typedef rtCallableProgramX<void(torch::GeometrySample&)> SampleFunction;

rtDeclareVariable(TypeFunction,   GetGeometryType, , );
rtDeclareVariable(SampleFunction, SampleGeometryGroups, , );
rtDeclareVariable(SampleFunction, SampleMeshes, , );
rtDeclareVariable(SampleFunction, SampleSpheres, , );

RT_CALLABLE_PROGRAM void Sample(torch::GeometrySample& sample)
{
  switch (sample.type)
  {
    case torch::GEOM_TYPE_GROUP:
      SampleGeometryGroups(sample);
      break;

    case torch::GEOM_TYPE_MESH:
      SampleMeshes(sample);
      break;

    case torch::GEOM_TYPE_SPHERE:
      SampleSpheres(sample);
      break;
  }
}