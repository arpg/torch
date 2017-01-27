#include <optix.h>
#include <torch/GeometryData.h>

rtBuffer<torch::SphereData, 1> spheres;

RT_CALLABLE_PROGRAM void Sample(torch::GeometrySample& sample)
{
}