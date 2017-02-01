#include <optix.h>
#include <torch/device/Geometry.h>
#include <torch/device/Random.h>

rtBuffer<torch::GeometryGroupData, 1> groups;

RT_CALLABLE_PROGRAM void Sample(torch::GeometrySample& sample)
{

}