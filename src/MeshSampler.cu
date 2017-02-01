#include <optix.h>
#include <torch/device/Geometry.h>
#include <torch/device/Random.h>

rtBuffer<torch::MeshData, 1> meshes;

RT_CALLABLE_PROGRAM void Sample(torch::GeometrySample& sample)
{

}