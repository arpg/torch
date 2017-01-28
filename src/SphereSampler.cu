#include <optix.h>
#include <torch/device/GeometryData.h>
#include <torch/device/Random.h>

rtBuffer<torch::SphereData, 1> spheres;

RT_CALLABLE_PROGRAM void Sample(torch::GeometrySample& sample)
{
  const float3 offset = 1.5 * torch::randf3(sample.seed);
  sample.position = make_float3(4, -1, -1) + offset;
  sample.pdf = 1.0 / (4 * M_PIf * 0.5 * 0.5);
}