#include <optix.h>
#include <torch/device/Geometry.h>
#include <torch/device/Random.h>

rtBuffer<torch::SphereData, 1> spheres;

RT_CALLABLE_PROGRAM void Sample(torch::GeometrySample& sample)
{
  // const torch::SphereData& sphere = spheres[sample.id];
  // const float3 center = make_float3(sphere.T_lw * make_float4(0, 0, 0, 1));

  const float3 offset = 3 * (0.5 - torch::randf3(sample.seed));
  const float3 position = make_float3(4, -1, -4) + offset; // TODO: implement
  const float3 difference = position - sample.origin;
  sample.direction = normalize(difference);
  sample.tmax = length(difference);
  sample.pdf = 1.0 / (4 * M_PIf * 0.5 * 0.5); // TODO: implement
}