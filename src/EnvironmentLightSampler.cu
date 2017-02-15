#include <torch/device/Light.h>
#include <torch/device/Random.h>
#include <torch/device/Visibility.h>
#include <torch/device/Camera.h>

typedef rtCallableProgramX<uint(float, float&)> Distribution1D;
typedef rtCallableProgramId<uint2(const float2&, float&)> Distribution2D;

rtDeclareVariable(Distribution1D, GetLightIndex, , );
rtBuffer<Distribution2D> SampleLight;
rtBuffer<optix::Matrix3x3> rotations;
rtBuffer<rtBufferId<uint, 1>, 1> offsets;
rtBuffer<rtBufferId<float3, 1>, 1> radiance;

rtDeclareVariable(uint, computeLightDerivs, , );
rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtBuffer<torch::CameraData> cameras;
rtBuffer<torch::PixelSample> pixelSamples;
rtBuffer<rtBufferId<float3, 2>, 1> lightDerivatives;

TORCH_DEVICE uint GetRowCount(uint light)
{
  return offsets[light].size() - 1;
}

TORCH_DEVICE uint GetColumnCount(uint light, uint row)
{
  return offsets[light][row + 1] - offsets[light][row];
}

TORCH_DEVICE float3 GetRadiance(uint light, uint row, uint col)
{
  const uint offset = offsets[light][row];
  return radiance[light][offset + col];
}

TORCH_DEVICE void GetDirection(uint light, uint row, uint col, float3& dir)
{
  const uint rowCount = GetRowCount(light);
  const uint colCount = GetColumnCount(light, row);
  const float rowRadians = row * M_PIf / (rowCount - 1);
  const float colRadians = col * 2 * M_PIf / colCount;
  const float rowRadius = sinf(rowRadians);
  dir.x = rowRadius * sinf(colRadians);
  dir.z = rowRadius * cosf(colRadians);
  dir.y = cosf(rowRadians);
  dir = -normalize(dir);
}

RT_CALLABLE_PROGRAM void Sample(torch::LightSample& sample)
{
  const float rand = torch::randf(sample.seed);
  const uint light = GetLightIndex(rand, sample.pdf);

  float dirPdf;
  const float2 uv = torch::randf2(sample.seed);
  const uint2 index = SampleLight[light](uv, dirPdf);
  GetDirection(light, index.x, index.y, sample.direction);

  sample.direction = rotations[light] * sample.direction;
  sample.direction = normalize(sample.direction);
  sample.radiance = GetRadiance(light, index.x, index.y);

  sample.tmax = RT_DEFAULT_MAX;
  sample.pdf *= dirPdf;

  const bool visible = torch::IsVisible(sample);
  if (!visible) sample.radiance = make_float3(0, 0, 0);

  if (computeLightDerivs && visible) // TODO: check if light has non-empty derivs
  {
    const float theta = dot(sample.direction, sample.snormal);
    const uint paramIndex = offsets[light][index.x] + index.y;
    const uint2 derivIndex = make_uint2(paramIndex, launchIndex.x);
    lightDerivatives[light][derivIndex] -= theta * sample.throughput / sample.pdf;
  }
}