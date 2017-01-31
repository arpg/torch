#include <torch/device/Light.h>
#include <torch/device/Random.h>

typedef rtCallableProgramX<uint(float, float&)> Distribution1D;
typedef rtCallableProgramId<void(const float2&, uint&, uint&, float&)> Distribution2D;

rtDeclareVariable(Distribution1D, GetLightIndex, , );
rtBuffer<Distribution2D> SampleLight;
rtBuffer<optix::Matrix3x3> rotations;
rtBuffer<uint> lightOffsets;
rtBuffer<uint> rowOffsets;
rtBuffer<float3> radiance;

TORCH_DEVICE uint GetRowCount(uint light)
{
  return lightOffsets[light + 1] - lightOffsets[light];
}

TORCH_DEVICE uint GetColumnCount(uint light, uint row)
{
  const uint lightOffset = lightOffsets[light];
  const uint rowOffset = rowOffsets[lightOffset] + row;
  return rowOffsets[rowOffset + 1] - rowOffsets[rowOffset];
}

TORCH_DEVICE float3 GetRadiance(uint light, uint row, uint col)
{
  const uint lightOffset = lightOffsets[light];
  const uint rowOffset = rowOffsets[lightOffset] + row;
  return radiance[rowOffset] + col;
}

TORCH_DEVICE void GetDirection(uint light, uint row, uint col, float3& dir)
{
  const uint rowCount = GetRowCount(light);
  const uint colCount = GetColumnCount(light, row);
  const float rowRadians = M_PIf / (rowCount - 1);
  const float colRadians = 2 * M_PIf / (col / colCount);
  const float rowRadius = sinf(rowRadians);
  dir.x = rowRadius * cosf(colRadians);
  dir.y = rowRadius * sinf(colRadians);
  dir.z = cosf(rowRadians);
  dir = normalize(dir);
}

RT_CALLABLE_PROGRAM void Sample(torch::LightSample& sample)
{
  const float rand = torch::randf(sample.seed);
  const uint light = GetLightIndex(rand, sample.pdf);

  float dirPdf;
  uint row, col;
  const float2 uv = torch::randf2(sample.seed);
  // SampleLight[light](uv, row, col, dirPdf);
  GetDirection(light, row, col, sample.direction);

  sample.direction = rotations[light] * sample.direction;
  sample.direction = normalize(sample.direction);
  sample.radiance = GetRadiance(light, row, col);
  sample.tmax = RT_DEFAULT_MAX;
  sample.pdf *= dirPdf;

  // sample.direction = normalize(make_float3(0, -1, -1));
  // sample.radiance = make_float3(2.5, 2.5, 2.5);
  // sample.tmax = RT_DEFAULT_MAX;
  // sample.pdf = 1;
}