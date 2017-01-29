#include <optix.h>
#include <optixu/optixu_matrix.h>
#include <torch/device/Light.h>
#include <torch/device/Random.h>

typedef rtCallableProgramX<unsigned int(float, float&)> SampleFunctionX;
typedef rtCallableProgramId<unsigned int(float, float&)> SampleFunctionId;
rtDeclareVariable(SampleFunctionX, GetLightIndex, , );
rtBuffer<SampleFunctionId> SampleLight;
rtBuffer<SampleFunctionId> GetRow;
rtBuffer<SampleFunctionId> GetCol;

rtBuffer<float3, 1> radiance;
rtBuffer<unsigned int, 1> rowOffsets;
rtBuffer<unsigned int, 1> colOffsets;
rtBuffer<optix::Matrix3x3, 1> rotations;

RT_CALLABLE_PROGRAM void Sample(torch::LightSample& sample)
{
  // sample light
  float rand = torch::randf(sample.seed);
  const unsigned int lightIndex = GetLightIndex(rand, sample.pdf);
  float pdf;

  // sample row
  rand = torch::randf(sample.seed);
  const unsigned int row = GetRow[lightIndex](rand, pdf);
  const unsigned int rowOffset = rowOffsets[lightIndex];
  const unsigned int rowIndex = rowOffset + row;
  sample.pdf *= pdf;

  // sample column
  rand = torch::randf(sample.seed);
  const unsigned int col = GetCol[rowIndex](rand, pdf);
  const unsigned int colOffset = colOffsets[rowIndex];
  const unsigned int colIndex = colOffset + col;
  sample.pdf *= pdf;

  // compute row radius
  const unsigned int rowCount = rowOffsets[rowIndex + 1] - rowOffset;
  const unsigned int colCount = colOffsets[rowIndex + 1] - colOffset;
  const float rowRadians = row * M_PIf / (rowCount - 1);
  const float colRadians = col * 2 * M_PIf / (colCount - 1);
  const float radius = sinf(rowRadians);

  // compute local direction
  sample.direction.y = cosf(rowRadians);
  sample.direction.x = radius * cosf(colRadians);
  sample.direction.z = radius * sinf(colRadians);
  sample.direction = normalize(sample.direction);

  // compute final light sample
  sample.direction = rotations[lightIndex] * sample.direction;
  sample.direction = normalize(sample.direction);
  sample.radiance = radiance[colIndex];
  sample.tmax = RT_DEFAULT_MAX;
}