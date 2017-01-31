#include <optix.h>
#include <torch/device/Random.h>

typedef rtCallableProgramX<uint2(const float2&, float&)> SampleFunction;

rtDeclareVariable(unsigned int, launchIndex, rtLaunchIndex, );
rtDeclareVariable(SampleFunction, Sample2D, , );
rtBuffer<unsigned int> offsets;
rtBuffer<unsigned int> counts;
rtBuffer<float> pdfs;

TORCH_DEVICE unsigned int GetBufferIndex(const uint2& index)
{
  return offsets[index.x] + index.y;
}

RT_PROGRAM void Sample()
{
  float pdf;
  unsigned int seed = torch::init_seed<16>(launchIndex, 0);
  const uint2 index = Sample2D(torch::randf2(seed), pdf);
  const unsigned int bufferIndex = GetBufferIndex(index);
  atomicAdd(&counts[bufferIndex], 1);
  pdfs[bufferIndex] = pdf;
}