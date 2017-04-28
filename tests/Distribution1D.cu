#include <optix.h>
#include <torch/device/Random.h>

typedef rtCallableProgramX<unsigned int(float, float&)> SampleFunction;

rtDeclareVariable(unsigned int, launchIndex, rtLaunchIndex, );
rtDeclareVariable(SampleFunction, Sample1D, , );
rtBuffer<unsigned int> counts;
rtBuffer<float> pdfs;

RT_PROGRAM void Sample()
{
  float pdf;
  unsigned int seed = torch::init_seed<16>(launchIndex, 0);
  const unsigned int index = Sample1D(torch::randf(seed), pdf);
  atomicAdd(&counts[index], 1);
  pdfs[index] = pdf;
}
