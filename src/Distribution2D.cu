#include <optix.h>
#include <optix_math.h>
#include <torch/device/Core.h>

rtBuffer<float, 1> rowCdf;
rtBuffer<float, 1> colCdfs;
rtBuffer<unsigned int, 1> offsets;

TORCH_DEVICE unsigned int GetIndex(float sample, float* cdf,
    unsigned int begin, unsigned int end, float& pdf)
{
  unsigned int index = begin;

  while (begin < end)
  {
    index = begin + (end - begin) / 2;
    (sample < cdf[index]) ? end = index : begin = ++index;
  }

  pdf = (index - begin == 0) ? cdf[index] : cdf[index] - cdf[index - 1];
  return index - begin;
}

RT_CALLABLE_PROGRAM void Sample(const float2& sample, uint2& index, float& pdf)
{
  unsigned int begin, end;
  float colPdf;

  begin = 0;
  end = rowCdf.size() - 1;
  index.x = GetIndex(sample.x, &rowCdf[0], begin, end, pdf);

  begin = offsets[index.x];
  end = offsets[index.x + 1] - begin;
  index.y = GetIndex(sample.y, &colCdfs[0], begin, end, colPdf);

  pdf *= colPdf;
}