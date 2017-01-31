#include <optix.h>
#include <optix_math.h>
#include <torch/device/Core.h>

rtBuffer<float, 1> rowCdf;
rtBuffer<float, 1> colCdfs;
rtBuffer<uint, 1> offsets;

TORCH_DEVICE uint GetIndex(float sample, float* cdf, uint begin, uint end,
    float& pdf)
{
  uint index = begin;
  const uint offset = begin;

  while (begin < end)
  {
    index = begin + (end - begin) / 2;
    (sample < cdf[index]) ? end = index : begin = ++index;
  }

  pdf = (index - offset == 0) ? cdf[index] : cdf[index] - cdf[index - 1];
  return index - offset;
}

RT_CALLABLE_PROGRAM uint2 Sample(const float2& sample, float& pdf)
{
  uint begin, end;
  float colPdf;
  uint2 index;

  begin = 0;
  end = rowCdf.size() - 1;
  index.x = GetIndex(sample.x, &rowCdf[0], begin, end, pdf);

  begin = offsets[index.x];
  end = offsets[index.x + 1] - 1;
  index.y = GetIndex(sample.y, &colCdfs[0], begin, end, colPdf);

  pdf *= colPdf;
  return index;
}