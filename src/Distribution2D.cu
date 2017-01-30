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

  while (begin < end)
  {
    index = begin + (end - begin) / 2;
    (sample < cdf[index]) ? end = index : begin = ++index;
  }

  pdf = (index - begin == 0) ? cdf[index] : cdf[index] - cdf[index - 1];
  return index - begin;
}

RT_CALLABLE_PROGRAM void Sample(const float2& sample, uint& row, uint& col,
    float& pdf)
{
  uint begin, end;
  float colPdf;

  begin = 0;
  end = rowCdf.size() - 1;
  row = GetIndex(sample.x, &rowCdf[0], begin, end, pdf);

  begin = offsets[row];
  end = offsets[row + 1] - begin;
  col = GetIndex(sample.y, &colCdfs[0], begin, end, colPdf);

  pdf *= colPdf;
}