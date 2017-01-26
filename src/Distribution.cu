#include <optix.h>

rtBuffer<float, 1> cdf;
rtDeclareVariable(unsigned int, count, , );

RT_CALLABLE_PROGRAM unsigned int Sample(float sample, float& pdf)
{
  unsigned int index = 1;
  unsigned int begin = 1;
  unsigned int end = count - 1;

  while (begin < end)
  {
    index = begin + (end - begin) / 2;
    (sample < cdf[index]) ? end = index : begin = ++index;
  }

  pdf = cdf[index] - cdf[index - 1];
  return index - 1;
}