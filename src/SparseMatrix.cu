#include <torch/device/Core.h>

rtBuffer<float3, 1> values;
rtBuffer<unsigned int, 1> rowOffsets;
rtBuffer<unsigned int, 1> colIndices;

RT_CALLABLE_PROGRAM void Add(uint row, uint col, float3 value)
{
  unsigned int begin = rowOffsets[row];
  unsigned int end = rowOffsets[row + 1];

  if (begin < end)
  {
    unsigned int index = begin;

    while (begin < end)
    {
      index = (begin + end) / 2;

      if (col == colIndices[index])
      {
        values[index] += value;
        return;
      }

      (col < colIndices[index]) ? end = index : begin = index + 1;
    }
  }
}