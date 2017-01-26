#pragma once

#include <optix_math.h>

#define TORCH_BOTH static __device__ __host__ __inline__

namespace torch
{

template <unsigned int N>
TORCH_BOTH unsigned int init_seed(unsigned int a, unsigned int b)
{
  unsigned int c = 0;

  for (unsigned int i = 0; i < N; ++i)
  {
    c += 0x9E3779B9;
    a += ((b << 4) + 0xA341316C) ^ (b + c) ^ ((b >> 5) + 0xC8013EA4);
    b += ((a << 4) + 0xAD90777D) ^ (a + c) ^ ((a >> 5) + 0x7E95761E);
  }

  return a;
}

TORCH_BOTH unsigned int next_seed(unsigned int& seed)
{
  seed = 1664525u * seed + 1013904223u;
  return seed & 0x00FFFFFF;
}

TORCH_BOTH float randf(unsigned int& seed)
{
  return float(next_seed(seed)) / 0x01000000;
}

TORCH_BOTH float2 randf2(unsigned int& seed)
{
  return make_float2(randf(seed), randf(seed));
}

TORCH_BOTH float3 randf3(unsigned int& seed)
{
  return make_float3(randf(seed), randf(seed), randf(seed));
}

TORCH_BOTH float rand(unsigned int& seed, unsigned int max)
{
  return (unsigned int)((max + 1) * randf(seed));
}

} // namespace torch