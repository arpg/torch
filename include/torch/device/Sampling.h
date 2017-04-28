#include <torch/device/Core.h>

namespace torch
{

TORCH_HOSTDEVICE void SampleDiskUniform(const float2& sample, float2& p)
{
  float r = 0.0f;
  float theta = 0.0f;
  const float sx = 2 * sample.x - 1;
  const float sy = 2 * sample.y - 1;

  if (sx == 0.0 && sy == 0.0)
  {
    p.x = 0.0;
    p.y = 0.0;
  }
  else if (sx >= -sy)
  {
    if (sx > sy)
    {
      r = sx;
      if (sy > 0.0) theta = sy / r;
      else          theta = 8.0 + sy / r;
    }
    else
    {
      r = sy;
      theta = 2.0 - sx / r;
    }
  }
  else
  {
    if (sx <= sy)
    {
      r = -sx;
      theta = 4.0 - sy / r;
    }
    else
    {
      r = -sy;
      theta = 6.0 + sx / r;
    }
  }

  theta *= M_PIf / 4.0;
  p.x = r * cosf(theta);
  p.y = r * sinf(theta);
}

TORCH_HOSTDEVICE void SampleHemisphereCosine(const float2& sample, float3& p)
{
  float2 uv;
  SampleDiskUniform(sample, uv);
  p.z = sqrtf(1 - dot(uv, uv));
  p.x = uv.x;
  p.y = uv.y;
}

} // namespace torch
