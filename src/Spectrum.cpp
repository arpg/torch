#include <torch/Spectrum.h>

namespace torch
{

Spectrum::Spectrum() :
  m_rgb(make_float3(0, 0, 0))
{
}

Spectrum::~Spectrum()
{
}

float3 Spectrum::GetRGB() const
{
  return m_rgb;
}

Spectrum Spectrum::FromRGB(const float3& rgb)
{
  Spectrum result;
  result.m_rgb = rgb;
  return result;
}

Spectrum Spectrum::FromRGB(float r, float g, float b)
{
  return FromRGB(make_float3(r, g, b));
}

} // namespace torch