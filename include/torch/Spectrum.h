#pragma once

#include <optix_math.h>

namespace torch
{

class Spectrum
{
  public:

    Spectrum();

    ~Spectrum();

    float3 GetRGB() const;

    static Spectrum FromRGB(const float3& rgb);

    static Spectrum FromRGB(float r, float g, float b);

  protected:

    float3 m_rgb;
};

} // namespace torch