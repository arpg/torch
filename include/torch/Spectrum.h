#pragma once

#include <optix_math.h>

namespace torch
{

class Spectrum
{
  public:

    Spectrum();

    ~Spectrum();

    float GetY() const;

    float3 GetXYZ() const;

    float3 GetRGB() const;

    Spectrum operator*(float a) const;

    Spectrum operator*(const Spectrum& a) const;

    Spectrum& operator*=(float a);

    Spectrum& operator*=(const Spectrum& a);

    static Spectrum FromXYZ(const float3& xyz);

    static Spectrum FromXYZ(float x, float y, float z);

    static Spectrum FromRGB(const float3& rgb);

    static Spectrum FromRGB(float r, float g, float b);

  protected:

    float3 m_rgb;
};

Spectrum operator*(float a, const Spectrum& b);

} // namespace torch