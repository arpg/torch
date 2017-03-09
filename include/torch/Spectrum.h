#pragma once

#include <torch/Core.h>
#include <torch/Vector.h>

namespace torch
{

class Spectrum
{
  public:

    Spectrum();

    float GetY() const;

    Vector GetXYZ() const;

    Vector GetRGB() const;

    Spectrum operator+(const Spectrum& a) const;

    Spectrum& operator+=(const Spectrum& a);

    Spectrum operator*(float a) const;

    Spectrum operator*(const Spectrum& a) const;

    Spectrum& operator*=(float a);

    Spectrum& operator*=(const Spectrum& a);

    Spectrum operator/(float a);

    Spectrum& operator/=(float a);

    static Spectrum FromXYZ(const Vector& xyz);

    static Spectrum FromXYZ(float x, float y, float z);

    static Spectrum FromRGB(const Vector& rgb);

    static Spectrum FromRGB(float r, float g, float b);

  protected:

    Vector m_rgb;
};

Spectrum operator*(float a, const Spectrum& b);

} // namespace torch