#include <torch/Spectrum.h>

namespace torch
{

Spectrum::Spectrum() :
  m_rgb(Vector(0, 0, 0))
{
}

float Spectrum::GetY() const
{
  return 0.212671f * m_rgb.x + 0.715160f * m_rgb.y + 0.072169f * m_rgb.z;
}

Vector Spectrum::GetXYZ() const
{
  Vector xyz;
  xyz.x = 0.412453f * m_rgb.x + 0.357580f * m_rgb.y + 0.180423f * m_rgb.z;
  xyz.y = 0.212671f * m_rgb.x + 0.715160f * m_rgb.y + 0.072169f * m_rgb.z;
  xyz.z = 0.019334f * m_rgb.x + 0.119193f * m_rgb.y + 0.950227f * m_rgb.z;
  return xyz;
}

Vector Spectrum::GetRGB() const
{
  return m_rgb;
}

Spectrum Spectrum::operator+(const Spectrum& a) const
{
  Spectrum result(*this);
  result += a;
  return result;
}

Spectrum& Spectrum::operator+=(const Spectrum& a)
{
  m_rgb = m_rgb + a.m_rgb;
  return *this;
}

Spectrum Spectrum::operator*(float a) const
{
  Spectrum result(*this);
  result *= a;
  return result;
}

Spectrum Spectrum::operator*(const Spectrum& a) const
{
  Spectrum result(*this);
  result *= a;
  return result;
}

Spectrum& Spectrum::operator*=(float a)
{
  m_rgb = m_rgb * a;
  return *this;
}

Spectrum& Spectrum::operator*=(const Spectrum& a)
{
  m_rgb.x *= a.m_rgb.x;
  m_rgb.y *= a.m_rgb.y;
  m_rgb.z *= a.m_rgb.z;
  return *this;
}

Spectrum Spectrum::operator/(float a)
{
  Spectrum result(*this);
  result /= a;
  return result;
}

Spectrum&Spectrum::operator/=(float a)
{
  m_rgb = m_rgb / a;
  return *this;
}

Spectrum operator*(float a, const Spectrum& b)
{
  Spectrum result(b);
  result *= a;
  return result;
}

Spectrum Spectrum::FromXYZ(const Vector& xyz)
{
  return FromXYZ(xyz.x, xyz.y, xyz.z);
}

Spectrum Spectrum::FromXYZ(float x, float y, float z)
{
  Spectrum result;
  result.m_rgb.x =  3.240479f * x - 1.537150f * y - 0.498535f * z;
  result.m_rgb.y = -0.969256f * x + 1.875991f * y + 0.041556f * z;
  result.m_rgb.z =  0.055648f * x - 0.204043f * y + 1.057311f * z;
  return result;
}

Spectrum Spectrum::FromRGB(const Vector& rgb)
{
  Spectrum result;
  result.m_rgb = rgb;
  return result;
}

Spectrum Spectrum::FromRGB(float r, float g, float b)
{
  return FromRGB(Vector(r, g, b));
}

} // namespace torch
