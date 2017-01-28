#include <torch/Normal.h>
#include <cmath>
#include <torch/Vector.h>

namespace torch
{

Normal::Normal() :
  x(0), y(0), z(0)
{
}

Normal::Normal(float x, float y, float z) :
  x(x), y(y), z(z)
{
}

Normal::Normal(const Vector& v) :
  x(v.x), y(v.y), z(v.z)
{
}

Normal Normal::operator+(const Normal& v) const
{
  return Normal(x + v.x, y + v.y, z + v.z);
}

Normal& Normal::operator+=(const Normal& v)
{
  x += v.x;
  y += v.y;
  z += v.z;
  return *this;
}

Normal Normal::operator-(const Normal& v) const
{
  return Normal(x - v.x, y - v.y, z - v.z);
}

Normal& Normal::operator-=(const Normal& v)
{
  x -= v.x;
  y -= v.y;
  z -= v.z;
  return *this;
}

Normal Normal::operator*(float f) const
{
  return Normal(x * f, y * f, z * f);
}

Normal& Normal::operator*=(float f)
{
  x *= f;
  y *= f;
  z *= f;
  return *this;
}

Normal Normal::operator/(float f) const
{
  return Normal(x / f, y / f, z / f);
}

Normal& Normal::operator/=(float f)
{
  x /= f;
  y /= f;
  z /= f;
  return *this;
}

float Normal::operator*(const Normal& v) const
{
  return x * v.x + y * v.y + z * v.z;
}

float Normal::operator[](int i) const
{
  return (&x)[i];
}

float&Normal::operator[](int i)
{
  return (&x)[i];
}

float Normal::LengthSquared() const
{
  return (*this) * (*this);
}

float Normal::Length() const
{
  return std::sqrt(LengthSquared());
}

Normal Normal::Normalize() const
{
  return (*this) / Length();
}

Normal operator*(float f, const Normal& p)
{
  return p * f;
}

} // namespace torch