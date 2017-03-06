#include <torch/Vector.h>
#include <cmath>
#include <torch/Normal.h>

namespace torch
{

Vector::Vector() :
  x(0), y(0), z(0)
{
}

Vector::Vector(float x, float y, float z) :
  x(x), y(y), z(z)
{
}

Vector::Vector(const Normal& n) :
  x(n.x), y(n.y), z(n.z)
{

}

Vector Vector::operator-() const
{
  return Vector(-x, -y, -z);
}

Vector Vector::operator+(const Vector& v) const
{
  return Vector(x + v.x, y + v.y, z + v.z);
}

Vector& Vector::operator+=(const Vector& v)
{
  x += v.x;
  y += v.y;
  z += v.z;
  return *this;
}

Vector Vector::operator-(const Vector& v) const
{
  return Vector(x - v.x, y - v.y, z - v.z);
}

Vector& Vector::operator-=(const Vector& v)
{
  x -= v.x;
  y -= v.y;
  z -= v.z;
  return *this;
}

Vector Vector::operator*(float f) const
{
  return Vector(x * f, y * f, z * f);
}

Vector& Vector::operator*=(float f)
{
  x *= f;
  y *= f;
  z *= f;
  return *this;
}

Vector Vector::operator/(float f) const
{
  return Vector(x / f, y / f, z / f);
}

Vector& Vector::operator/=(float f)
{
  x /= f;
  y /= f;
  z /= f;
  return *this;
}

float Vector::operator*(const Vector& v) const
{
  return x * v.x + y * v.y + z * v.z;
}

Vector operator*(float f, const Vector& v)
{
  return v * f;
}

float Vector::operator[](int i) const
{
  return (&x)[i];
}

float&Vector::operator[](int i)
{
  return (&x)[i];
}

Vector Vector::Cross(const Vector& v) const
{
  const float xp = y * v.z - z * v.y;
  const float yp = z * v.x - x * v.z;
  const float zp = x * v.y - y * v.x;
  return Vector(xp, yp, zp);
}

float Vector::LengthSquared() const
{
  return (*this) * (*this);
}

float Vector::Length() const
{
  return std::sqrt(LengthSquared());
}

Vector Vector::Normalize() const
{
  const float length = Length();
  return (length < 1E-8) ? Vector() : (*this) / length;
}

} // namespace torch