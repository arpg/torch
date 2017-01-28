#include <torch/Point.h>
#include <torch/Vector.h>

namespace torch
{

Point::Point() :
  x(0), y(0), z(0)
{
}

Point::Point(float x, float y, float z) :
  x(x), y(y), z(z)
{
}

Point Point::operator+(const Point& p) const
{
  return Point(x + p.x, y + p.y, z + p.z);
}

Point& Point::operator+=(const Point& p)
{
  x += p.x;
  y += p.y;
  z += p.z;
  return *this;
}

Point Point::operator+(const Vector& v) const
{
  return Point(x + v.x, y + v.y, z + v.z);
}

Point& Point::operator+=(const Vector& v)
{
  x += v.x;
  y += v.y;
  z += v.z;
  return *this;
}

Point Point::operator-(const Vector& v) const
{
  return Point(x - v.x, y - v.y, z - v.z);
}

Point& Point::operator-=(const Vector& v)
{
  x -= v.x;
  y -= v.y;
  z -= v.z;
  return *this;
}

Vector Point::operator-(const Point& p) const
{
  return Vector(x - p.x, y - p.y, z - p.z);
}

Point Point::operator*(float f) const
{
  return Point(x * f, y * f, z * f);
}

Point& Point::operator*=(float f)
{
  x *= f;
  y *= f;
  z *= f;
  return *this;
}

Point Point::operator/(float f) const
{
  return Point(x / f, y / f, z / f);
}

Point&Point::operator/=(float f)
{
  x /= f;
  y /= f;
  z /= f;
  return *this;
}

float Point::operator[](int i) const
{
  return (&x)[i];
}

float&Point::operator[](int i)
{
  return (&x)[i];
}

Point operator*(float f, const Point& p)
{
  return p * f;
}

} // namespace torch