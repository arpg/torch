#pragma once

namespace torch
{

class Vector;

struct Point
{
  Point();

  Point(float x, float y, float z);

  Point operator+(const Vector& v) const;

  Point& operator+=(const Vector& v);

  Point operator-(const Vector& v) const;

  Point& operator-=(const Vector& v);

  Vector operator-(const Point& p) const;

  float operator[](int i) const;

  float& operator[](int i);

  float x, y, z;
};

} // namespace torch