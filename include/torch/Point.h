#pragma once

#include <torch/Core.h>

namespace torch
{

struct Point
{
  Point();

  Point(float x, float y, float z);

  Point operator+(const Point& p) const;

  Point& operator+=(const Point& p);

  Point operator+(const Vector& v) const;

  Point& operator+=(const Vector& v);

  Point operator-(const Vector& v) const;

  Point& operator-=(const Vector& v);

  Vector operator-(const Point& p) const;

  Point operator*(float f) const;

  Point& operator*=(float f);

  Point operator/(float f) const;

  Point& operator/=(float f);

  float operator[](int i) const;

  float& operator[](int i);

  float x, y, z;
};

Point operator*(float f, const Point& p);

} // namespace torch