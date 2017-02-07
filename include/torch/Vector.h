#pragma once

#include <torch/Core.h>

namespace torch
{

struct Vector
{
  Vector();

  Vector(float x, float y, float z);

  explicit Vector(const Normal& n);

  Vector operator-() const;

  Vector operator+(const Vector& v) const;

  Vector& operator+=(const Vector& v);

  Vector operator-(const Vector& v) const;

  Vector& operator-=(const Vector& v);

  Vector operator*(float f) const;

  Vector& operator*=(float f);

  Vector operator/(float f) const;

  Vector& operator/=(float f);

  float operator*(const Vector& v) const;

  float operator[](int i) const;

  float& operator[](int i);

  Vector Cross(const Vector& v) const;

  float LengthSquared() const;

  float Length() const;

  Vector Normalize() const;

  float x, y, z;
};

Vector operator*(float f, const Vector& v);

} // namespace torch