#pragma once

#include <torch/Core.h>

namespace torch
{

struct Normal
{
  Normal();

  Normal(float x, float y, float z);

  explicit Normal(const Vector& v);

  Normal operator+(const Normal& v) const;

  Normal& operator+=(const Normal& v);

  Normal operator-(const Normal& v) const;

  Normal& operator-=(const Normal& v);

  Normal operator*(float f) const;

  Normal& operator*=(float f);

  Normal operator/(float f) const;

  Normal& operator/=(float f);

  float operator*(const Normal& v) const;

  float operator[](int i) const;

  float& operator[](int i);

  float LengthSquared() const;

  float Length() const;

  Normal Normalize() const;

  float x, y, z;
};

} // namespace torch