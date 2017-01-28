#pragma once

#include <torch/Core.h>
#include <torch/Point.h>

namespace torch
{

struct BoundingBox
{
  BoundingBox();

  void Union(const Point& p);

  void Union(float x, float y, float z);

  void Union(const BoundingBox& b);

  float GetRadius() const;

  Point bmin;

  Point bmax;
};

} // namespace torch