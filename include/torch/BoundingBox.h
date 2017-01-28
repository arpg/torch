#pragma once

#include <torch/Core.h>
#include <torch/Point.h>

namespace torch
{

struct BoundingBox
{
  BoundingBox();

  void Add(const Point& p);

  void Add(float x, float y, float z);

  void Add(const BoundingBox& b);

  Point bmin;

  Point bmax;
};

} // namespace torch