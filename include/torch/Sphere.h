#pragma once

#include <torch/Geometry.h>

namespace torch
{

class Sphere : public SingleGeometry
{
  public:

    Sphere(std::shared_ptr<Context> context);

    ~Sphere();
};

} // namespace torch