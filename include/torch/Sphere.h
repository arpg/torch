#pragma once

#include <torch/SingleGeometry.h>

namespace torch
{

class Sphere : public SingleGeometry
{
  public:

    Sphere(std::shared_ptr<Context> context);

    ~Sphere();
};

} // namespace torch