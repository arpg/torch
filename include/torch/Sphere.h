#pragma once

#include <torch/SingleGeometry.h>

namespace torch
{

class Sphere : public SingleGeometry
{
  public:

    Sphere(std::shared_ptr<Context> context);

    BoundingBox GetBounds(const Transform& transform) override;
};

} // namespace torch