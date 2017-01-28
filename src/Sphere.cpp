#include <torch/Sphere.h>
#include <torch/BoundingBox.h>

namespace torch
{

Sphere::Sphere(std::shared_ptr<Context> context) :
  SingleGeometry(context, "Sphere")
{
}

BoundingBox Sphere::GetBounds(const Transform& transform)
{
  BoundingBox bounds;
  bounds.Union(-0.5, -0.5, -0.5);
  bounds.Union(+0.5, +0.5, +0.5);
  return transform * m_transform * bounds;
}

} // namespace torch