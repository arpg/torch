#include <torch/Sphere.h>

namespace torch
{

Sphere::Sphere(std::shared_ptr<Context> context) :
  SingleGeometry(context, "Sphere")
{
}

Sphere::~Sphere()
{
}

} // namespace torch